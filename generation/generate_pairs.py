import os
import json
from collections import defaultdict
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from third_party.lietorch import SE3
from tqdm import tqdm
from generation.common import PoseLoss, covis_from_tri_ids
from depth_renderer import DepthRenderer
import time
NO_PARENT = "<no_parent>"

def build_tree(nodes):
    branch_p = {}
    branch_list = defaultdict(list)
    for node in nodes:
        branch_list[node['branch']].append(node)
        branch_p[node['branch']] = node['parent']

    for k in branch_list.keys():
        branch_list[k] = sorted(branch_list[k], key=lambda x: x['id'])
        
    parent = {}
    for lst in branch_list.values():
        for i in range(1, len(lst)):
            parent[lst[i]['id']] = lst[i - 1]['id']

        p_branch = lst[0]['parent']
        if p_branch not in branch_list:
            parent[lst[0]['id']] = NO_PARENT
            continue

        p_lst = branch_list[p_branch]
        parent[lst[0]['id']] = p_lst[-1]['id']

    return parent

def generate_rel_list(pose_dict, cfg):
    l_pose = PoseLoss(cfg)
    N = cfg.data.pairs_per_node
    SIGMA = cfg.data.pair_range
    fx = cfg.cam.intrinsics['fx']
    fy = cfg.cam.intrinsics['fy']
    cx = cfg.cam.intrinsics['cx']
    cy = cfg.cam.intrinsics['cy']
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=cfg.device)
    print(f"Generating {N} pairs for each node with range {SIGMA}...")
    pair_id = 0
  
    rel_list = []
    candidates = []
    key_bag = defaultdict(dict)

    for k, pose_info in pose_dict.items():
        key = k.split("_")[1]
        if key not in key_bag:
            key_bag[key]["items"] = []

        key_bag[key]['items'].append((k, pose_info))
        key_bag[key]['branch'] = pose_info['branch']
        key_bag[key]['parent'] = pose_info['parent']
        key_bag[key]['gen'] = pose_info['gen']

    tree_nodes = [ {'id': k, 'branch': v['branch'], 'parent': v['parent']} for k, v in key_bag.items() ]
    parent = build_tree(tree_nodes)

    for p1_k in tqdm(key_bag.keys()):
        if key_bag[p1_k]['gen'] >= 9:
            continue
        chunk = {'src_poses': [], 'target_poses': [], 'p2s': [], 'p1s': []}
            
        for _ in range(N):
            p1 = random.choice(key_bag[p1_k]['items'])[0]
        
            ofs = abs(round(np.random.normal(0, SIGMA, 1)[0]))
            p2_k = p1_k
            for _ in range(ofs):
                if parent[p2_k] == NO_PARENT:
                    break

                p2_k = parent[p2_k]

            p2 = random.choice(key_bag[p2_k]['items'])[0]
            src_pose = pose_dict[p1]['pose']
            target_pose = pose_dict[p2]['pose']
            
            # accumulate poses for batch processing
            chunk['src_poses'].append(torch.as_tensor(src_pose, device=cfg.device))
            chunk['target_poses'].append(torch.as_tensor(target_pose, device=cfg.device))
            chunk['p2s'].append(p2)
            chunk['p1s'].append(p1)
        
        src_poses = SE3(torch.stack(chunk['src_poses'], dim=0))
        target_poses = SE3(torch.stack(chunk['target_poses'], dim=0))
        pose_dis = l_pose(src_poses.matrix(), target_poses.matrix())
        pose_grad = pose_dist_grad(src_poses, target_poses)
        [depth_1,_, tri_ids_1] = renderer(src_poses.matrix(), color_render=True, return_tri_ids=True)
        [depth_2,_, tri_ids_2] = renderer(target_poses.matrix(), color_render=True, return_tri_ids=True)
        s_faces = covis_from_tri_ids(tri_ids_1, tri_ids_2)
        
        for i in range(len(chunk['src_poses'])):
            # Only keep pairs with sufficient overlap
            if s_faces[i] >= 0.2:
                candidates.append({
                    'pair_id': pair_id,
                    'anchor': chunk['p2s'][i],
                    'sample': chunk['p1s'][i],
                    'pose_grad': pose_grad[i],
                    's_faces': float(s_faces[i].item()) if torch.is_tensor(s_faces[i]) else float(s_faces[i]),
                    'pose_dis': float(pose_dis[i].detach().cpu().item()),  # <-- record scalar distance
                })
                pair_id += 1
    
    print(f"Generated {len(candidates)}/{pair_id} pairs")

    # --- Build level edges once, globally -------------------------------------
    ds = torch.tensor([c['pose_dis'] for c in candidates], device=cfg.device)
    if EDGES is not None and len(EDGES) == L + 1:
        edges = torch.tensor(EDGES, device=cfg.device, dtype=ds.dtype)
    else:
        # quantile-based levels: balanced counts
        qs = torch.linspace(0, 1, L + 1, device=cfg.device)
        edges = torch.quantile(ds, qs)
        # widen ends slightly to include extremes
        eps = 1e-8
        edges[0]  = torch.min(edges[0],  ds.min()) - eps
        edges[-1] = torch.max(edges[-1], ds.max()) + eps

    # --- Assign levels and finalize rel_list ----------------------------------
    # Cut the very long tail (outliers in pose_dis)
    clip_quantile = 0.99
    max_pose_dis = torch.quantile(ds, clip_quantile).item()
    cut = edges[1:-1]  # internal boundaries
    level_counts = [0] * L
    for c in candidates:
        # Skip pairs with pose_dis in the very long tail
        if c['pose_dis'] > max_pose_dis:
            continue
        d = torch.tensor(c['pose_dis'], device=ds.device)
        lvl = int(torch.bucketize(d, cut, right=False).item()) + 1  # 1..L
        c['level'] = lvl
        rel_list.append(c)
        level_counts[lvl - 1] += 1

    print(f"Generated {len(rel_list)} pairs. Per-level counts: {level_counts}")
    return rel_list

def pose_dist_grad(src_pose, target_pose, return_steps=False, eps=1e-8):
    """
    Generate pose 'gradient' labels as SE(3) residual directions using lietorch.

    Args:
        src_pose:    [4,4] or [B,4,4] tensor/array-like (current pose)
        target_pose: [4,4] or [B,4,4] tensor/array-like (target pose)
        return_steps: if True, also return rotation/translation magnitudes (radians, meters if metric)
        eps:         small constant for numeric stability

    Returns:
        If B==1: list of 6 floats [t_dir(3), r_dir(3)]
        If B>1:  list of lists for each batch element
        If return_steps: also returns (rot_step, trans_step) tensors
    """
    # --- to tensor, preserve device/dtype ---
    device = (src_pose.device if isinstance(src_pose, torch.Tensor)
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = (src_pose.dtype if isinstance(src_pose, torch.Tensor) else torch.float32)

    # T_src = SE3(torch.as_tensor(src_pose, dtype=dtype, device=device))
    # T_tgt = SE3(torch.as_tensor(target_pose, dtype=dtype, device=device))
    T_src = src_pose
    T_tgt = target_pose
    # --- residual twist: xi = log(T_src^{-1} * T_tgt)  ---
    T_rel = T_src.inv() * T_tgt           # SE3
    xi = T_rel.log()                      # [B,6] -> [w(0:3), v(3:6)] in lietorch convention

    # Split and normalize to get unit directions
    r = xi[:,3:]                         # rotation (axis-angle)
    t = xi[:,:3]                         # translation
    r_step = r.norm(dim=-1, keepdim=True) # radians
    t_step = t.norm(dim=-1, keepdim=True) # meters (if metric) or scale units

    r_dir = r / (r_step + eps)
    t_dir = t / (t_step + eps)
    # Match your previous return layout: [t_grad, r_grad] and squeeze to list for B=1
    out_dirs = torch.cat([t_dir, r_dir], dim=-1)  # [B,6], order: t then r

    if return_steps:
        return out_dirs.squeeze(0).tolist(), [r_step.squeeze(0).item(), t_step.squeeze(0).item()]
    else:
        return out_dirs.squeeze(0).tolist()

def plot_pose_dis_levels(rel_list, L=5, edges=None, bins=60, title="pose_dis distribution", save_path=None):
    """
    Plot the distribution of pose_dis and the counts per difficulty level.

    Args:
        rel_list: list of dicts, each with at least {'pose_dis': float, ...}
                  (optional) {'level': int in [1..L]}
        L:        number of levels if 'level' not provided (default 5)
        edges:    optional array-like of length L+1 with level boundaries for pose_dis
                  (if None and 'level' missing, edges are computed by quantiles)
        bins:     histogram bins for pose_dis
        title:    chart title
        save_path: if not None, save the figure to this path
    """
    if not rel_list:
        print("Nothing to plot: rel_list is empty.")
        return

    pose_dis = np.asarray([x["pose_dis"] for x in rel_list], dtype=float)

    # Use provided levels if present; otherwise assign by edges/quantiles
    has_levels = ("level" in rel_list[0])
    if has_levels:
        levels = np.asarray([int(x["level"]) for x in rel_list], dtype=int)
        L = int(levels.max())
        # If no edges provided, we won't draw separators in the histogram
        level_edges = np.asarray(edges, dtype=float) if edges is not None else None
    else:
        if edges is None:
            # Balanced levels via quantiles
            qs = np.linspace(0, 1, L + 1)
            level_edges = np.quantile(pose_dis, qs)
            # widen a hair so min/max are included
            eps = 1e-12
            level_edges[0]  = min(level_edges[0],  pose_dis.min()) - eps
            level_edges[-1] = max(level_edges[-1], pose_dis.max()) + eps
        else:
            level_edges = np.asarray(edges, dtype=float)
            assert len(level_edges) == L + 1, "edges must have length L+1"
        # Assign 1..L
        cut = level_edges[1:-1]
        levels = np.digitize(pose_dis, cut, right=False) + 1

    # Counts per level (1..L)
    counts = np.bincount(levels, minlength=L + 1)[1:]

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    # Top: pose_dis histogram (+ optional vertical level edges)
    ax = axes[0]
    ax.hist(pose_dis, bins=bins, alpha=0.8)
    ax.set_xlabel("pose_dis")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if level_edges is not None:
        for e in level_edges:
            ax.axvline(e, linestyle="--", linewidth=1)

    # Bottom: bar chart of per-level counts
    ax = axes[1]
    xs = np.arange(1, L + 1)
    ax.bar(xs, counts, width=0.8)
    ax.set_xlabel("level")
    ax.set_ylabel("#samples")
    ax.set_xticks(xs)
    for x, c in zip(xs, counts):
        ax.text(x, c, str(int(c)), ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(cfg: DictConfig):
    global renderer
    renderer = DepthRenderer(cfg).to(cfg.device)
    # load json
    json_file = os.path.join(cfg.data_root, "train", "pose_dict.json")
    assert os.path.exists(json_file)
    
    # l_pose = PoseLoss(cfg)
    global L, EDGES
    L = cfg.data.num_levels
    # EDGES = [0.0, 0.4, 0.8, 1.6]
    EDGES = None

    data_dict = {}
    with open(json_file, "r") as f:
        pose_dict = json.load(f)
        all_keys = [ key for key, _ in pose_dict.items() ]
        data_dict["abs"] = all_keys
        data_dict["rel"] = generate_rel_list(pose_dict, cfg)
        plot_pose_dis_levels(rel_list=data_dict["rel"], L=L, edges=EDGES, save_path=f'{cfg.data_root}_pose_dist_levels.png')    
        save_json = os.path.join(cfg.data_root, "train", "train_src.json")
        with open(save_json, "w") as f:
            json.dump(data_dict, f)

if __name__ == "__main__":
    main()