import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from depth_renderer import DepthRenderer
from omegaconf import DictConfig
from third_party.lietorch import SE3
import hydra
import os
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
from generation.common import PoseLoss, SDFLoss
from scipy.spatial.transform import Rotation as R
from PIL import Image
from generation.common import create_extrinsics_matrix, create_vectors_from_matrix

def pose_process(output_file):
    extrinsics_list = []
    vec_list = []
    axis_list = []
    id_list = []
    with open(output_file, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip() or line.startswith("id"):
                continue
            # Parse the line, assuming space-delimited values
            values = line.split()
            branch_idx = int(values[0])  # branch id
            x, y, z = map(float, values[1:4])  # Translation components
            q_x, q_y, q_z, q_w = map(float, values[4:8])  # Quaternion components
            majorAxis, minorAxis = map(float, values[8:10])  # Major and minor axis
            parent_idx, gen_idx = map(int, values[10:12])  # Parent and generation indices
            # Create the 4x4 matrix
            extrinsics = create_extrinsics_matrix(x, y, z, q_x, q_y, q_z, q_w)
            extrinsics_list.append(extrinsics)
            vec_list.append(torch.tensor(create_vectors_from_matrix(extrinsics)))
            axis_list.append([majorAxis, minorAxis])
            id_list.append([branch_idx, parent_idx, gen_idx])
    return extrinsics_list, vec_list, axis_list, id_list

def mat2vec(poses):
        """ 
        Args:
            pose (torch.Tensor): Nx4x4 homogenous matrix
        Return:
            numpy.ndarray: Nx7d pose        
        """
        quat = R.from_matrix(poses[:, :3, :3].cpu().numpy()).as_quat().astype(np.float32)
        return np.hstack((poses[:, :3, 3].cpu().numpy().astype(np.float32), quat))

def _skew(v):
    # v: (..., 3)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O,   -vz,  vy], dim=-1),
        torch.stack([vz,   O,  -vx], dim=-1),
        torch.stack([-vy, vx,   O], dim=-1),
    ], dim=-2)

def new_poses(
    pose: torch.Tensor,
    t_noise: tuple[float, float] = (0.0, 0.015),
    rot_noise: tuple[float, float] = (0.0, 45.0),
    num_new_poses: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generates perturbed poses for a batch of 4x4 poses.

    Args:
        pose (torch.Tensor): (B, 4, 4)
        t_noise (tuple[float, float]): translation noise range (min, max)
        rot_noise (tuple[float, float]): rotation angle range (deg) (min, max)
        num_new_poses (int): number of new poses to generate per input pose
        generator (torch.Generator): random generator

    Returns:
        perturbed pose (B * num_new_poses, 4, 4)
    """
    # If single pose, unsqueeze to (1, 4, 4)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    B = pose.shape[0]
    device = pose.device
    dtype = pose.dtype

    # Repeat poses for num_new_poses
    pose_repeat = pose.repeat_interleave(num_new_poses, dim=0)
    Rot = pose_repeat[:, :3, :3]    # (B * num_new_poses,3,3)
    t   = pose_repeat[:, :3, 3]     # (B * num_new_poses,3)
    big_B = pose_repeat.shape[0]

    # --- translation perturbation (S^2 in batch)
    direction = torch.randn(big_B, 3, device=device, dtype=dtype, generator=generator)   # (big_B,3)
    direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-12)
    t_noise_min, t_noise_max = t_noise if isinstance(t_noise, (tuple, list)) else (0.0, t_noise)
    dist = torch.empty(big_B, 1, device=device, dtype=dtype).uniform_(t_noise_min, t_noise_max, generator=generator)  # (big_B,1)
    noise = direction * dist                            # (big_B,3)
    t_perturbed = t + noise                             # (big_B,3)

    # --- rotation perturbation with geodesic control (per batch)
    axis = torch.randn(big_B, 3, device=device, dtype=dtype, generator=generator)       # (big_B,3)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-12)                             # (big_B,3)

    # Rotation angle in degree [rot_noise_min, rot_noise_max]
    rot_noise_min, rot_noise_max = rot_noise if isinstance(rot_noise, (tuple, list)) else (0.0, rot_noise)
    # Uniformly sample desired max rotation per pose in deg
    theta_max_deg = torch.empty(big_B, device=device, dtype=dtype).uniform_(rot_noise_min, rot_noise_max, generator=generator)
    theta_max_deg = torch.clamp(theta_max_deg, 0.0, 180.0)
    theta_max_rad = theta_max_deg * (torch.pi / 180.0)

    u = torch.rand(big_B, device=device, dtype=dtype, generator=generator)             # (big_B,)
    angle = 2.0 * torch.asin(torch.sin(theta_max_rad / 2.0) * (u.clamp(1e-12, 1.0)) ** (1.0 / 3.0))  # (big_B,)

    # axis-angle to rotation matrix in batch
    def _batch_axis_angle_to_R(axis, angle):
        # axis: (big_B,3), angle: (big_B,)
        K = _skew(axis)  # (big_B,3,3)
        I = torch.eye(3, dtype=axis.dtype, device=axis.device).unsqueeze(0).expand(axis.size(0), -1, -1)  # (big_B,3,3)
        s = torch.sin(angle).view(-1, 1, 1)
        c = torch.cos(angle).view(-1, 1, 1)
        return I + s * K + (1.0 - c) * (K @ K) # (big_B,3,3)

    R_noise = _batch_axis_angle_to_R(axis, angle)  # (big_B,3,3)
    R_perturbed = torch.bmm(R_noise, Rot)          # (big_B,3,3)

    pose_out = torch.eye(4, device=device, dtype=dtype).repeat(big_B, 1, 1)
    pose_out[:, :3, :3] = R_perturbed
    pose_out[:, :3, 3]  = t_perturbed
    return pose_out

def sample_poses(cfg,
    depth_renderer,
    base_pose: SE3,
    t: tuple[float, float] = (0.0, 0.015),
    rot: tuple[float, float] = (0.0, 45.0),
    num_samples: int = 5,
):
    """
    Sample camera poses around `base_pose` on manifold.
    """
   
    sampled_poses = []
    l_p = PoseLoss(cfg)
    sdfLoss = SDFLoss(cfg)


    def pose_filter(poses, cfg=cfg):
        # Filter out poses in bad render quality
        [depths] = depth_renderer(poses)
        # # blank filter
        # blank_ratio = cfg.data.blank_filter
        # near_ratio = cfg.data.near_filter
        sdf_values = sdfLoss(poses, mean=False)
        mask = torch.ones(poses.shape[0], device=poses.device, dtype=torch.bool)
        # Vectorized blank filter
        blank_mask = sdf_values > 1

        # Vectorized close filter (median per image)
        medians = depths.median(dim=3)[0].median(dim=2)[0].median(dim=1)[0]
        near_mask = medians < 1e-4
        
        # Combine all masks
        mask &= ~blank_mask
        mask &= ~near_mask
        return poses[mask]

    def sample_by_quantiles(new_SE3, base_pose, K, l_p):
        # 1) Compute scalar residuals D \in [N]
        base = base_pose.matrix().reshape(1, 4, 4).repeat(new_SE3.shape[0], 1, 1)
        D = l_p(new_SE3, base).reshape(-1)            # ensure [N]
        D_np = D.detach().cpu().numpy()

        # 2) Quantile edges
        edges = np.quantile(D_np, np.linspace(0, 1, K + 1))
        edges = np.unique(edges)  # handle ties
        if len(edges) - 1 < K:  # many duplicates, fallback to linspace
            lo, hi = D_np.min(), D_np.max()
            edges = np.linspace(lo, hi, K + 1)

        # 3) One index per bin (closest to bin center)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.argmin(np.abs(D_np[:, None] - centers[None, :]), axis=0)
        idx = np.unique(idx)  # dedup

        # 4) If duplicates reduced count, top up by farthest-point sampling in 1D, with limit to prevent endless loop
        if len(idx) < K:
            sel = set(idx.tolist())
            available = set(range(D_np.shape[0]))
            max_attempts = D_np.shape[0]  # Can't sample more than what's available
            attempts = 0
            while len(sel) < K and len(sel) < len(available) and attempts < max_attempts:
                if not sel:  # If selection is empty, pick a random point
                    cand = np.random.randint(0, D_np.shape[0])
                else:
                    remaining = list(available - sel)
                    if not remaining:
                        break  # No more candidates, break to avoid endless loop
                    dist = np.min(np.abs(D_np[np.array(remaining)][:, None] - D_np[list(sel)][None, :]), axis=1)
                    cand = remaining[int(np.argmax(dist))]
                sel.add(cand)
                attempts += 1
            idx = np.array(sorted(sel))
            # if still fewer than K due to too few unique candidates, just return as many as possible
            if len(idx) == 0:
                return new_SE3[:1]
            if len(idx) < K:
                # fallback: fill up by random sample without replacement of remaining indices
                rest = np.array(list(available - set(idx)))
                num_fill = min(K - len(idx), len(rest))
                if num_fill > 0 and len(rest) > 0:
                    add_idx = np.random.choice(rest, size=num_fill, replace=False)
                    idx = np.concatenate([idx, np.sort(add_idx)])
                idx = np.unique(idx)
        if len(idx) > K:
            idx = idx[:K]  # Cannot guarantee more than K, just truncate
        idx_t = torch.as_tensor(idx, device=D.device, dtype=torch.long)
        return new_SE3[idx_t]  # [K,4,4]


    new_SE3 = new_poses(base_pose.matrix(), t, rot, num_new_poses=100)
    new_SE3 = pose_filter(new_SE3)
    if new_SE3.shape[0] == 0:
        return None
    sampled_poses = sample_by_quantiles(new_SE3, base_pose, num_samples, l_p)
    # res = l_p(new_SE3, base_pose.matrix().repeat(new_SE3.shape[0], 1, 1))
    # # Plot the distribution of res
    # res_np = res.detach().cpu().numpy()

    # # Downsample poses to ensure even distribution of residuals
    # num_bins = num_samples  # Number of bins for even distribution
    # bin_edges = np.linspace(res_np.min(), res_np.max(), num_bins + 1)
    # bin_indices = np.digitize(res_np, bin_edges) - 1

    # # Select one pose per bin to ensure even distribution
    # selected_indices = []
    # for bin_idx in range(num_bins):
    #     indices_in_bin = np.where(bin_indices == bin_idx)[0]
    #     if len(indices_in_bin) > 0:
    #         selected_indices.append(np.random.choice(indices_in_bin))

    # # Filter poses and residuals based on selected indices
    # sampled_poses = [new_SE3[i] for i in selected_indices]
    # if len(sampled_poses) == 0:
    #     print("No poses sampled.")
    #     return None
    # # Convert sampled poses to an nx4x4 tensor
    # sampled_poses = torch.stack([pose for pose in sampled_poses], dim=0)
    return sampled_poses


@hydra.main(version_base=None, config_path="../configs", config_name="generation")

def main(cfg: DictConfig):
    root_path = Path(cfg.data_root)
    centerline_file = root_path / f"centerlines.txt"
    out_path = root_path / "train"
    eval_path = root_path / "eval"
    depth_path = out_path / "depths"
    color_path = out_path / "images"
    pose_path = out_path / "poses"

    poses_list, vec_poses_list, axis_list, id_list = pose_process(centerline_file)
    depth_renderer = DepthRenderer(cfg).to(cfg.device)
    poses_for_render = [SE3.InitFromVec(vec_poses_list[i].to(cfg.device)) for i in range(len(vec_poses_list))]
    leaf_ids = np.load(root_path / 'leaf_ids.npy')

    if cfg.data.train_data:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(color_path, exist_ok=True)
        os.makedirs(pose_path, exist_ok=True)
    
        print('generate train data...')
        with open(pose_path / 'poses.txt', "w") as tum_file:
            tum_file.write("# frame_index tx ty tz qx qy qz qw branch_id parent_id gen_id\n")
            for i, [pose, [majorAxis, minorAxis],[branch_idx, parent_idx, gen_idx]] in tqdm(enumerate(zip(poses_for_render, axis_list, id_list))):
                # skip the tip branchs
                if branch_idx in leaf_ids:
                    continue
                # if i < 1100:
                #     continue
                pose = pose.to(cfg.device)
                if cfg.data.color_render:
                    [render_depth, rgb] = depth_renderer(pose.matrix().unsqueeze(0), color_render=True)
                else:
                    [render_depth] = depth_renderer(pose.matrix())
       
                skip = False
                while torch.count_nonzero(render_depth == 0.1) / render_depth.numel() > cfg.data.blank_filter and i >= 120:
                    sample_pose = sample_poses(cfg, depth_renderer, base_pose=pose, num_samples=20, t=0.001, rot=1)
                    if sample_pose is None:
                        print(f"frame {i} skip.")
                        skip = True # not likely to have a good render even sample around
                        break

                    pose = sample_pose[0]
                    pose = SE3.InitFromVec(torch.from_numpy(mat2vec(pose[None]))).to(cfg.device)
                    if cfg.data.color_render:
                        [render_depth, rgb] = depth_renderer(pose.matrix(), color_render=True)
                    else:
                        [render_depth] = depth_renderer(pose.matrix())

                if not skip:
                    if cfg.data.color_render:
                        rgb_np = (rgb.clone().permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(color_path / f"{i}.png")
                            
                    render_depth_np = render_depth.clone().squeeze(1).cpu().numpy()
                    np.save(depth_path / f"{i}.npy", render_depth_np[0, ...])
                        
                    # save pose
                    tum_pose = pose.vec().clone().detach().squeeze().cpu().numpy()
                    tum_file.write(f'{i} {tum_pose[0]} {tum_pose[1]} {tum_pose[2]} {tum_pose[3]} {tum_pose[4]} {tum_pose[5]} {tum_pose[6]} {branch_idx} {parent_idx} {gen_idx}\n')
                    
                    nearby_poses = sample_poses(cfg, depth_renderer, base_pose=pose, num_samples=cfg.data.train_num_samples, t=cfg.data.train_radius, rot=cfg.data.train_rot_degs)
                    if nearby_poses is None:
                        print(f"No nearby frames sampled for {i}.")
                        continue
                    tum_poses = mat2vec(nearby_poses)

                    sub_pose_path = pose_path / f"{i}"
                    os.makedirs(sub_pose_path, exist_ok=True)
                    with open(sub_pose_path / "poses.txt", "w") as sub_tum_file:
                        sub_tum_file.write("# frame_index tx ty tz qx qy qz qw\n")
                        if cfg.data.color_render:
                            [render_depth, rgb] = depth_renderer(nearby_poses, color_render=True)
                        else:
                            [render_depth] = depth_renderer(nearby_poses)
                        
                        render_depth_np = render_depth.clone().squeeze(1).cpu().numpy()
                        
                        os.makedirs(depth_path / f"{i}", exist_ok=True)
                        for j, render_depth in enumerate(render_depth_np):
                            np.save(depth_path / f"{i}" / f"{j}.npy", render_depth)
                            tum_pose = tum_poses[j]
                            sub_tum_file.write(f'{j} {tum_pose[0]} {tum_pose[1]} {tum_pose[2]} {tum_pose[3]} {tum_pose[4]} {tum_pose[5]} {tum_pose[6]}\n')
                        
                        if cfg.data.color_render:
                            rgb_np = (rgb.clone().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                            os.makedirs(color_path / f"{i}", exist_ok=True)
                            for j, img in enumerate(rgb_np):
                                Image.fromarray(img).save(color_path / f"{i}" / f"{j}.png")

    # The benchmark pairs come from the renders above (train/) via generate_pairs.py.


if __name__ == "__main__":
    main()
