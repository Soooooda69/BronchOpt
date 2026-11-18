"""Evaluate a relative-pose model on the BronchOpt synthetic benchmark.

For every pair in ``eval_src.json`` the model gets two images:

* ``sim_rgb``: a render of the CT mesh at the **sample (initial) pose**, and
* ``rgb``: the CycleGAN image of the **anchor (target) view**,

both ``[B,3,H,W]`` in [0,1] at ``cfg.model.input_dim``. It returns the relative transform
``T_rel`` ([B,4,4]) and the prediction is ``T_pred = T_init @ T_rel``. Poses are
camera-to-world in metres; the anchor pose is the ground truth.

Replace :class:`IdentityModel` in :func:`build_model` with your model. The identity model
predicts no correction, so its numbers are the "Init" baseline.
"""
import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoaders import PoseEvalDataset, sequence_roots
from depth_renderer import DepthRenderer
from metrics import (FAILED, MeshSDF, ScaleInvariantDepthLoss, dispSimilarity, is_success,
                     normal_correlation, rotation_error_rad, translation_error_mm)

METHODS = ('ours', 'init')
DEPTH_METRICS = ('ds', 'nc', 'si')
POSE_METRICS = ('t_mm', 'rot_rad')


class IdentityModel(nn.Module):
    """Placeholder: predicts no correction, so the prediction equals the initial pose."""

    def forward(self, sim_rgb, rgb):
        return torch.eye(4, device=rgb.device).expand(rgb.shape[0], 4, 4)


def build_model(cfg):
    # TODO: replace with your model, e.g.
    # model = YourPoseModel(cfg)
    # model.load_state_dict(torch.load(cfg.model.checkpoint, map_location=cfg.device)['model_state_dict'])
    return IdentityModel()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pose_vec_to_matrix(vec):
    """[B,7] (tx, ty, tz, qx, qy, qz, qw) -> [B,4,4] camera-to-world matrices."""
    t, q = vec[:, :3], vec[:, 3:] / vec[:, 3:].norm(dim=1, keepdim=True)
    x, y, z, w = q.unbind(1)
    Rm = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=1).view(-1, 3, 3)
    T = torch.eye(4, dtype=vec.dtype, device=vec.device).repeat(vec.shape[0], 1, 1)
    T[:, :3, :3] = Rm
    T[:, :3, 3] = t
    return T


def matrix_to_row(T):
    """[4,4] -> [tx, ty, tz, qx, qy, qz, qw]."""
    T = T.detach().cpu().double().numpy()
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()])


# --------------------------------------------------------------------------------------
# output files
# --------------------------------------------------------------------------------------
def save_trajectory(rows, which, path):
    with open(path, 'w') as f:
        f.write('# anchor_key tx ty tz qx qy qz qw  (camera-to-world, metres)\n')
        for r in rows:
            f.write(r['key'] + ' ' + ' '.join(f'{v:.9g}' for v in matrix_to_row(r[which])) + '\n')


def summarize(rows):
    """Mean/std of every metric (ours: successful pairs only; init: all pairs) + success rate."""
    out = {'n': len(rows)}
    ok = [r for r in rows if r['ours_ok']]
    out['ours_success_rate'] = 100.0 * len(ok) / max(len(rows), 1)
    for m in DEPTH_METRICS + POSE_METRICS:
        for method, subset in (('ours', ok), ('init', rows)):
            vals = np.array([r[f'{method}_{m}'] for r in subset], dtype=float)
            out[f'{method}_{m}'] = (vals.mean(), vals.std()) if len(vals) else (float('nan'), float('nan'))
    return out


def format_summary(s):
    lines = [f"# pairs: {s['n']} | ours_success_rate: {s['ours_success_rate']:.1f} %"]
    for method in METHODS:
        parts = [f"{method}_{m}: {s[f'{method}_{m}'][0]:.4f} ± {s[f'{method}_{m}'][1]:.4f}"
                 for m in DEPTH_METRICS + POSE_METRICS]
        lines.append('# ' + ' | '.join(parts))
    lines.append('# ours: successful pairs only; init: all pairs')
    return '\n'.join(lines) + '\n'


def save_tables(rows, out_dir):
    with open(os.path.join(out_dir, 'res.txt'), 'w') as f:
        f.write('# anchor_key ours_t_mm ours_rot_rad init_t_mm init_rot_rad ours_success\n')
        for r in rows:
            f.write(f"{r['key']} {r['ours_t_mm']:.4f} {r['ours_rot_rad']:.4f} "
                    f"{r['init_t_mm']:.4f} {r['init_rot_rad']:.4f} {int(r['ours_ok'])}\n")
        f.write(format_summary(summarize(rows)))
    with open(os.path.join(out_dir, 'dsim.txt'), 'w') as f:
        f.write('# anchor_key ours_ds ours_nc ours_si init_ds init_nc init_si\n')
        for r in rows:
            f.write(r['key'] + ' ' + ' '.join(f"{r[f'{m}_{d}']:.4f}" for m in METHODS for d in DEPTH_METRICS) + '\n')
        f.write(format_summary(summarize(rows)))
    save_trajectory(rows, 'T_pred', os.path.join(out_dir, 'poses_pred.txt'))
    save_trajectory(rows, 'T_init', os.path.join(out_dir, 'poses_init.txt'))
    save_trajectory(rows, 'T_gt', os.path.join(out_dir, 'poses_gt.txt'))


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_sequence(cfg, root, model):
    """Evaluate one sequence; returns {level: [row, ...]} and writes per-level tables."""
    device = cfg.device
    renderer = DepthRenderer(cfg, mesh_path=os.path.join(root, 'localMedium.ply')).to(device)
    sdf = MeshSDF(renderer.mesh)
    rs = transforms.Resize((cfg.model.input_dim.h, cfg.model.input_dim.w),
                           interpolation=transforms.InterpolationMode.BICUBIC)
    dispSim, SI = dispSimilarity(), ScaleInvariantDepthLoss()

    def depth_metrics(depth, gt_depth):
        return {'ds': dispSim(depth, gt_depth).item(),
                'nc': normal_correlation(depth, gt_depth).item(),
                'si': SI(depth, gt_depth).item()}

    results = {}
    for level in cfg.eval.levels:
        loader = DataLoader(PoseEvalDataset(cfg, root, level=level), batch_size=1, shuffle=False)
        rows = []
        for batch in tqdm(loader, desc=f'{os.path.basename(root)} level {level}'):
            rgb = batch['anchor_rgb'].to(device)
            T_gt = pose_vec_to_matrix(batch['anchor_pose'].to(device))
            T_init = pose_vec_to_matrix(batch['sample_pose'].to(device))

            gt_depth = rs(renderer(T_gt)[0])                              # target depth at the GT pose
            init_depth, sim_rgb = renderer(T_init, color_render=True)     # render at the INITIAL pose
            rel_pose_pred = model(rs(sim_rgb), rgb)                       # your model
            T_pred = T_init @ rel_pose_pred
            pred_depth = rs(renderer(T_pred)[0])

            row = {'key': batch['anchor_key'][0], 'T_gt': T_gt[0], 'T_init': T_init[0], 'T_pred': T_pred[0]}
            row['ours_ok'] = is_success(sdf, T_pred[0])
            for method, T, depth, ok in (('ours', T_pred, pred_depth, row['ours_ok']),
                                         ('init', T_init, rs(init_depth), True)):
                if ok:
                    row.update({f'{method}_{k}': v for k, v in depth_metrics(depth, gt_depth).items()})
                    row[f'{method}_t_mm'] = translation_error_mm(T_gt[0], T[0]).item()
                    row[f'{method}_rot_rad'] = rotation_error_rad(T_gt[0], T[0]).item()
                else:
                    row.update({f'{method}_{k}': FAILED for k in DEPTH_METRICS + POSE_METRICS})
            rows.append(row)

        out_dir = os.path.join(cfg.output, os.path.basename(os.path.normpath(root)), f'level_{level}')
        os.makedirs(out_dir, exist_ok=True)
        save_tables(rows, out_dir)
        results[level] = rows
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    model = build_model(cfg).to(cfg.device).eval()

    all_rows = {level: [] for level in cfg.eval.levels}
    for root in sequence_roots(cfg):
        for level, rows in evaluate_sequence(cfg, root, model).items():
            all_rows[level].extend(rows)

    os.makedirs(cfg.output, exist_ok=True)
    with open(os.path.join(cfg.output, 'summary.txt'), 'w') as f:
        for level, rows in all_rows.items():
            f.write(f'## level {level}\n' + format_summary(summarize(rows)))
        f.write('## all levels\n' + format_summary(summarize([r for rows in all_rows.values() for r in rows])))
    print(open(os.path.join(cfg.output, 'summary.txt')).read())


if __name__ == "__main__":
    main()
