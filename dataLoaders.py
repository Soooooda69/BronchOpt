"""Loading of benchmark sequences (``bronch_real_aero*``).

A sequence folder holds ``eval_src.json`` (pairs), ``pose_dict.json`` (poses),
``fake_images/`` (CycleGAN images, 256x256) and ``localMedium.ply`` (the airway mesh).

Poses are 7-vectors ``[tx, ty, tz, qx, qy, qz, qw]``: camera-to-world, metres, Hamilton
quaternion with the scalar last (the LieTorch ``SE3`` layout and SciPy ``as_quat()`` order).
"""
import json
import os
import re

import cv2
import torch
from torch.utils.data import Dataset


def sequence_roots(cfg):
    """Sequence folders to evaluate: ``cfg.data_root``, or every ``bronch_real_aero*``
    folder under ``cfg.dataset_root`` when ``data_root == 'all'``."""
    if cfg.data_root != 'all':
        return [cfg.data_root]
    names = [d for d in os.listdir(cfg.dataset_root)
             if d.startswith('bronch_real_aero') and os.path.isdir(os.path.join(cfg.dataset_root, d))]
    names.sort(key=lambda d: int(re.sub(r'\D', '', d) or 0))
    return [os.path.join(cfg.dataset_root, d) for d in names]


def key_to_relpath(key):
    """Image path of a pose key without extension: ``0_00012`` -> ``12`` (centerline node),
    ``1_00741_00008`` -> ``741/8`` (sample 8 around node 741)."""
    return '/'.join(str(int(k)) for k in key.split('_')[1:])


class PoseEvalDataset(Dataset):
    """Pairs of one sequence, optionally restricted to one difficulty level (1, 2 or 3).

    Each item holds the anchor (target) view's CycleGAN image, the anchor pose (ground
    truth) and the sample pose (initial guess). ``cfg.eval.start_idx / end_idx / subsample``
    select a slice of the pairs (``end_idx = -1`` means to the end).
    """

    def __init__(self, cfg, root, level=None):
        self.root = root
        self.size = (int(cfg.model.input_dim.w), int(cfg.model.input_dim.h))
        self.fake_dir = os.path.join(root, 'fake_images')

        with open(os.path.join(root, 'eval_src.json')) as f:
            pairs = json.load(f)['rel']
        if level is not None:
            pairs = [p for p in pairs if p.get('level', 0) == level]
        end = None if cfg.eval.end_idx in (-1, None) else int(cfg.eval.end_idx)
        self.pairs = pairs[int(cfg.eval.start_idx):end:int(cfg.eval.subsample)]

        with open(os.path.join(root, 'pose_dict.json')) as f:
            self.poses = {k: torch.tensor(v['pose'], dtype=torch.float32) for k, v in json.load(f).items()}

    def load_fake_rgb(self, key):
        """CycleGAN image of ``key`` as a [3,H,W] float tensor in [0,1], resized to the model input."""
        path = os.path.join(self.fake_dir, key_to_relpath(key) + '.png')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size)
        return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, sample = self.pairs[idx]['anchor'], self.pairs[idx]['sample']
        return {
            'anchor_key': anchor,
            'anchor_rgb': self.load_fake_rgb(anchor),   # query image
            'anchor_pose': self.poses[anchor],          # ground truth
            'sample_pose': self.poses[sample],          # initial pose
        }
