"""Helpers shared by the data-generation scripts.

Collected from the internal BronchOpt codebase at the state used to generate the released
benchmark (pose maths, the pose distance used for sampling and difficulty levels, the SDF grid
and its loss, and triangle covisibility). Numerics are unchanged.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from metrics import GeodesicLoss


# --------------------------------------------------------------------------------------
# poses: [tx, ty, tz, qx, qy, qz, qw] <-> 4x4 camera-to-world
# --------------------------------------------------------------------------------------
def create_extrinsics_matrix(x, y, z, q_x, q_y, q_z, q_w):
    rotation = R.from_quat([q_x, q_y, q_z, q_w])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation.as_matrix()
    extrinsics[:3, 3] = [x, y, z]
    return extrinsics


def create_vectors_from_matrix(matrix):
    """4x4 -> [x, y, z, qx, qy, qz, qw] (float32)."""
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([translation, rotation], dtype=np.float32)


# --------------------------------------------------------------------------------------
# pose distance (sampling spread and the `pose_dis` / `level` fields)
# --------------------------------------------------------------------------------------
class PoseLoss(nn.Module):
    """Heuristic pose difficulty score between [B,4,4] poses.

    Angle between the two world-frame translation vectors + rotation geodesic angle (rad)
    + 100 * | |t_pred| - |t_gt| | (metres). Depends on the world origin; used only to spread
    samples and to bin pairs into difficulty levels.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.geodesic_loss = GeodesicLoss(mean=False)

    def forward(self, pred, gt) -> Tensor:
        t_pred = pred[:, :3, 3]
        t_gt = gt[:, :3, 3]
        eps = 1e-8
        t_pred_dir = F.normalize(t_pred + eps, dim=1)
        t_gt_dir = F.normalize(t_gt + eps, dim=1)
        cos_sim = F.cosine_similarity(t_pred_dir, t_gt_dir, dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        t_dir_ls = torch.arccos(cos_sim)
        t_mag_ls = F.l1_loss(t_pred.norm(dim=1), t_gt.norm(dim=1), reduction='none')
        rot_ls = self.geodesic_loss(pred[:, :3, :3], gt[:, :3, :3])
        return t_dir_ls + rot_ls + 100 * t_mag_ls


# --------------------------------------------------------------------------------------
# SDF grid (generation/mesh2sdf.py writes <data_root>/sdf/{sdf.npy, meta.json})
# --------------------------------------------------------------------------------------
class DifferentiableSDFGrid(nn.Module):
    def __init__(self, grid_whd: torch.Tensor, origin_xyz, voxel_size: float,
                 normals_dhw: torch.Tensor | None = None):
        super().__init__()
        assert grid_whd.ndim == 3, "SDF grid must be (W,H,D)"
        W, H, D = grid_whd.shape
        with torch.no_grad():
            self.register_buffer("grid", grid_whd.permute(2, 1, 0)[None, None].contiguous())
            self.register_buffer("origin", torch.tensor(origin_xyz, dtype=torch.float32))
        self.voxel = float(voxel_size)
        self.dims = (int(D), int(H), int(W))
        if normals_dhw is not None:
            assert normals_dhw.ndim == 4 and normals_dhw.shape[0] == 3, "normals must be (3,D,H,W)"
            assert tuple(normals_dhw.shape[1:]) == self.dims, "normals DHW must match SDF DHW"
            with torch.no_grad():
                self.register_buffer("nrm", normals_dhw[None].contiguous())
        else:
            self.nrm = None

    def _world_to_gridnorm(self, points_world: torch.Tensor):
        """World coords -> grid-normalized coords in [-1,1]^3 (align_corners=True)."""
        D, H, W = self.dims
        g = (points_world - self.origin.to(points_world.device)) / self.voxel
        x, y, z = g[:, 0], g[:, 1], g[:, 2]
        return torch.stack([2.0 * x / (W - 1) - 1.0, 2.0 * y / (H - 1) - 1.0, 2.0 * z / (D - 1) - 1.0], dim=-1)

    def forward(self, points_world: torch.Tensor, clamp_outside: bool = True) -> torch.Tensor:
        """SDF at world points, (N,) in world units (negative inside)."""
        grid = self._world_to_gridnorm(points_world).view(1, -1, 1, 1, 3)
        sdf_in = self.grid if self.grid.device == grid.device else self.grid.to(grid.device)
        return F.grid_sample(sdf_in.detach().clone(), grid, mode="bilinear",
                             padding_mode="border" if clamp_outside else "zeros",
                             align_corners=True).view(-1)


def _ensure_whd_from_meta_shape(sdf_np: np.ndarray, meta: dict) -> np.ndarray:
    Dm, Hm, Wm = int(meta["dims"][0]), int(meta["dims"][1]), int(meta["dims"][2])
    shp = tuple(sdf_np.shape)
    if shp == (Wm, Hm, Dm):
        return sdf_np
    if shp == (Dm, Hm, Wm):
        return np.transpose(sdf_np, (2, 1, 0))
    raise ValueError(f"sdf.npy shape {shp} does not match meta dims {(Dm, Hm, Wm)} in either WHD or DHW order.")


def load_precomputed_sdf(folder: str, device="cuda:0", dtype: torch.dtype = torch.float32):
    """Load <folder>/sdf.npy (+ normals.npy if present) and meta.json -> (DifferentiableSDFGrid, meta)."""
    sdf_path = os.path.join(folder, "sdf.npy")
    meta_path = os.path.join(folder, "meta.json")
    if not (os.path.exists(sdf_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Expected {sdf_path} and {meta_path}; run generation/mesh2sdf.py first")
    with open(meta_path) as f:
        meta = json.load(f)
    sdf_t = torch.from_numpy(_ensure_whd_from_meta_shape(np.load(sdf_path), meta)).to(dtype)

    normals_t = None
    normals_path = os.path.join(folder, "normals.npy")
    if os.path.exists(normals_path):
        nrm_np = np.load(normals_path)
        Dm, Hm, Wm = int(meta["dims"][0]), int(meta["dims"][1]), int(meta["dims"][2])
        if nrm_np.shape != (3, Dm, Hm, Wm):
            raise ValueError(f"normals.npy shape {nrm_np.shape} does not match (3,D,H,W)=({3, Dm, Hm, Wm})")
        normals_t = torch.from_numpy(nrm_np.astype(np.float32)).to(device)

    sdf = DifferentiableSDFGrid(sdf_t.to(device), meta["origin"], meta["voxel"], normals_t).to(device)
    return sdf, meta


class SDFLoss(nn.Module):
    """Barrier on the SDF at the camera centre (see metrics.sdf_barrier); > 1 = outside or
    within ~0.067 mm of the wall. Reads the grid from <cfg.data_root>/sdf."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.sdf, self.meta = load_precomputed_sdf(os.path.join(cfg.data_root, "sdf"), device=None)

    def sdf_loss(self, sdf_vals, tau=1.0e-4, bad=1.0e-4, gamma=5000.0, w_inside=1.0, w_outside=1.0):
        z = gamma * (sdf_vals + tau)
        L_inside = w_inside * F.softplus(z) ** 2
        s_neg = -sdf_vals.clamp_max(-1e-9)
        near = (s_neg <= bad).float()
        L_near = near * torch.clamp(-torch.log(s_neg / bad + 1e-12), min=0.0)
        L_out = w_outside * F.relu(sdf_vals) ** 2
        L = L_inside + L_near + L_out
        if L.dim() > 1:
            L = L.mean(dim=-1)
        return L

    def forward(self, poses, mean=True):
        t = poses[:, :3, 3]
        cam_sdfs = self.sdf.to(t.device)(t)
        with torch.amp.autocast('cuda', enabled=bool(self.cfg.model.mixed_precision)):
            loss = self.sdf_loss(cam_sdfs)
        return loss.mean() if mean else loss


# --------------------------------------------------------------------------------------
# covisibility (the `s_faces` field)
# --------------------------------------------------------------------------------------
def covis_from_tri_ids(tri_a, tri_b, background_val: int = -1):
    """s_faces = |F_A ∩ F_B| / min(|F_A|, |F_B|) over the sets of visible triangle ids, per batch."""
    def _prep(t):
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim == 4:
            t = t.squeeze(1)
        assert t.ndim == 3, f"Expected [B,H,W], got {t.shape}"
        return t

    A, B = _prep(tri_a).contiguous(), _prep(tri_b).contiguous()
    assert A.shape == B.shape, "tri_a and tri_b must have the same shape"
    out = torch.zeros(A.shape[0], device=A.device, dtype=torch.float32)
    for b in range(A.shape[0]):
        ids_a = A[b].view(-1)
        ids_b = B[b].view(-1)
        ids_a = ids_a[ids_a != background_val]
        ids_b = ids_b[ids_b != background_val]
        if ids_a.numel() == 0 or ids_b.numel() == 0:
            continue
        max_id = torch.maximum(ids_a.max(), ids_b.max()).item()
        pa = torch.bincount(ids_a, minlength=max_id + 1) > 0
        pb = torch.bincount(ids_b, minlength=max_id + 1) > 0
        denom = torch.minimum(pa.sum(), pb.sum()).clamp_min(1)
        out[b] = (pa & pb).sum().float() / denom.float()
    return out
