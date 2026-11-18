"""Benchmark metrics: depth consistency (DS, NC, SI), pose error, and the success test.

Depth maps are ``[B,1,H,W]`` metric z-depth (metres). Poses are ``[4,4]`` or ``[B,4,4]``
camera-to-world matrices in metres.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#: Value written for a metric on a failed frame (camera centre outside the airway).
FAILED = -1.0

#: Intrinsics the published NC numbers were computed with. The original call
#: ``normals_from_depth(z, eps)`` passed ``eps`` positionally into ``fx``, so ``fx`` was 1e-6
#: and the other three kept these defaults. Kept as the default of :func:`normal_correlation`
#: so results stay comparable with the paper.
LEGACY_NC_INTRINSICS = dict(fx=1e-6, fy=110.3, cx=100.58, cy=100.69)

#: Geometrically correct intrinsics of the 200x200 renders (OpenCV pixel-centre convention).
#: The nominal values in configs/config.yaml (cx=100.58, cy=103.69) are not what the renderer
#: produces: its projection matrix mirrors cy and the rasterizer samples at pixel centres.
#: Rescale for other sizes with fx' = s*fx, cx' = (cx + 0.5)*s - 0.5.
RENDER_INTRINSICS_200 = dict(fx=110.21, fy=110.31, cx=100.08, cy=95.81)


def normals_from_depth(z, fx, fy, cx, cy, eps=1e-6):
    """
    z: [B,1,H,W] depth (meters). Intrinsics: fx, fy, cx, cy (floats).
    Returns:
      n : [B,3,H,W] unit normals
      Px: [B,3,H,W] 3D finite diff along x
      Py: [B,3,H,W] 3D finite diff along y
    """
    B, _, H, W = z.shape
    device = z.device
    u = torch.arange(W, device=device).view(1,1,1,W).expand(B,1,H,W)
    v = torch.arange(H, device=device).view(1,1,H,1).expand(B,1,H,W)

    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    P = torch.cat([X, Y, z], dim=1)  # [B,3,H,W]

    Px = F.pad(P[:,:,:,1:] - P[:,:,:,:-1], (0,1,0,0), mode='replicate')
    Py = F.pad(P[:,:,1:,:] - P[:,:,:-1,:], (0,0,0,1), mode='replicate')

    n = torch.cross(Px, Py, dim=1)
    n = n / (n.norm(dim=1, keepdim=True) + eps)
    return n, Px, Py

def normal_correlation(z1, z2, mask=None, use_edge_weight=True, eps=1e-6, return_deg=False,
                       intrinsics=LEGACY_NC_INTRINSICS):
    """
    Scale-invariant normal correlation between two depth maps.
    - Invariant to global multiplicative depth scale per map.
    - Optional symmetric edge-weighting using 3D gradient magnitudes.

    ``intrinsics`` defaults to :data:`LEGACY_NC_INTRINSICS` so the numbers match the paper.
    Pass intrinsics matching the depth-map resolution for the geometrically correct version.
    """
    n1, Px1, Py1 = normals_from_depth(z1, eps=eps, **intrinsics)
    n2, Px2, Py2 = normals_from_depth(z2, eps=eps, **intrinsics)

    cos = (n1 * n2).sum(1).clamp(-1, 1)  # [B,H,W]

    if use_edge_weight:
        w1 = Px1.norm(dim=1) + Py1.norm(dim=1)       # [B,H,W]
        w2 = Px2.norm(dim=1) + Py2.norm(dim=1)
        w  = 0.5 * (w1 + w2)
    else:
        w  = torch.ones_like(cos)

    if mask is not None:
        w = w * mask

    corr = (cos * w).sum(dim=(1,2)) / (w.sum(dim=(1,2)) + eps)  # [B]

    if return_deg:
        ang = torch.arccos(cos)  # [B,H,W], radians
        mean_ang = (ang * w).sum(dim=(1,2)) / (w.sum(dim=(1,2)) + eps)
        mean_ang_deg = mean_ang * (180.0 / math.pi)
        return corr, mean_ang_deg
    return corr

class dispSimilarity(nn.Module):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def madnorm(self, d):
        d_flat = d.flatten()
        t_d = torch.median(d_flat)
        s_d = torch.mean(torch.abs(d_flat - t_d))
        return (d_flat - t_d) / s_d

    def forward(self, render_depth, target_disp) -> Tensor:
        flat_depth = self.madnorm(render_depth)
        flat_disp = self.madnorm(target_disp)
        sim = self.cos(flat_depth, flat_disp)
        return (sim+1)/2

class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, mean: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.mean = mean

    def forward(self, rot_pred, rot_gt) -> Tensor:
        R_pred, R_gt = rot_pred, rot_gt
        m = torch.bmm(R_gt, R_pred.transpose(1,2)) #batch*3*3
        cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1)/2
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
        if self.mean:
            return torch.mean(theta)
        else:
            return theta

class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-invariant depth loss (Eigen et al. 2014).
    Works on log(depth).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, error_map=False, eps=1e-6):
        """
        pred:   [B,1,H,W] predicted depth
        target: [B,1,H,W] ground truth depth
        mask:   [B,1,H,W] optional validity mask (1=valid, 0=ignore)
        error_map: [B,1,H,W] optional error map
        """
        # clamp to avoid log(0)
        pred = pred.clamp(min=eps)
        target = target.clamp(min=eps)

        log_diff = torch.log(pred) - torch.log(target)   # [B,1,H,W]

        if mask is not None:
            log_diff = log_diff[mask > 0]

        n = log_diff.numel()
        if n == 0:
            return torch.tensor(0.0, device=pred.device)

        term1 = torch.sum(log_diff ** 2) / n
        term2 = (torch.sum(log_diff) ** 2) / (n ** 2)
        if not error_map:
            return term1 - term2
        else:
            mean_d  = torch.sum(log_diff)/ n
            mean_d_broadcast = mean_d.view(-1,1,1,1)          # [B,1,1,1]
            si_map = (log_diff - mean_d_broadcast)**2
            return term1 - term2, si_map


# --------------------------------------------------------------------------------------
# pose error and success
# --------------------------------------------------------------------------------------
def translation_error_mm(T_gt: Tensor, T_pred: Tensor) -> Tensor:
    """Distance between camera centres in millimetres, for [4,4] or [B,4,4] poses in metres."""
    return torch.linalg.norm(T_gt[..., :3, 3] - T_pred[..., :3, 3], dim=-1) * 1000.0


def rotation_error_rad(T_gt: Tensor, T_pred: Tensor) -> Tensor:
    """Geodesic rotation angle in radians, for [4,4] or [B,4,4] poses."""
    R_gt = T_gt[..., :3, :3].reshape(-1, 3, 3)
    R_pred = T_pred[..., :3, :3].reshape(-1, 3, 3)
    return GeodesicLoss(mean=False)(R_pred, R_gt).reshape(T_gt.shape[:-2])


def sdf_barrier(s: Tensor, tau: float = 1.0e-4, bad: float = 1.0e-4, gamma: float = 5000.0) -> Tensor:
    """The SDF loss used for the success test in the paper (s in metres, negative inside).

    Its value exceeds 1 once the camera centre is less than about 0.067 mm inside the wall,
    or outside the airway.
    """
    L_inside = F.softplus(gamma * (s + tau)) ** 2
    s_neg = -s.clamp_max(-1e-9)
    near = (s_neg <= bad).float()
    L_near = near * torch.clamp(-torch.log(s_neg / bad + 1e-12), min=0.0)
    L_out = F.relu(s) ** 2
    return L_inside + L_near + L_out


class MeshSDF:
    """Signed distance to the airway mesh (metres, negative inside the lumen).

    The sign comes from the barycentric-interpolated vertex normal at the closest surface
    point; mesh normals point out of the lumen. Needs ``rtree`` for trimesh's proximity query.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.vertex_normals = np.asarray(mesh.vertex_normals)

    def __call__(self, points) -> np.ndarray:
        import trimesh

        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        closest, dist, tri = trimesh.proximity.closest_point(self.mesh, p)
        bary = trimesh.triangles.points_to_barycentric(self.mesh.triangles[tri], closest)
        n = (bary[:, :, None] * self.vertex_normals[self.mesh.faces[tri]]).sum(1)
        sign = np.where(((p - closest) * n).sum(1) > 0, 1.0, -1.0)
        return sign * dist


def is_success(sdf: MeshSDF, T: Tensor, threshold: float = 1.0) -> bool:
    """A pose succeeds unless :func:`sdf_barrier` at its camera centre exceeds ``threshold``.

    The paper evaluated the SDF on a precomputed 160^3 grid (generation/mesh2sdf.py); this
    evaluates it on the mesh directly. On one airway the two agreed on 98.9 % of decisions
    (12k poses); the disagreements were poses within about 0.2 mm of the wall.
    """
    s = sdf(T[..., :3, 3].detach().cpu().numpy())
    return bool(sdf_barrier(torch.as_tensor(s)).max() <= threshold)
