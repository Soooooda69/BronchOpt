import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from torch import Tensor

def normals_from_depth(z, fx=110.2, fy=110.3, cx=100.58, cy=100.69, eps=1e-6):
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

def normal_correlation(z1, z2, mask=None, use_edge_weight=True, eps=1e-6, return_deg=False):
    """
    Scale-invariant normal correlation between two depth maps.
    - Invariant to global multiplicative depth scale per map.
    - Optional symmetric edge-weighting using 3D gradient magnitudes.
    """
    n1, Px1, Py1 = normals_from_depth(z1,  eps)
    n2, Px2, Py2 = normals_from_depth(z2, eps)

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
    def __init__(self, cfg) -> None:
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
        # predicted_disp = self.d2d(flat_depth.view(-1, 1)).squeeze()
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
        # theta = 2 * torch.asin(torch.clamp(torch.sqrt((1 - cos) / 2), 0, 1-self.eps))
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