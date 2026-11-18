import os, json, numpy as np, trimesh
import mesh_to_sdf as mts
import torch
import matplotlib.pyplot as plt
from generation.common import load_precomputed_sdf
import time
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
import os

def load_mesh_canonical(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, process=False)  # deterministic
    if isinstance(m, trimesh.Scene):
        # Dump returns geometries with transforms applied (world coords)
        m = trimesh.util.concatenate(tuple(m.dump().values()))
    assert isinstance(m, trimesh.Trimesh)
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    m.remove_duplicate_faces()
    m.remove_infinite_values()
    return m

def precompute_sdf_mesh_to_sdf(
    ply_path: str,
    out_dir: str,
    resolution: int = 160,           # samples along the normalized [-1,1] range
    pad: bool = True,                # add 1-voxel border on each side
    surface_point_method: str = "sample",   # "sample" avoids GL; "scan" needs OpenGL
    sign_method: str = "normal",     # "normal" for non-watertight; "depth" for watertight
    scan_count: int = 100,
    scan_resolution: int = 400,
    save_float16: bool = False,
):
    """
    Builds and saves a dense SDF from a .ply mesh using mesh-to-sdf.

    Writes:
      out_dir/sdf.npy   : (D,H,W) float grid (Z,Y,X), negative inside, in WORLD units
      out_dir/meta.json : origin (world xyz of voxel 0,0,0), voxel size, dims (D,H,W), center, scale

    Returns: (sdf_dhw: np.ndarray, meta: dict)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load mesh (no auto-processing that might change scale)
    mesh = load_mesh_canonical(ply_path)

    if not isinstance(mesh, trimesh.Trimesh):
        # If it's a Scene, merge into one mesh
        mesh = trimesh.util.concatenate(tuple(mesh.dump().values()))


    # 2) Center at the mesh centroid (consistent with the unit-cube normalization)
    mn, mx = mesh.bounds
    center = 0.5 * (mn + mx)     # <- AABB center (replace mesh.centroid)
    mesh_norm = mesh.copy()
    mesh_norm.apply_translation(-center)
    
    print("voxelizing...")
    start_time = time.time()
    # 3) Voxel SDF in normalized space (mesh_to_sdf maps to cube [-1,1] with uniform scale)
    vox = mts.mesh_to_voxels(
        mesh_norm,
        voxel_resolution=resolution,
        surface_point_method=surface_point_method,
        sign_method=sign_method,
        scan_count=scan_count,
        scan_resolution=scan_resolution,
        pad=pad,
        check_result=False,
        return_gradients=False,
    )
    # vox shape: (N,N,N) or (N+2,N+2,N+2) if pad=True
    print(f"Mesh to SDF took {time.time() - start_time:.2f} seconds")

    # 4) Convert normalized SDF -> WORLD using AABB max half-extent as the uniform scale
    half_extents = 0.5 * (mx - mn)            # (3,)
    scale = float(half_extents.max())         # uniform scale to fit [-1,1]
    sdf_xyz = vox.astype(np.float32) * scale  # distances back to world units

    # 5) World-space grid metadata (must use the same 'scale' as above)
    pad_vox = 1 if pad else 0
    N_eff = resolution                         # samples across [-1,1]
    norm_voxel = 2.0 / (N_eff - 1)            # normalized voxel step
    world_voxel = norm_voxel * scale          # world voxel size

    # index (0,0,0) in normalized coords then map to world about 'center'
    origin_norm = -1.0 - pad_vox * norm_voxel
    origin_world = center + origin_norm * scale  # same for x,y,z

    # 6) Reorder to (D,H,W)=(Z,Y,X) and save
    sdf_whd = sdf_xyz.astype(np.float16 if save_float16 else np.float32)
    W, H, D = sdf_whd.shape
    np.save(os.path.join(out_dir, "sdf.npy"), sdf_whd)

    meta = {
        "origin": [float(origin_world[0]), float(origin_world[1]), float(origin_world[2])],
        "voxel": float(world_voxel),
        "dims": [int(D), int(H), int(W)],  # (Z,Y,X)
        "center": center.tolist(),
        "centering": "aabb_center",
        "scale": float(scale),             # AABB max half-extent used for normalization
        "resolution": int(resolution),
        "pad": bool(pad),
        "surface_point_method": surface_point_method,
        "sign_method": sign_method,
        "centering": "mesh.centroid",
        "normalization": "unit_cube_max_half_extent",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return sdf_whd, meta

@torch.no_grad()
def _make_xy_mesh(H, W, device, dtype):
    """Cached HxW integer index mesh (y,x) on device."""
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return yy, xx

def precompute_normals_from_sdf(
    folder: str,
    out_path: str | None = None,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.float32,
    z_chunk: int = 8,              # process this many z-slices per autograd pass
    save_float16: bool = False,
):
    """
    Load ./sdf.npy + ./meta.json, compute world-space normals via autograd at voxel centers,
    and save normals as ./normals.npy with shape (3, D, H, W). Also updates meta.json.

    Notes:
      - Uses DifferentiableSDFGrid.forward (grid_sample) so the gradient is that of the
        trilinearly interpolated SDF field.
      - Processes in z-slice chunks to bound memory.
    """
    os.makedirs(folder, exist_ok=True)
    out_path = out_path or os.path.join(folder, "normals.npy")

    # 1) Load SDF grid and meta
    sdf, meta = load_precomputed_sdf(folder, device=device, dtype=dtype)
    D, H, W = int(meta["dims"][0]), int(meta["dims"][1]), int(meta["dims"][2])

    # 2) Prepare output tensor on device
    normals_dhw = torch.empty(3, D, H, W, dtype=torch.float32, device=device)

    # 3) Build (H,W) index mesh once
    yy, xx = _make_xy_mesh(H, W, device, dtype=dtype)

    # 4) Iterate over z in chunks
    for z0 in range(0, D, z_chunk):
        z1 = min(z0 + z_chunk, D)
        n_slices = z1 - z0

        # (n_slices, H, W) integer z indices
        zz = torch.arange(z0, z1, device=device, dtype=dtype)[:, None, None].expand(n_slices, H, W)

        # voxel indices -> world coords: origin + idx * voxel
        # shapes: (n_slices, H, W, 3)
        xw = sdf.origin[0] + (xx[None, ...] * sdf.voxel)
        yw = sdf.origin[1] + (yy[None, ...] * sdf.voxel)
        zw = sdf.origin[2] + (zz * sdf.voxel)
        pts_world = torch.stack([xw.expand_as(zw), yw.expand_as(zw), zw], dim=-1)

        # flatten to (N,3) and enable grad for autograd normals
        pts_flat = pts_world.reshape(-1, 3).detach().requires_grad_(True)

        # Forward SDF at these points (no no_grad here!)
        s = sdf(pts_flat)  # (N,)

        # âS/âp via autograd; create_graph=False (we're precomputing)
        g = torch.autograd.grad(s.sum(), pts_flat, create_graph=False, retain_graph=False)[0]  # (N,3)

        # normalize -> (n_slices, H, W, 3)
        g = g.reshape(n_slices, H, W, 3)
        g = g / (g.norm(dim=-1, keepdim=True).clamp_min(1e-12))

        # write into output (3, D, H, W)
        g = g.permute(3, 0, 1, 2).contiguous()  # (3, n_slices, H, W)
        normals_dhw[:, z0:z1] = g

        # (optional) free some transient memory
        del pts_world, pts_flat, s, g
        torch.cuda.empty_cache() if "cuda" in str(device) else None

    # 5) Save as numpy
    nrm_np = normals_dhw.detach().float().cpu().numpy()
    if save_float16:
        nrm_np = nrm_np.astype(np.float16)
    np.save(out_path, nrm_np)

    # 6) Update meta.json
    meta["has_normals"] = True
    meta["normals_path"] = os.path.relpath(out_path, folder)
    meta["normals_layout"] = "C,D,H,W (C=x,y,z in world axes)"
    meta["normals_source"] = "autograd_grid_sample_trilinear"
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_path

# @torch.no_grad()
def check_vertices_close_to_zero(mesh_path, sdf, sample=50000, batch=200000, device=None):
    """
    Prints |SDF| stats at mesh vertices (should be ~0 if the SDF grid aligns to the mesh).
    - sample: random subset size (None = all)
    - batch:  chunk size to avoid OOM
    """
    # Resolve device from sdf buffers if not provided
    if device is None:
        if hasattr(sdf, "grid"):
            device = sdf.grid.device
        else:
            try:
                device = next(sdf.buffers()).device
            except StopIteration:
                device = torch.device("cpu")

    mesh = load_mesh_canonical(mesh_path)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    if sample is not None and len(V) > sample:
        idx = np.random.choice(len(V), sample, replace=False)
        V = V[idx]

    # Query in chunks
    ds = []
    for i in range(0, len(V), batch):
        vt = torch.from_numpy(V[i:i+batch]).to(device)
        d = sdf(vt).abs().detach().cpu().numpy()
        ds.append(d)
    d = np.concatenate(ds, axis=0)

    # Tolerance reference: ~half a voxel is good
    voxel = getattr(getattr(sdf, "meta", None), "voxel", None)
    vox_info = f" (voxel={voxel:.6f})" if isinstance(voxel, (float, int)) else ""
    print(f"[SDF@verts]{vox_info}  mean={d.mean():.6f}  median={np.median(d):.6f}  "
          f"p95={np.percentile(d,95):.6f}  max={d.max():.6f}")
    return d

def sample_points_and_get_sdf(sdf, K, T_cw, num_points=400, H=200, W=200, znear=0, zfar=0.1, device=None):
    R = T_cw[:3, :3]                       # world <- cam
    t = T_cw[:3, 3]                        # cam origin in world
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # --- build per-pixel rays in camera frame ---
    v = torch.linspace(0, H - 1, int(num_points ** 0.5), device=device)  # evenly spaced row indices
    u = torch.linspace(0, W - 1, int(num_points ** 0.5), device=device)  # evenly spaced column indices
    V, U = torch.meshgrid(v, u, indexing="ij")                 # shapes (sqrt(H), sqrt(W))
    x = (U - cx) / fx
    y = (V - cy) / fy
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1).contiguous()
    # --- transform to world ---
    dirs_w = (R @ dirs_cam.view(-1, 3).T).T  # Transform to world and reshape to (N, 3)
    orig_w = t.expand(dirs_w.shape[0], 3)  # Expand origin to match (N, 3)
    # --- sample points along rays ---
    t_vals = torch.linspace(znear, zfar, steps=10, device=device)
    points = orig_w.unsqueeze(1) + dirs_w.unsqueeze(1) * t_vals.view(1, -1, 1)  # (N, num_samples, 3)
    # breakpoint()
    points = points.reshape(-1, 3)  # (N * num_samples, 3)
    # --- query SDF ---
    sdf_values = sdf(points)
    camera_sdf = sdf(t[None])
    ###########################
    # Check which rays intersect the surface (zero-crossing of SDF along samples)
    S = t_vals.numel()
    s = sdf_values.view(-1, S)
    if s.shape[-1] == 1:
        s = s.squeeze(-1)

    eps = 1e-3
    s_prev = s[:, :-1]
    s_next = s[:, 1:]
    zero_or_cross = (s_prev * s_next <= 0) | (s_prev.abs() < eps) | (s_next.abs() < eps)
    ray_hit_mask = zero_or_cross.any(dim=1)

    # Rays that start outside (positive SDF) and then hit the surface are not allowed
    starts_outside = s[:, 0] > 0
    invalid_outside_hits = starts_outside & ray_hit_mask
    if invalid_outside_hits.any():
        print(f"Disallowed rays (outside->surface) removed: {invalid_outside_hits.sum().item()}")

    # Enforce: remove those rays from the hit set
    ray_hit_mask = ray_hit_mask & (~invalid_outside_hits)

    hit_rate = ray_hit_mask.float().mean().item()
    miss_mask = ~ray_hit_mask
    miss_rate = miss_mask.float().mean().item()

    print(f"Ray surface hit rate (allowed): {hit_rate*100:.1f}% ({ray_hit_mask.sum().item()}/{ray_hit_mask.numel()})")

    if miss_rate > 0:
        print(f"Background rays (no allowed hit): {miss_mask.sum().item()} ({miss_rate*100:.1f}%).")
    else:
        print("No background rays among allowed rays.")

    lsdf = 0.1*sdf_loss(camera_sdf)
    print(f"Camera SDF loss: {lsdf.item():.6f} camera sdf: {camera_sdf.item()*1000:.2f} mm")
    return points, sdf_values, camera_sdf


def plot_sdf_loss(curve_range=(-5e-3, 1e-3), n=800,
                    device="cpu", show=True, save_path=None):
    xs = torch.linspace(curve_range[0], curve_range[1], n, device=device)
    xs.requires_grad_(False)

    l_sdf = 0.1*sdf_loss(xs)
    plt.figure(figsize=(8, 6))
    plt.plot(xs.cpu().numpy(), l_sdf.cpu().numpy(), label="SDF Loss")
    plt.title("SDF Loss Curve")
    plt.xlabel("SDF Value (m)")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(cfg: DictConfig):
    data_root = cfg.data_root
    mesh_path = os.path.join(cfg.data_root, "localMedium.ply")
    out_dir = os.path.join(cfg.data_root, "sdf")
    centerline_file = os.path.join(cfg.data_root, "centerlines.txt")
    
    # COMPUTE SDF FROM MESH
    sdf, meta = precompute_sdf_mesh_to_sdf(
    mesh_path, f"{data_root}/sdf",
    resolution=160, pad=True, sign_method="normal"
    )

    sdf, meta = load_precomputed_sdf(f"{data_root}/sdf", device=cfg.device)
    
    # COMPUTE NORMALS FROM SDF
    precompute_normals_from_sdf(f"{data_root}/sdf")

if __name__ == "__main__":
    main()
    # data_root = 'bronch_real_004'
    # mesh_path = f'./processed/{data_root}/localMedium.ply'
    # centerline_file = f"./processed/{data_root}/centerlines.txt"

    # # sdf, meta = precompute_sdf_mesh_to_sdf(
    # # mesh_path, f"./processed/{data_root}/sdf",
    # # resolution=160, pad=True, sign_method="normal"
    # # )
    # # plot_sdf_loss(save_path="sdf_loss_curve.png")

    # sdf, meta = load_precomputed_sdf(f"./processed/{data_root}/sdf", device="cuda:0")
    # # breakpoint()
    # precompute_normals_from_sdf(f"./processed/{data_root}/sdf")

    # intrinsics = np.array([[110.21, 0.0, 100.58],
    #                        [0.0, 110.31, 103.69],
    #                        [0.0, 0.0, 1.0]], dtype=np.float32)
    # poses_list, vec_poses_list, _, id_list = pose_process(centerline_file)
    # centerline_poses = torch.tensor(np.array([pose for pose in poses_list], dtype=np.float32), device="cuda:0")

    # K = torch.tensor(intrinsics, device=centerline_poses.device)
    
    # from hydra import compose, initialize
    # from omegaconf import OmegaConf
    # from utils.losses import SDFLoss
    # with initialize(version_base=None, config_path="../configs", job_name="test_data"):
    #     cfg = compose(config_name="demo")
    # l_sdf = SDFLoss(cfg)
    # losses = l_sdf(centerline_poses)

    # for T_cw in centerline_poses:
    #     # T_cw = pose_preturb(T_cw, t_noise=0.006, rot_noise=20).to(T_cw.device)
    #     start = time.time()
    #     points, sdf_values, camera_sdf = sample_points_and_get_sdf(
    #         sdf, K, T_cw, num_points=400, H=200, W=200, device=T_cw.device
    #     )
    #     print(f"Sampled {len(points)} points in {time.time() - start} seconds")
    #     depth_pred, mask, normal = render_sdf_depth_normals(
    #     sdf, K, T_cw, H=200, W=200,
    #     near=0.0, far=0.1, max_steps=128, eps=1e-4
    #     )