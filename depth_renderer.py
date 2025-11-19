import numpy as np
import torch
from torch import nn
import os
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
import trimesh
from torch.nn import functional as F
from third_party.lietorch import SE3

class DepthRenderer(nn.Module):
    def __init__(self, cfg, mesh_path=None):
        super().__init__()
        self.znear = cfg.cam.znear
        self.zfar = cfg.cam.zfar
        self.resolution = [cfg.cam.W, cfg.cam.H]
        self.glctx = dr.RasterizeCudaContext() # Initialize later based on device
        self.intrinsics = cfg.cam.intrinsics
        self.cfg = cfg  # Store the configuration for later use
        if cfg.data_root == 'all':
            # Load all meshes in the directory and store them in a dict
            ids = [f.split('_')[-1] for f in os.listdir('./processed') if f.startswith('bronch_real_')]
            data_roots = [f'./processed/bronch_real_{i}' for i in ids]
            mesh_dir = [os.path.join(data_root, 'localMedium.ply') for data_root in data_roots]
            self.meshes_dict = {}
            # store meshes properties
            self.normals_dict = {}
            self.vertices_dict = {}
            self._pos_dict = {}
            self._pos_idx_dict = {}
            for mesh_id, mesh_path in zip(ids, mesh_dir):
                mesh = trimesh.load(mesh_path, process=True)
                self.meshes_dict[mesh_id] = mesh
                self.normals_dict[mesh_id] = -mesh.vertex_normals
                self.vertices_dict[mesh_id] = mesh.vertices
                self._pos_dict[mesh_id] = torch.tensor(mesh.vertices, dtype=torch.float32)
                self._pos_idx_dict[mesh_id] = torch.tensor(np.array(mesh.faces), dtype=torch.int32)
        else:
            if mesh_path is not None:
                self.setup(mesh_path)
            else:
                self.setup(os.path.join(cfg.data_root, 'localMedium.ply'))
        
        self.camera_setup()

    def setup_mesh(self, mesh):
        self.mesh = mesh
        self.normals = self.mesh.vertex_normals
        self.normals = -self.normals  # Multiply by -1 to flip direction
        self.vertices = self.mesh.vertices
        faces = np.array(self.mesh.faces)
        self._pos = torch.tensor(self.vertices, dtype=torch.float32)
        self._pos_idx = torch.tensor(faces, dtype=torch.int32)
        self._col_idx = self._pos_idx
        print("Mesh has %d triangles and %d vertices." % (self._pos_idx.shape[0], self._pos.shape[0]))
        
    def camera_setup(self):
        glproj = glprojection(K=self.intrinsics, w=self.resolution[0], h=self.resolution[1], n=self.znear, f=self.zfar+0.1)
        self.glproj = torch.tensor(glproj, dtype=torch.float32)
        self.convert_rotation = torch.eye(4)
        rot_mat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
        self.convert_rotation[:3, :3] = torch.tensor(rot_mat, dtype=torch.float32)
        
    def setup(self, geometry_path):
        mesh = trimesh.load(geometry_path, process=True)
        self.setup_mesh(mesh)

    def select_mesh(self, mesh_id):
        if mesh_id in self.meshes_dict:
            self.mesh = self.meshes_dict[mesh_id]
            self.normals = self.normals_dict[mesh_id]
            self.vertices = self.vertices_dict[mesh_id]
            self._pos = self._pos_dict[mesh_id]
            self._pos_idx = self._pos_idx_dict[mesh_id]
        else:
            raise ValueError(f"Mesh id {mesh_id} not found in meshes_dict.")
            
    def light_setup(self, camera_center):
        vertex_normals = torch.tensor(self.normals, dtype=torch.float32).unsqueeze(0).to(camera_center.device)
        vertex_normals = F.normalize(vertex_normals, dim=-1)
        vertices = torch.tensor(self.vertices, dtype=torch.float32).unsqueeze(0).to(camera_center.device)
        light_dirs = camera_center.unsqueeze(1) - vertices  # Shape: [1, N, 3]
        light_dirs = F.normalize(light_dirs, dim=-1)
        diffuse = torch.clamp(torch.sum(vertex_normals * light_dirs, dim=-1), 0.0, 1.0)
        ambient = 0.1
        diffuse = ambient + (1.0 - ambient) * diffuse
        diffuse = diffuse ** (1/2.2)
        base_color = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32).to(camera_center.device)
        vertex_colors = diffuse.unsqueeze(-1) * base_color.unsqueeze(0)  # [1, N, 3]
        self._col = vertex_colors
        
    def transform_pos(self, mtx, pos):
        t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
        if t_mtx.ndim == 2:
            t_mtx = t_mtx.unsqueeze(0)
        ones = torch.ones((pos.shape[0], 1), device=pos.device)
        posw = torch.cat([pos, ones], dim=-1)
        transformed = torch.matmul(posw.unsqueeze(0), t_mtx.transpose(1, 2))
        return transformed
    
    def render(self, mtx, color_render=False, return_tri_ids=False):
        pos_clip = self.transform_pos(mtx, self._pos)
        depth_ = pos_clip[..., 2:3].contiguous()
        rast_out, _ = dr.rasterize(self.glctx, pos_clip.contiguous(), self._pos_idx, resolution=self.resolution)
        depth, _ = dr.interpolate(depth_, rast_out, self._pos_idx)
        outputs = [depth, None, None] # depth, color, triangle_ids
        if color_render:
            color, _ = dr.interpolate(self._col, rast_out, self._pos_idx)
            outputs[1] = color

        if return_tri_ids:
            # rast_out[..., 3] contains triangle indices, -1 means background
            outputs[2] = rast_out[..., 3][..., None].long()
        return outputs

    def forward(self, extrinsics, opt_rel=None, color_render=False, return_tri_ids=False, mesh_id=None):
        if isinstance(extrinsics, SE3):
            extrinsics = extrinsics.matrix()
        device = extrinsics.device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.glproj = self.glproj.to(device)
        self.convert_rotation = self.convert_rotation.to(device)
        if mesh_id is not None and self.cfg.data_root == 'all':
            depths = [None] * len(mesh_id)
            colors = [None] * len(mesh_id)
            tri_ids = [None] * len(mesh_id)
            all_mesh_ids = list(dict.fromkeys(mesh_id))
            for i, mid in enumerate(all_mesh_ids):
                self.select_mesh(mid)
                self._pos = self._pos.to(device)
                self._pos_idx = self._pos_idx.to(device)
                if color_render:
                    ids = [i for i, x in enumerate(mesh_id) if x == mid]
                    extrinsics_subset = extrinsics[ids]
                    self.light_setup(extrinsics_subset[:, :3, 3])
                    self._col = self._col.to(device)
                # breakpoint()
                if opt_rel is None:
                    opt_rel = torch.eye(4, device=device)
                w2cs = torch.linalg.inv(extrinsics_subset @ opt_rel)
                gl_w2cs = torch.matmul(self.convert_rotation, w2cs)
                mvp = torch.matmul(self.glproj, gl_w2cs)
                indices = [i for i, x in enumerate(mesh_id) if x == mid]
                outputs = self.render(mvp, color_render=color_render, return_tri_ids=return_tri_ids)
                for output, ls in zip(outputs, [depths, colors, tri_ids]):
                    if output is not None:
                        for j, idx in enumerate(indices):
                            ls[idx] = output[j].unsqueeze(0).permute(0, 3, 1, 2)

            # Concatenate results after processing all mesh groups
            depths_cat = torch.cat([t for t in depths if t is not None], dim=0) if any(d is not None for d in depths) else None
            colors_cat = torch.cat([t for t in colors if t is not None], dim=0) if any(c is not None for c in colors) else None
            tri_ids_cat = torch.cat([t for t in tri_ids if t is not None], dim=0) if any(ti is not None for ti in tri_ids) else None

            return [o for o in [depths_cat, colors_cat, tri_ids_cat] if o is not None]
            
        else:
            device = extrinsics.device
            self._pos = self._pos.to(device)
            self._pos_idx = self._pos_idx.to(device)
            if color_render:
                self.light_setup(extrinsics[:,:3,3])
                self._col = self._col.to(device)
            self.glproj = self.glproj.to(device)
            self.convert_rotation = self.convert_rotation.to(device)
            self.glctx = dr.RasterizeCudaContext(device=device)
            if opt_rel is None:
                opt_rel = torch.eye(4, device=device)
            w2cs = torch.linalg.inv(extrinsics @ opt_rel)
            gl_w2cs = torch.matmul(self.convert_rotation, w2cs)
            mvp = torch.matmul(self.glproj, gl_w2cs)
            outputs = self.render(mvp, color_render=color_render, return_tri_ids=return_tri_ids)
            return [o.permute(0, 3, 1, 2) for o in outputs if o is not None]            # if color_render:
     

def glprojection(K, w, h, n=0.0001, f=0.1):
    """
    get projection matrix from camera K
    refer to https://blog.csdn.net/hjwang1/article/details/94781272 for more information

    Args:
        camera_K (_type_): _description_
        width (_type_): _description_
        height (_type_): _description_
        near (int, optional): _description_. Defaults to 1.
        far (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    fx = K['fx']
    fy = K['fy']
    cx, cy = K['cx'], K['cy']

    A = (2 * fx) / w
    B = (2 * fy) / h
    C = (w - 2 * cx) / w
    D = -(h - 2 * cy) / h
    E = (n + f) / (n - f)
    F = (2 * n * f) / (n - f)

    projectionMatrix = np.array([
        [A, 0, C, 0],
        [0, B, D, 0],
        [0, 0, E, F],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    
    flip_y = np.array([
    [1,  0, 0, 0],
    [0, -1, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1]
    ], dtype=np.float32)
    projectionMatrix = projectionMatrix @ flip_y
    return projectionMatrix