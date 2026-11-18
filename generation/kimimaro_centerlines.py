#!/usr/bin/env python3
"""
Generate 'segment.csv' and 'segment_cp.csv' from an airway mask using Kimimaro.
Outputs are ordered in **DFS** (depth-first) link order unless --order original is set.
"""

import argparse, csv, os
from collections import defaultdict, deque
import numpy as np
import nibabel as nib
import kimimaro
from scipy.ndimage import distance_transform_edt, zoom, map_coordinates
import pickle
import time

def skeletonize_with_kimimaro(mask_zyx, spacing_xyz):
    sx, sy, sz = spacing_xyz
    skels = kimimaro.skeletonize(
        (mask_zyx > 0).astype(np.uint32),
        teasar_params={'scale':2, 'const':5, 'pdrf_exponent':4, 'pdrf_scale':100000},
        object_ids=[1],
        anisotropy=(sz, sy, sx),
        dust_threshold=2000,
        fix_branching=True,
        fix_borders=True,
        progress=False
    )
    sk = skels[1]
    return sk.vertices, sk.edges

def build_adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    return adj

def split_into_links(vertices_zyx, edges):
    edges = np.asarray(edges, dtype=int)
    adj = build_adjacency(edges)
    deg = {i: len(adj[i]) for i in adj}
    def ekey(u,v): return (u, v) if u < v else (v, u)
    visited = set()
    links = []
    for src, d in deg.items():
        if d == 0: continue
        if d != 2:
            for nb in adj[src]:
                if ekey(src, nb) in visited: continue
                path = [src]
                prev, cur = src, nb
                visited.add(ekey(prev, cur))
                while deg.get(cur,0) == 2:
                    a,b = adj[cur]
                    nxt = a if b == prev else b
                    path.append(cur)
                    if ekey(cur, nxt) in visited: break
                    visited.add(ekey(cur, nxt))
                    prev, cur = cur, nxt
                path.append(cur)
                if len(path) >= 2 and path[0] != path[1]:
                    links.append(path)
    if not links and edges.size > 0:
        u, v = edges[0]
        path = [u, v]
        prev, cur = u, v
        seen = {(u,v)}
        while True:
            nxts = [w for w in adj[cur] if w != prev]
            if not nxts: break
            nxt = nxts[0]
            if (cur, nxt) in seen or (nxt, cur) in seen: break
            seen.add((cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        links.append(path)
    return links

def compute_edt_mm(mask_zyx, spacing_xyz, target_mm=1.0):
    sx, sy, sz = spacing_xyz
    scale_z = sz / target_mm
    scale_y = sy / target_mm
    scale_x = sx / target_mm
    # nearest-neighbor downsample of binary mask
    mask_coarse = zoom(mask_zyx, (scale_z, scale_y, scale_x), order=0) > 0
    return distance_transform_edt(mask_coarse, sampling=(sz, sy, sx))

def sample_radius_mm(dm_mm_zyx, v_zyx, spacing_xyz):
    sz, sy, sx = spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]
    iz = int(np.clip(np.rint(v_zyx[0] / sz), 0, dm_mm_zyx.shape[0]-1))
    iy = int(np.clip(np.rint(v_zyx[1] / sy), 0, dm_mm_zyx.shape[1]-1))
    ix = int(np.clip(np.rint(v_zyx[2] / sx), 0, dm_mm_zyx.shape[2]-1))
    return float(dm_mm_zyx[iz, iy, ix])

def _ortho_basis_from_tangent(d_xyz):
    d = d_xyz / (np.linalg.norm(d_xyz) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, ref);  u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(d, u);    v /= (np.linalg.norm(v) + 1e-9)
    return d, u, v

def _gather_cross_section_points(mask_zyx, spacing_xyz, center_xyz, d_xyz,
                                 slab_half_mm=1.0, roi_radius_mm=8.0):
    sx, sy, sz = spacing_xyz
    cz = int(np.clip(np.rint(center_xyz[2] / sz), 0, mask_zyx.shape[0]-1))
    cy = int(np.clip(np.rint(center_xyz[1] / sy), 0, mask_zyx.shape[1]-1))
    cx = int(np.clip(np.rint(center_xyz[0] / sx), 0, mask_zyx.shape[2]-1))

    rx = int(np.ceil(roi_radius_mm / sx))
    ry = int(np.ceil(roi_radius_mm / sy))
    rz = int(np.ceil(roi_radius_mm / sz))

    zlo, zhi = max(0, cz - rz), min(mask_zyx.shape[0]-1, cz + rz)
    ylo, yhi = max(0, cy - ry), min(mask_zyx.shape[1]-1, cy + ry)
    xlo, xhi = max(0, cx - rx), min(mask_zyx.shape[2]-1, cx + rx)

    d, u, v = _ortho_basis_from_tangent(d_xyz)
    pts2d = []

    for iz in range(zlo, zhi+1):
        z_mm = iz * sz
        for iy in range(ylo, yhi+1):
            y_mm = iy * sy
            for ix in range(xlo, xhi+1):
                if not mask_zyx[iz, iy, ix]:
                    continue
                x_mm = ix * sx
                p = np.array([x_mm, y_mm, z_mm])
                sd = float(np.dot(p - center_xyz, d))
                if abs(sd) > slab_half_mm:
                    continue
                rel = p - center_xyz
                uu, vv = float(np.dot(rel, u)), float(np.dot(rel, v))
                if uu*uu + vv*vv <= roi_radius_mm * roi_radius_mm:
                    pts2d.append([uu, vv])

    if not pts2d:
        return np.zeros((0,2), dtype=np.float32)
    return np.asarray(pts2d, dtype=np.float32)

def _ellipse_axes_from_points2d(pts2d):
    if pts2d.shape[0] < 10:
        return np.nan, np.nan
    X = pts2d - pts2d.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(1.0, (X.shape[0] - 1.0))
    evals = np.linalg.eigvalsh(C)
    evals = np.maximum(evals, 1e-9)
    a = 2.0 * np.sqrt(evals[-1])
    b = 2.0 * np.sqrt(evals[0])
    return (float(a), float(b)) if a >= b else (float(b), float(a))

def polyline_length(P):
    if len(P) < 2: return 0.0
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    return float(d.sum())

def resample_polyline(P, step_mm=0.6):
    if len(P) <= 2: return P.copy()
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0: return P.copy()
    n_new = max(2, int(np.round(s[-1] / step_mm)) + 1)
    s_new = np.linspace(0.0, s[-1], n_new)
    P_new = np.stack([np.interp(s_new, s, P[:,k]) for k in range(3)], axis=1)
    return P_new

def savgol_smooth(P, window=9, poly=3):
    try:
        from scipy.signal import savgol_filter
        w = max(3, int(window) | 1)
        if len(P) < w: return P.copy()
        Q = np.empty_like(P)
        for k in range(3):
            Q[:,k] = savgol_filter(P[:,k], window_length=w, polyorder=poly, mode='interp')
        return Q
    except Exception:
        return P.copy()

def pick_root_link(vertices_zyx, links, dm_mm_zyx, spacing_xyz,
                   deg_map, link_end_nodes,
                   mode='auto', hint_xyz=None, bbox_face='zmin'):
    """
    Pick root as the point closest to origin in z axis:
    - Find the skeleton vertex with minimum z (in millimeters).
    - Map that vertex to its link and return (root_link, root_node).
    """
    import numpy as np

    if len(vertices_zyx) == 0 or len(links) == 0:
        return 0, 0

    V_zyx = np.asarray(vertices_zyx, dtype=float)  # (N,3) [z_mm,y_mm,x_mm]
    # Find the index of vertex with smallest z value
    root_node = int(np.argmin(V_zyx[:, 0]))
    # Map node -> one link that contains it
    node2link = {}
    for lid, path in enumerate(links):
        for n in path:
            if n not in node2link:  # keep first assignment; any is fine
                node2link[n] = lid

    root_link = int(node2link.get(root_node, 0))
    return root_link, root_node


def build_link_tree(vertices_zyx, links, dm_mm_zyx, spacing_xyz,
                    root_mode='auto', root_hint_xyz=None, bbox_face='zmin',
                    generation_base=1):
    # 1) Build node-degree over the skeleton graph
    all_edges = []
    for path in links:
        all_edges += list(zip(path[:-1], path[1:]))
    adj = defaultdict(list)
    for u,v in all_edges:
        adj[u].append(v); adj[v].append(u)
    deg = {i: len(adj[i]) for i in adj}

    # 2) Endpoint pair (per link)
    link_end_nodes = [(path[0], path[-1]) for path in links]

    # 3) Link adjacency via shared endpoints
    node2links = defaultdict(list)
    for lid, (a,b) in enumerate(link_end_nodes):
        node2links[a].append(lid); node2links[b].append(lid)
    link_adj = defaultdict(set)
    for node, lids in node2links.items():
        for i in range(len(lids)):
            for j in range(i+1, len(lids)):
                u, v = lids[i], lids[j]
                link_adj[u].add(v); link_adj[v].add(u)

    # 4) Robust root pick
    root_link, root_node = pick_root_link(
        vertices_zyx, links, dm_mm_zyx, spacing_xyz,
        deg, link_end_nodes,
        mode=root_mode, hint_xyz=root_hint_xyz, bbox_face=bbox_face
    )
    
    # 5) Parent & generation by BFS from the root (pure topological depth)
    parent = {root_link: -1}
    generation0 = {root_link: 0}
    q = deque([root_link])
    visited = {root_link}
    while q:
        u = q.popleft()
        for v in sorted(link_adj[u]):  # sort for determinism
            if v in visited: continue
            parent[v] = u
            generation0[v] = generation0[u] + 1
            visited.add(v); q.append(v)

    # 6) Apply desired base (1 = Weibel-like with trachea=1)
    generation = {lid: (g + generation_base) for lid, g in generation0.items()}

    # (optional) sanity print: expected max ~10–12 for cropped lungs
    try:
        from collections import Counter
        hist = Counter(generation.values())
        print("Generation histogram:", dict(sorted(hist.items())))
        print("Max generation:", max(generation.values()))
    except Exception:
        pass
    return parent, generation, root_link

def compute_children(parent, n_links):
    children = defaultdict(list)
    for lid in range(n_links):
        p = parent.get(lid, None)
        if p is not None and p >= 0:
            children[p].append(lid)
    for k in list(children.keys()):
        children[k] = sorted(children[k])
    return children

def dfs_order(root_link, children, n_links):
    order = []
    seen = set()
    def rec(u):
        seen.add(u)
        order.append(u)
        for v in children.get(u, []):
            rec(v)
    rec(root_link)
    for lid in range(n_links):
        if lid not in seen:
            rec(lid)
    return order

def write_segment_cp_csv(path, vertices_zyx, links, proximal_node, order_lids,
                         parent,
                         per_point_ellipse, mask_zyx, spacing_xyz,
                         slab_half_mm=0.8, roi_radius_mm=6.0,
                         smooth=False, step_mm=0.6, savgol_w=9, savgol_p=3):
    id_map = {old: (i+1) for i, old in enumerate(order_lids)}
    last_end = {}
    eps = 1e-9

    with open(path, 'w', newline='') as f:
        cols = ['LinkId','x','y','z']
        if per_point_ellipse:
            cols += ['MinorInnerDiameter','MajorInnerDiameter','InnerCrossSecArea']
        writer = csv.writer(f)
        writer.writerow(cols)

        for old_lid in order_lids:
            nodes = links[old_lid]
            new_id = id_map[old_lid]

            # initial orientation by proximal endpoint (kept)
            prox_node = proximal_node.get(old_lid, nodes[0])
            if nodes[0] != prox_node and nodes[-1] == prox_node:
                nodes = list(reversed(nodes))

            # build polyline (XYZ mm)
            P = np.array([vertices_zyx[n][::-1] for n in nodes], float)
            if smooth:
                P = resample_polyline(P, step_mm=step_mm)
                P = savgol_smooth(P, window=savgol_w, poly=savgol_p)

            # parent continuity + +Z preference
            old_parent = parent.get(old_lid, -1)
            parent_id_new = id_map.get(old_parent, -1) if old_parent >= 0 else -1

            if parent_id_new > 0 and parent_id_new in last_end:
                parent_end = last_end[parent_id_new]
                cand0 = P
                cand1 = P[::-1]
                def score(Q):
                    dz_ok = (Q[-1, 2] - Q[0, 2]) >= -1e-6
                    cont = np.linalg.norm(Q[0] - parent_end)
                    # prefer +Z first, then smaller distance to parent_end
                    return (1 if dz_ok else 0, -cont)
                P = cand0 if score(cand0) >= score(cand1) else cand1
                # snap start to parent end
                P[0] = parent_end
            else:
                # root or no parent: enforce +Z
                if P[-1, 2] < P[0, 2] - eps:
                    P = P[::-1]

            # write points
            for i in range(len(P)):
                x, y, z = P[i]
                row = [new_id, f'{x:.6f}', f'{y:.6f}', f'{z:.6f}']
                if per_point_ellipse:
                    if i+1 < len(P): p0, p1 = P[i], P[i+1]
                    elif i > 0:      p0, p1 = P[i-1], P[i]
                    else:            p0, p1 = P[i], P[i] + np.array([1e-3,0,0])
                    d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-9)
                    pts2d = _gather_cross_section_points(
                        mask_zyx, spacing_xyz, np.array([x,y,z], float), d,
                        slab_half_mm=slab_half_mm, roi_radius_mm=roi_radius_mm
                    )
                    maj, minr = _ellipse_axes_from_points2d(pts2d)
                    area = np.pi * (0.5*maj) * (0.5*minr) if (maj==maj and minr==minr) else 0.0
                    row += [f'{minr:.6f}', f'{maj:.6f}', f'{area:.6f}']
                writer.writerow(row)

            last_end[new_id] = P[-1]

def write_segment_csv(path, vertices_zyx, links, parent, generation, order_lids,
                      proximal_node, dm_mm_zyx, spacing_xyz, mask_zyx,
                      compute_entry_axes=True, slab_half_mm=1.0, roi_radius_mm=8.0,
                      smooth=False, step_mm=0.6, savgol_w=9, savgol_p=3):
    id_map = {old: (i+1) for i, old in enumerate(order_lids)}
    last_end = {}
    eps = 1e-9

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','parentId','generation','centerLineLength',
                         'startPoint_x','startPoint_y','startPoint_z',
                         'MajorAxisLengthAtEntry','MinorAxisLengthAtEntry'])

        for old_lid in order_lids:
            nodes = links[old_lid]

            # initial orientation by proximal endpoint (kept)
            prox_node = proximal_node.get(old_lid, nodes[0])
            if nodes[0] != prox_node and nodes[-1] == prox_node:
                nodes = list(reversed(nodes))

            # build polyline (XYZ mm)
            P = np.array([vertices_zyx[n][::-1] for n in nodes], float)
            if smooth:
                P = resample_polyline(P, step_mm=step_mm)
                P = savgol_smooth(P, window=savgol_w, poly=savgol_p)
             
            new_id = id_map[old_lid]
            old_parent = parent.get(old_lid, -1)
            parent_id_new = id_map.get(old_parent, -1) if old_parent >= 0 else -1

            # parent continuity + +Z preference
            if parent_id_new > 0 and parent_id_new in last_end:
                parent_end = last_end[parent_id_new]
                cand0 = P
                cand1 = P[::-1]
                def score(Q):
                    dz_ok = (Q[-1, 2] - Q[0, 2]) >= -1e-6
                    cont = np.linalg.norm(Q[0] - parent_end)
                    return (1 if dz_ok else 0, -cont)
                P = cand0 if score(cand0) >= score(cand1) else cand1
                # snap start to parent end
                P[0] = parent_end
            else:
                # root or no parent: enforce +Z
                if P[-1, 2] < P[0, 2] - eps:
                    P = P[::-1]

            start_xyz = P[0]
            length = polyline_length(P)

            if compute_entry_axes:
                if len(P) > 1: p0, p1 = P[0], P[1]
                else:          p0, p1 = P[0], P[0] + np.array([1e-3,0,0])
                d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-9)
                pts2d = _gather_cross_section_points(
                    mask_zyx, spacing_xyz, p0, d,
                    slab_half_mm=slab_half_mm, roi_radius_mm=roi_radius_mm
                )
                maj, minr = _ellipse_axes_from_points2d(pts2d)
            else:
                maj, minr = (-1.0, -1.0)

            parent_id = parent_id_new
            gen = generation.get(old_lid, 0) + 1  # keep your 1-based output

            writer.writerow([new_id, parent_id, gen, f'{length:.6f}',
                             f'{start_xyz[0]:.6f}', f'{start_xyz[1]:.6f}', f'{start_xyz[2]:.6f}',
                             f'{maj:.6f}', f'{minr:.6f}'])

            last_end[new_id] = P[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--mask', required=True, help='Path to airway lumen mask (NIfTI .nii/.nii.gz)')
    ap.add_argument('-o','--out_dir', default='.', help='Output directory')
    ap.add_argument('--per_point_ellipse', action='store_true', help='Compute per-point diameters/area (slower)')
    ap.add_argument('--compute_entry_axes', action='store_true', help='Compute entry ellipse axes for each link')
    ap.add_argument('--entry_slab_half_mm', type=float, default=1.0, help='Entry slab half-thickness (mm)')
    ap.add_argument('--entry_roi_radius_mm', type=float, default=8.0, help='Entry slab ROI radius (mm)')
    ap.add_argument('--smooth', action='store_true', help='Apply resample + Savitzky-Golay smoothing')
    ap.add_argument('--resample_step_mm', type=float, default=1.2, help='Target resample step (mm)')
    ap.add_argument('--savgol_window', type=int, default=5, help='Savitzky-Golay window length (odd)')
    ap.add_argument('--savgol_poly', type=int, default=3, help='Savitzky-Golay polynomial order')
    ap.add_argument('--order', choices=['dfs','original'], default='dfs',
                    help='Order of links; DFS remaps LinkIds to depth-first order (default).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_cp_file = os.path.join(args.out_dir, 'segment_cp.csv')
    seg_param_file = os.path.join(args.out_dir, 'segment.csv')

    nii = nib.load(args.mask)
    affine = nii.affine         # 4×4
    mask_xyz = (nii.get_fdata() > 0).astype(np.uint8)
    # mask_xyz = mask_xyz[:, ::-1, ::-1]
    # breakpoint()
    mask_zyx = np.transpose(mask_xyz, (2,1,0))
    spacing_xyz = nii.header.get_zooms()[:3]
    sx, sy, sz = spacing_xyz 
    
    # vertices_zyx, edges = skeletonize_with_kimimaro(mask_zyx, spacing_xyz)
    skel_pkl_file = os.path.join(args.out_dir, 'kimimaro_skel.pkl')
    if os.path.exists(skel_pkl_file):
        with open(skel_pkl_file, 'rb') as fp:
            vertices_zyx, edges = pickle.load(fp)
        print(f"Loaded skeleton from {skel_pkl_file}")
    else:
        vertices_zyx, edges = skeletonize_with_kimimaro(mask_zyx, spacing_xyz)
        # Rotate vertices by 180 degrees around x-axis, then shift z so min=0 (load precomputed z shift if available)
        verts = vertices_zyx[:, ::-1].copy()  # (N, 3) in (x, y, z)
        R = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
        verts = verts @ R.T
        z_translation_file = os.path.join(args.out_dir, 'z_translation.txt')
        if os.path.exists(z_translation_file):
            with open(z_translation_file, 'r') as f:
                z_shift = float(f.readline().strip())
            verts[:, 2] -= z_shift*1000
            print(f"Applied loaded z translation: {z_shift:.6f} (from {z_translation_file})")
        else:
            print(f"(no z_translation.txt found)")
        vertices_zyx = verts[:, ::-1]  # back to (z, y, x)
        with open(skel_pkl_file, 'wb') as fp:
            pickle.dump((vertices_zyx, edges), fp)
        print(f"Saved skeleton to {skel_pkl_file}")


    start_time = time.time()
    links = split_into_links(vertices_zyx, edges)
    print(f"split_into_links run time: {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    dm_mm_zyx = compute_edt_mm(mask_zyx, spacing_xyz)
    print(f"compute_edt_mm run time: {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    parent, generation, root_link = build_link_tree(vertices_zyx, links, dm_mm_zyx, spacing_xyz)
    print(f"build_link_tree run time: {time.time() - start_time:.3f} seconds")

    # proximal endpoint per link
    start_time = time.time()
    link_end_nodes = [(path[0], path[-1]) for path in links]
    proximal_node = {}
    for lid, (a, b) in enumerate(link_end_nodes):
        p = parent.get(lid, -1)
        if p >= 0:
            pa, pb = link_end_nodes[p]
            if a == pa or a == pb: proximal_node[lid] = a
            elif b == pa or b == pb: proximal_node[lid] = b
            else:
                ra = sample_radius_mm(dm_mm_zyx, vertices_zyx[a], spacing_xyz)
                rb = sample_radius_mm(dm_mm_zyx, vertices_zyx[b], spacing_xyz)
                proximal_node[lid] = a if ra >= rb else b
        else:
            ra = sample_radius_mm(dm_mm_zyx, vertices_zyx[a], spacing_xyz)
            rb = sample_radius_mm(dm_mm_zyx, vertices_zyx[b], spacing_xyz)
            proximal_node[lid] = a if ra >= rb else b
    print(f"proximal_node computation run time: {time.time() - start_time:.3f} seconds")

    n_links = len(links)
    if args.order == 'dfs':
        start_time = time.time()
        children = compute_children(parent, n_links)
        ordered_lids = dfs_order(root_link, children, n_links)
        print(f"dfs_order run time: {time.time() - start_time:.3f} seconds")
    else:
        ordered_lids = list(range(n_links))

    start_time = time.time()
    write_segment_cp_csv(seg_cp_file, vertices_zyx, links, proximal_node, ordered_lids,
                        parent,
                        per_point_ellipse=args.per_point_ellipse,
                        mask_zyx=mask_zyx, spacing_xyz=spacing_xyz,
                        slab_half_mm=args.entry_slab_half_mm, roi_radius_mm=args.entry_roi_radius_mm,
                        smooth=args.smooth, step_mm=args.resample_step_mm,
                        savgol_w=args.savgol_window, savgol_p=args.savgol_poly)
    print(f"write_segment_cp_csv run time: {time.time() - start_time:.3f} seconds")

    start_time = time.time()
    write_segment_csv(seg_param_file, vertices_zyx, links, parent, generation, ordered_lids,
                        proximal_node, dm_mm_zyx, spacing_xyz, mask_zyx,
                        compute_entry_axes=args.compute_entry_axes,
                        slab_half_mm=args.entry_slab_half_mm, roi_radius_mm=args.entry_roi_radius_mm,
                        smooth=args.smooth, step_mm=args.resample_step_mm,
                        savgol_w=args.savgol_window, savgol_p=args.savgol_poly)
    print(f"write_segment_csv run time: {time.time() - start_time:.3f} seconds")

    print(f"Wrote: {seg_cp_file}")
    print(f"Wrote: {seg_param_file}")
    print("Order:", args.order, "| Units: millimeters (mm)")

if __name__ == '__main__':
    main()
