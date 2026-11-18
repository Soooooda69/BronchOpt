import pandas as pd
import argparse
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from generation.common import create_extrinsics_matrix, create_vectors_from_matrix


def plot_and_save_trajectory(
    poses_list,
    subsampling_factor=5,
    save_name="trajectory.ply",
    draw_connections=True,
    connection_color=[0, 1, 0],
    show=False,
    jump_multiplier=6.0,   # robust threshold = jump_multiplier * median step
):
    import open3d as o3d  # only needed for the centerlines.ply visualisation
    def convert_lineset_to_mesh(lineset, radius=1e-4):
        """
        Converts a LineSet object to a TriangleMesh by creating a cylinder mesh for each line.
        """
        combined_mesh = o3d.geometry.TriangleMesh()
        pts = np.asarray(lineset.points)
        lns = np.asarray(lineset.lines, dtype=int)
        if len(lns) == 0:
            return combined_mesh

        z_axis = np.array([0.0, 0.0, 1.0])
        for start_idx, end_idx in lns:
            start = pts[start_idx]
            end = pts[end_idx]
            seg = end - start
            seg_len = np.linalg.norm(seg)
            if seg_len <= 0:
                continue

            # cylinder centered at origin, axis along +Z, height = seg_len
            line_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=seg_len)

            # rotate +Z to segment direction (Rodrigues)
            dir_vec = seg / seg_len
            if not np.allclose(dir_vec, z_axis):
                v = np.cross(z_axis, dir_vec)
                c = float(np.dot(z_axis, dir_vec))
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c + 1e-12))
                line_mesh.rotate(R, center=(0, 0, 0))

            # move cylinder so its center is at the segment midpoint
            mid = (start + end) * 0.5
            line_mesh.translate(mid)
            line_mesh.paint_uniform_color(connection_color)
            combined_mesh += line_mesh

        return combined_mesh

    # subsample poses → points
    pose_array = poses_list
    sub_idx = np.zeros(len(pose_array), dtype=bool)
    points = []
    for j, cam_pose in enumerate(pose_array):
        if j % subsampling_factor != 0:
            continue
        transl = cam_pose[:3, 3]
        points.append(transl)
        sub_idx[j] = True
    transformation_matrices = [pose_array[i] for i in range(len(pose_array)) if sub_idx[i]]

    # draw coordinate frames along the path
    trajectory = None
    for i, pose in enumerate(transformation_matrices):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.003)
        mesh_frame.transform(pose)
        if i == 0:
            mesh_frame.paint_uniform_color([1, 0, 0])  # red for start
            trajectory = mesh_frame
        else:
            trajectory += mesh_frame

    # build filtered connections that avoid "jumps"
    line_set = None
    if draw_connections and len(points) >= 2:
        P = np.asarray(points, dtype=float)
        dists = np.linalg.norm(P[1:] - P[:-1], axis=1)

        nonzero = dists[dists > 1e-9]
        if nonzero.size:
            med = float(np.median(nonzero))
            # robust threshold; increases if sampling is sparse
            thr = max(jump_multiplier * med, 10 * 1e-6)
        else:
            thr = 0.0

        lines = []
        for i in range(len(P) - 1):
            # only connect if step is not a big "jump"
            if dists[i] <= thr:
                lines.append([i, i + 1])

        if len(lines) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(P)
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(connection_color, float), (len(lines), 1))
            )
            if show:
                o3d.visualization.draw_geometries([trajectory, line_set])
        else:
            if show:
                o3d.visualization.draw_geometries([trajectory])

    else:
        if show:
            o3d.visualization.draw_geometries([trajectory])

    # bake line tubes into the mesh (only if we actually have filtered lines)
    if line_set is not None:
        line_mesh = convert_lineset_to_mesh(line_set, radius=1e-4)
        trajectory += line_mesh

    o3d.io.write_triangle_mesh(save_name, trajectory)

# File paths
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=str, required=True, help="Number identifier for the video and timestamp files.")
    return parser.parse_args()

def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)[1:]  # Exclude the first row
        # Modify the data to keep 3 decimals
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = format(float(data[i][j]), '.3f')
                
    with open(txt_file, 'w') as file:
        for row in data:
            del row[1:3]  # Delete the second and third columns
            file.write('\t'.join(row) + '\n')

def compute_rotation(row):
    current_pos = np.array([row['x'], row['y'], row['z']])
    next_pos = np.array([row['x_next'], row['y_next'], row['z_next']])
    
    # Compute forward vector
    forward = next_pos - current_pos
    norm = np.linalg.norm(forward)
   
    if norm == 0:
        forward = np.array([0, 0, 1])  # Default forward direction
    else:
        forward = forward / norm
    
    # Define the up vector (you may adjust this based on your application)
    up = np.array([0, 1, 0])
    
    # Check if forward and up are colinear
    if np.allclose(forward, up) or np.allclose(forward, -up):
        up = np.array([0, 0, 1])  # Change up vector to avoid gimbal lock
    
    # Compute right vector
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        right = np.array([1, 0, 0])  # Default right direction
    else:
        right = right / right_norm
    
    # Recompute up vector to ensure orthogonality
    up = np.cross(forward, right)
    
    # Create rotation matrix
    rot_matrix = np.vstack([right, up, forward]).T  # 3x3 rotation matrix
    
    # Convert rotation matrix to quaternion
    rot = R.from_matrix(rot_matrix)
    quat = rot.as_quat()  # Returns in [x, y, z, w] format
    return pd.Series({'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3]})

def quaternion_angle_degrees(q1, q2):
    """
    Computes the angle (in degrees) between two quaternions q1 and q2.
    Both q1 and q2 are assumed to be [qx, qy, qz, qw].
    """
    # Dot product
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    
    # Clamp the dot product to avoid numerical issues outside [-1, 1].
    dot = max(min(dot, 1.0), -1.0)
    
    # Angle = 2 * arccos(|dot|)
    # Use absolute value because q and -q represent the same orientation.
    angle_radians = 2.0 * np.arccos(abs(dot))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def find_leaf_ids(parent_to_children):
    # All parents are keys; all children are values in nested lists
    all_parents = set(parent_to_children.keys())
    all_children = set(child for children in parent_to_children.values() for child in children)
    
    # Leaves are children that never appear as a parent
    leaf_ids = all_children - all_parents
    return sorted(leaf_ids)

def compute_distance_matrix(parent_to_children):
    # Build child_to_parent dictionary
    child_to_parent = {}
    for parent in parent_to_children:
        for child in parent_to_children[parent]:
            child_to_parent[child] = parent

    # Collect all unique nodes
    nodes = list(child_to_parent.keys())
    nodes = [int(n) for n in nodes]  # to int
    max_node = max(nodes)
    n = max_node + 1
    distance_matrix = np.full((n, n), np.inf)

    # Precompute depths for each node
    depth = {}
    def get_depth(node):
        if node not in depth:
            parent = child_to_parent.get(node)
            if parent is None:
                depth[node] = 0
            else:
                depth[node] = get_depth(parent) + 1
        return depth[node]
    for node in nodes:
        get_depth(node)

    # Function to find the lowest common ancestor (LCA)
    def find_lca(u, v):
        # Bring both nodes to the same depth
        du, dv = depth[u], depth[v]
        while du > dv:
            u = child_to_parent[u]
            du -= 1
        while dv > du:
            v = child_to_parent[v]
            dv -= 1
        # Now move up until they meet
        while u != v:
            u = child_to_parent[u]
            v = child_to_parent[v]
        return u

    # Compute all pairwise distances
    for u in nodes:
        for v in nodes:
            lca_node = find_lca(u, v)
            dist = depth[u] + depth[v] - 2 * depth[lca_node]
            distance_matrix[u][v] = dist
            distance_matrix[v][u] = dist

    return distance_matrix

def save_centerlines(num):
    # ---------------------------------------------
    # 0. Helper Functions and Constants
    # ---------------------------------------------
    mm_to_m = lambda x: x / 1000

    def apply_mm_to_m(df, cols):
        for col in cols:
            df[col] = mm_to_m(df[col])
        return df

    # ---------------------------------------------
    # 1. File Paths and Input
    # ---------------------------------------------
    if isinstance(num, dict):
        segment_csv_file = num["segment_csv_file"]
        seg_param_csv_file = num["seg_param_csv_file"]
        output_file = num["centerline_output_file"]
        out_ply_file = num["centerline_out_ply_file"]
        dist_mtx_file = num["distance_matrix_file"]
    else:
        # parent = os.path.relpath(f"{num}/..")
        # num = os.path.basename(num)
        segment_csv_file = os.path.join(num, "segment_cp.csv")
        seg_param_csv_file = os.path.join(num, "segment.csv")
        output_file = os.path.join(num, "centerlines.txt")
        out_ply_file = os.path.join(num, "centerlines.ply")
        dist_mtx_file = os.path.join(num, "distance_matrix.npy")

    # ---------------------------------------------
    # 2. Load Data and Initial Processing
    # ---------------------------------------------
    segments = pd.read_csv(segment_csv_file)
    seg_param = pd.read_csv(seg_param_csv_file)

    # Sort and prepare columns
    segments = segments.sort_values(by="LinkId")
    segments["id"] = segments["LinkId"]
    # Convert coordinates from mm to m
    segments = apply_mm_to_m(segments, ["x", "y", "z"])
    # Initialize quaternion columns
    for q in ["qx", "qy", "qz", "qw"]:
        segments[q] = 0 if q != "qw" else 1
    segments = segments[["id", "x", "y", "z", "qx", "qy", "qz", "qw"]]

    # Merge with segment parameters and convert units
    segments = segments.merge(
        seg_param[['id', 'parentId', 'generation', 'centerLineLength', 'startPoint_x', 'startPoint_y',
                   'startPoint_z', 'MajorAxisLengthAtEntry', 'MinorAxisLengthAtEntry']],
        left_on='id', right_on='id', how='left', suffixes=('', '_seg_param')
    )
    segments = apply_mm_to_m(segments, ['startPoint_x', 'startPoint_y', 'startPoint_z',
                                        'MajorAxisLengthAtEntry', 'MinorAxisLengthAtEntry'])

    # ---------------------------------------------
    # 3. Build Parent-to-Children Mapping and Distance Matrix
    # ---------------------------------------------
    parent_to_children = defaultdict(list)
    for _, row in seg_param.iterrows():
        parent_to_children[row['parentId']].append(row['id'])
    dis_mtx = compute_distance_matrix(parent_to_children)
    np.save(dist_mtx_file, dis_mtx)
    leaf_ids = find_leaf_ids(parent_to_children)
    np.save(os.path.join(os.path.dirname(dist_mtx_file), "leaf_ids.npy"), leaf_ids)

    # ---------------------------------------------
    # 4. DFS Traversal to Sort Segments by Branch
    # ---------------------------------------------
    root_ids = seg_param[seg_param['parentId'] == -1]['id'].tolist()
    def dfs(current_id, visited, sorted_ids):
        if current_id in visited:
            return
        visited.add(current_id)
        sorted_ids.append(current_id)
        for child in parent_to_children[current_id]:
            dfs(child, visited, sorted_ids)
    visited = set()
    sorted_branch_ids = []
    for root_id in root_ids:
        dfs(root_id, visited, sorted_branch_ids)

    # ---------------------------------------------
    # 5. Sort Rows Within Each Branch by Distance from Start
    # ---------------------------------------------
    sorted_rows = []
    for bid in sorted_branch_ids:
        branch_rows = segments.loc[segments['id'] == bid].copy()
        branch_rows['distance'] = np.linalg.norm(
            branch_rows[["x", "y", "z"]].values - branch_rows[["startPoint_x", "startPoint_y", "startPoint_z"]].values,
            axis=1
        )
        branch_rows = branch_rows.sort_values('distance')
        sorted_rows.append(branch_rows)
    segments_sorted = pd.concat(sorted_rows, ignore_index=True)

    # ---------------------------------------------
    # 6. Compute Forward Vectors and Rotations
    # ---------------------------------------------
    # Compute next positions, fill for last row in each group
    for axis in ['x', 'y', 'z']:
        segments_sorted[f"{axis}_next"] = segments_sorted.groupby('id')[axis].shift(-1)
        segments_sorted[f"{axis}_next"] = segments_sorted[f"{axis}_next"].fillna(segments_sorted[axis])
    # Compute new rotations (qx, qy, qz, qw)
    rotations = segments_sorted.apply(compute_rotation, axis=1)
    segments_sorted[['qx', 'qy', 'qz', 'qw']] = rotations

    # Outlier check and clamping for large rotation jumps
    THRESHOLD_DEG = 45.0
    for link_id, group in segments_sorted.groupby('id', sort=False):
        row_indices = group.index.to_list()
        for i in range(1, len(row_indices)):
            idx_prev, idx_cur = row_indices[i - 1], row_indices[i]
            q_prev = segments_sorted.loc[idx_prev, ['qx','qy','qz','qw']].values
            q_cur = segments_sorted.loc[idx_cur, ['qx','qy','qz','qw']].values
            angle_diff = quaternion_angle_degrees(q_prev, q_cur)
            if angle_diff > THRESHOLD_DEG:
                # print(f"[ID={link_id}] Large rotation jump at row index {idx_cur}: {angle_diff:.2f} deg. Clamping.")
                segments_sorted.loc[idx_cur, ['qx','qy','qz','qw']] = q_prev

    # ---------------------------------------------
    # 7. Clean Up for Output
    # ---------------------------------------------
    segments_cleaned = segments_sorted.drop(columns=['distance', 'x_next', 'y_next', 'z_next'])

    # ---------------------------------------------
    # 8. Transform Segments to Output Format
    # ---------------------------------------------
    convert_rotation = np.eye(4)
    rot_mat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    convert_rotation[:3, :3] = rot_mat

    def transform_row(row):
        id, x, y, z, qx, qy, qz, qw = row['id'], row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]
        majorAxis, minorAxis = row['MajorAxisLengthAtEntry'], row['MinorAxisLengthAtEntry']
        pid, gen = row['parentId'], row['generation']
        extrinsics = create_extrinsics_matrix(x, y, z, qx, qy, qz, qw)
        # extrinsics = convert_rotation @ extrinsics
        # extrinsics = extrinsics @ convert_rotation
        vec = np.concatenate([[int(id)], create_vectors_from_matrix(extrinsics), [majorAxis], [minorAxis], [int(pid)], [int(gen)]])
        return vec

    segments_transformed = np.vstack([transform_row(row) for _, row in segments_cleaned.iterrows()])
    columns = ["id", "x", "y", "z", "qx", "qy", "qz", "qw", "majorAxis", "minorAxis", "parentId", "generation"]
    segments_transformed_df = pd.DataFrame(segments_transformed, columns=columns)
    segments_transformed_df[["id", "parentId", "generation"]] = segments_transformed_df[["id", "parentId", "generation"]].astype(int)

    # ---------------------------------------------
    # 9. Save Results
    # ---------------------------------------------
    segments_transformed_df.to_csv(output_file, sep=" ", index=False)

    # ---------------------------------------------
    # 10. Generate and Save Trajectory Mesh
    # ---------------------------------------------
    extrinsics_list = []
    # Use DataFrame for parsing, skip header
    for _, row in segments_transformed_df.iterrows():
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        q_x, q_y, q_z, q_w = float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
        extrinsics = create_extrinsics_matrix(x, y, z, q_x, q_y, q_z, q_w)
        extrinsics_list.append(extrinsics)
    plot_and_save_trajectory(extrinsics_list, save_name=out_ply_file)
    print(f"Centerlines saved to {output_file}")
    print(f"Trajectory mesh saved to {out_ply_file}")

if __name__ == "__main__":
    args = parse_arguments()
    num = args.num
    save_centerlines(num)
        