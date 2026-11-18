import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import marching_cubes
import pyvista as pv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Generate airway mesh from mask image')
    parser.add_argument('-i', '--input_path', help='Path to the input airway mask image file')
    parser.add_argument('-o', '--output', default='aero_airway.ply', 
                       help='Output path for the PLY mesh file (default: aero_airway.ply)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # --- 1) Resample to isotropic, smaller voxels (keeps topology, adds sampling density) ---
    iso_spacing = (0.5, 0.5, 0.5)  # try 0.6 first if memory is tight

    img_sitk = sitk.ReadImage(args.input_path)
    orig_size    = np.array(img_sitk.GetSize(), dtype=int)
    orig_spacing = np.array(img_sitk.GetSpacing(), dtype=float)

    new_size = np.round(orig_size * (orig_spacing / np.array(iso_spacing))).astype(int).tolist()
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # for labels
    resample.SetOutputSpacing(iso_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img_sitk.GetOrigin())
    resample.SetOutputDirection(img_sitk.GetDirection())
    mask_iso = resample.Execute(img_sitk)

    mask_np = sitk.GetArrayFromImage(mask_iso).astype(np.uint8)  # z,y,x

    # --- 2) High-fidelity marching cubes (no decimation) ---
    # marching_cubes expects spacing in (z,y,x)
    verts_zyx, faces, normals, _ = marching_cubes(mask_np, level=0.5, step_size=1,
                                                  spacing=iso_spacing[::-1])

    # Convert verts to (x,y,z)
    verts = verts_zyx[:, ::-1]
    verts += np.array(iso_spacing) * 0.5
    # skimage returns faces as (n_tris, 3) indices; PyVista wants a flat array with leading 3s
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32),
                          faces.astype(np.int32)]).ravel()

    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.clean()  # remove degenerate cells/isolated pieces

    # --- 3) Taubin smoothing (preserves volume/shape better than Laplacian) ---
    # passband ~0.03–0.1; lower = more smoothing with less shrinkage
    mesh = mesh.smooth_taubin(n_iter=60, pass_band=0.05)

    # Convert vertices from mm to m
    mesh.points /= 1000.0
    # Rotate mesh 180 degrees around x-axis
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
    mesh.points = mesh.points @ R.T
    # Translate so that z axis is positive (shift so min z = 0), and save translation
    z_shift = mesh.points[:, 2].min()
    mesh.points[:, 2] -= z_shift
    # Ensure the base folder of args.output exists
    os.makedirs(args.output, exist_ok=True)
    mesh.save(os.path.join(args.output, 'localMedium.ply'))
    # Save the z axis translation value
    with open(os.path.join(args.output, 'z_translation.txt'), 'w') as f:
        f.write(f"{z_shift:.6f}\n")
    print(mesh.n_points, "verts |", mesh.n_cells, "tris")
    print(f"Mesh saved to: {os.path.join(args.output, 'localMedium.ply')}")
    print(f"z translation applied (m): {z_shift:.6f} (saved to z_translation.txt)")


if __name__ == "__main__":
    main()
