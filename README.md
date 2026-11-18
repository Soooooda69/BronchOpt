# BronchOpt: Vision-Based Pose Optimization with Fine-Tuned Foundation Models for Accurate Bronchoscopy Navigation

Synthetic benchmark dataset for bronchoscopy navigation evaluation. This repository provides the evaluation framework mentioned in the paper for vision-based pose optimization methods in bronchoscopy navigation, addressing the lack of paired CT-endoscopy data for benchmarking.



## Overview

BronchOpt provides a synthetic benchmark dataset for evaluating bronchoscopy navigation algorithms. The dataset includes:

- **Synthetic bronchoscopy sequences**: 15 sequences (`bronch_real_aero1` through `bronch_real_aero15`) built from the public [AeroPath](https://github.com/raidionics/AeroPath) CT airway labels, with CycleGAN-translated RGB images, poses, and the airway mesh
- **Evaluation framework**: Tools for evaluating pose estimation methods using multiple metrics
- **Depth rendering**: GPU-accelerated RGB and depth rendering of the airway mesh at any pose
- **Data generation**: The scripts used to build the benchmark from the AeroPath labels (`generation/`)


## Our Quantitative Results

Below is a quantitative comparison between OffsetNet and our proposed BronchOpt method on the BronchOpt synthetic benchmark, evaluated across three difficulty levels. The metrics reported are: Disparity Similarity (DS, ↑), Normal Correlation (NC, ↑), Scale-Invariant Depth Error (SI, ↓), translation error in mm (↓), rotation error in radians (↓), and success rate (%). 

The initial camera poses (Init), sampled around the ground-truth views, show increasing difficulty across levels, with mean translation/rotation distances of **$1.94 \pm 2.03$ mm / $0.14 \pm 0.11$ rad** for *Easy*, **$4.29 \pm 2.68$ mm / $0.34 \pm 0.15$ rad** for *Medium*, and **$6.96 \pm 3.09$ mm / $0.49 \pm 0.25$ rad** for *Hard* cases. The reported metrics are computed only over successful cases.

| Levels  | Models                     | DS $\uparrow$           | NC $\uparrow$           | SI $\downarrow$         | Trans. Err $\downarrow$ (mm) | Rot. Err $\downarrow$ (rad) | Success Rate (%) |
|---------|---------------------------|-------------------------|-------------------------|-------------------------|------------------------------|-----------------------------|-----------------|
| Easy    | OffsetNet | 0.76 $\pm$ 0.16         | 0.54 $\pm$ 0.20         | 0.58 $\pm$ 1.12         | 4.60 $\pm$ 2.52              | 0.15 $\pm$ 0.10             | 37.1            |
|         | **BronchOpt (Ours)**       | **0.94 $\pm$ 0.07**     | **0.74 $\pm$ 0.11**     | **0.08 $\pm$ 0.28**     | **1.81 $\pm$ 2.09**          | **0.13 $\pm$ 0.09**         | **94.7**        |
| Medium  | OffsetNet | 0.68 $\pm$ 0.16         | 0.44 $\pm$ 0.19         | 0.75 $\pm$ 1.01         | 6.17 $\pm$ 3.13              | 0.34 $\pm$ 0.15             | 45.5            |
|         | **BronchOpt (Ours)**       | **0.92 $\pm$ 0.09**     | **0.69 $\pm$ 0.13**     | **0.11 $\pm$ 0.39**     | **2.98 $\pm$ 3.10**          | **0.21 $\pm$ 0.14**         | **96.8**        |
| Hard    | OffsetNet | 0.63 $\pm$ 0.16         | 0.37 $\pm$ 0.19         | 0.99 $\pm$ 1.15         | 8.09 $\pm$ 3.42              | 0.47 $\pm$ 0.24             | 56.5            |
|         | **BronchOpt (Ours)**       | **0.90 $\pm$ 0.11**     | **0.65 $\pm$ 0.16**     | **0.15 $\pm$ 0.32**     | **4.53 $\pm$ 3.93**          | **0.31 $\pm$ 0.24**         | **98.3**        |
| Average | OffsetNet | 0.70 $\pm$ 0.17         | 0.45 $\pm$ 0.20         | 0.72 $\pm$ 1.09         | 5.87 $\pm$ 3.22              | 0.28 $\pm$ 0.20             | 43.0            |
|         | **BronchOpt (Ours)**       | **0.93 $\pm$ 0.09**     | **0.71 $\pm$ 0.14**     | **0.10 $\pm$ 0.33**     | **2.65 $\pm$ 2.96**          | **0.19 $\pm$ 0.15**         | **96.0**        |

**Notes:**
- $\uparrow$ indicates higher is better; $\downarrow$ indicates lower is better.
- DS: Disparity Similarity, NC: Normal Correlation, SI: Scale-Invariant Depth Loss.
- All results are averages across all benchmarked sequences, and all metrics are computed on successful cases.


For more details, please refer to the [paper](https://arxiv.org/abs/2511.09443).

## Dataset Structure
**Dataset Download**

You can download the complete BronchOpt synthetic evaluation dataset from the following link:

[BronchOpt Dataset](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/hshu4_jh_edu/IgC08uQUfx9xRIruN6tB9KRlAceZhk3kLm_DLvcsaQIpeq4?e=9Vked6)

Extract the downloaded files into a folder (the examples below use `./processed`). Each sequence (e.g., `bronch_real_aero1`) is a self-contained subdirectory:

```
bronch_real_aero1/
├── eval_src.json          # evaluation pairs (anchor / sample keys, difficulty level, ...)
├── pose_dict.json         # ground-truth pose of every view
├── fake_images/           # CycleGAN-translated RGB images, 256x256 PNG
│   ├── 12.png             #   centerline node 12          (key 0_00012)
│   ├── 741/               #   samples around node 741
│   │   ├── 8.png          #     sample 8 of node 741      (key 1_00741_00008)
│   │   └── ...
│   └── ...
└── localMedium.ply        # airway surface mesh (metres); the renderer loads this file name
```

The release contains the translated images only. The CT-rendered RGB images (before translation) and the depth maps are not shipped: render them from `localMedium.ply` and the poses with `depth_renderer.py`, exactly as `eval_pose.py` does at evaluation time. The renderer is deterministic, so this reproduces the original renders (see [Conventions](#conventions) for the camera model and depth encoding).

## Conventions

**Poses.** `pose_dict.json` maps each key to `{"pose": [tx, ty, tz, qx, qy, qz, qw], "branch", "parent", "gen"}`:
- camera-to-world (`X_world = R · X_cam + t`), translations in metres;
- Hamilton quaternion with the scalar **last** (LieTorch `SE3` layout, SciPy `as_quat()` order);
- OpenCV camera axes: +x right, +y down, +z forward (viewing direction), right-handed.

The world frame is the mesh frame. For AeroPath, the mesh is built from the airway label resampled to 0.5 mm in voxel-index coordinates (the NIfTI origin and direction cosines are not applied), converted mm → m, rotated 180° about x, and shifted so that min z = 0 (`generation/aeropath_mesh.py`).

**Keys.** `0_XXXXX` is centerline node `XXXXX` (`fake_images/XXXXX.png`); `1_XXXXX_YYYYY` is perturbed sample `YYYYY` around node `XXXXX` (`fake_images/XXXXX/YYYYY.png`). File names drop the zero padding.

**Pairs (`eval_src.json`, list `"rel"`).**
- `anchor`: the target view. Its image is the query and its pose is the ground truth.
- `sample`: the initial pose, drawn around the same centerline node or an ancestor node (number of hops = round(|N(0,1)|)); pairs share ≥ 20 % of their visible mesh triangles.
- The relative pose is `T_rel = T_sample⁻¹ · T_anchor`, so `T_pred = T_init @ T_rel`.
- `level`: difficulty 1–3 (quantile bins of `pose_dis`; the top 1 % is dropped).
- `s_faces`: covisibility |F_anchor ∩ F_sample| / min(|F_anchor|, |F_sample|) over visible triangles.
- `pose_grad`: unit directions of the LieTorch twist log(T_sample⁻¹ · T_anchor), translation part (3) then rotation part (3); magnitudes are not stored.
- `pose_dis`: a heuristic difficulty score, not a distance: angle between the two world-frame translation vectors + rotation angle (rad) + 100 · | ‖t_s‖ − ‖t_a‖ | (m). It depends on the world origin; compute your own distances from `pose_dict.json`.
- `branch` / `parent` / `gen` (in `pose_dict.json`): airway segment id, parent segment id (−1 for the trachea), airway generation (trachea = 1). Samples inherit the labels of their node.

**Camera.** Renders are 200 × 200. `configs/config.yaml` holds the values the renderer is configured with (fx 110.21, fy 110.31, cx 100.58, cy 103.69), but the rendered images follow different OpenCV intrinsics, because the projection matrix mirrors the principal point vertically and the rasterizer samples pixel centres:

| image | fx | fy | cx | cy |
|---|---|---|---|---|
| render, 200 × 200 | 110.21 | 110.31 | 100.08 | 95.81 |
| `fake_images`, 256 × 256 | 141.07 | 141.20 | 128.24 | 122.78 |
| model input, 224 × 224 | 123.44 | 123.55 | 112.15 | 107.37 |

For a uniform resize by `s`: `fx' = s·fx`, `cx' = (cx + 0.5)·s − 0.5`. Ray-casting the mesh with the 200 × 200 values reproduces the rendered depth to a median of 0.001 mm.

**Images.** `fake_images` are the 200 × 200 renders resized to 256 × 256 (bicubic, no crop or flip) and translated by the CycleGAN; each output was min–max stretched to [0, 255], so brightness is not comparable across frames. The loaders resize images to 224 × 224.

**Depth.** `DepthRenderer` returns the rasterizer's clip-space z, `d = 1.0001·z − 2.0e-5`, i.e. metric z-depth `z = (d + 2.0e-5) / 1.0001` metres (z-depth, not ray length); 0 means no surface hit. The renderer's RGB output is in [0, 1]; the original renders were saved as `uint8(255 · rgb)`.

```python
import json, numpy as np, torch
from omegaconf import OmegaConf
from depth_renderer import DepthRenderer
from eval_pose import pose_vec_to_matrix

root = "./processed/bronch_real_aero1"
cfg = OmegaConf.load("configs/config.yaml"); cfg.data_root = root
renderer = DepthRenderer(cfg, mesh_path=f"{root}/localMedium.ply").to("cuda")
poses = json.load(open(f"{root}/pose_dict.json"))
key = next(iter(poses))
T = pose_vec_to_matrix(torch.tensor([poses[key]["pose"]], device="cuda"))   # [1,4,4] camera-to-world
d, rgb = renderer(T, color_render=True)                                    # [1,1,200,200], [1,3,200,200]
z = torch.where(d > 0, (d + 2.0e-5) / 1.0001, torch.zeros_like(d))         # metric z-depth (m)
```


## Installation

### Requirements

- Python 3.12
- PyTorch (with CUDA support)
- CUDA-capable GPU
- `hydra-core`, `omegaconf`, `trimesh`, `rtree`, `scipy`, `opencv-python`, `torchvision`, `tqdm`

### Setup

1. Clone the repository and initialize submodules:
```bash
git clone https://github.com/Soooooda69/BronchOpt.git
cd BronchOpt
git submodule update --init --recursive
```


2. Build and install [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for fast differentiable rendering (required for the depth renderer):
```bash
cd third_party/nvdiffrast
pip install .
cd ../..
```

If you encounter CUDA version issues during this step, ensure your environment matches the required CUDA toolkit for nvdiffrast, or consult their repository for troubleshooting instructions.

3. Build the CUDA extensions:
```bash
pip install -e .
```

This will compile the `lietorch_backends` CUDA extension required for SE(3) operations.

## Usage

### Evaluation

The evaluation script (`eval_pose.py`) evaluates pose estimation methods on the benchmark dataset. Run it from the repository root.

1. Configure the evaluation in `configs/config.yaml` (or override on the command line):
```yaml
device: 'cuda:0'
data_root: './processed/bronch_real_aero1'   # one sequence, or 'all'
dataset_root: './processed'                  # with data_root='all': every bronch_real_aero* folder here
output: 'output'
```

2. Run:
```bash
python eval_pose.py                                    # one sequence
python eval_pose.py data_root=all dataset_root=./processed
bash eval_pose.sh
```

With the built-in placeholder model (no correction), the predictions equal the initial poses, which gives the *Init* baseline.

For each pair, the script renders the ground-truth depth at the anchor pose, gives the model a render at the **initial** pose together with the anchor's CycleGAN image, applies the predicted correction, renders depth at the predicted pose, and scores it.

Results are written per sequence and level to `output/<sequence>/level_<1|2|3>/`:
- `res.txt`: per pair `anchor_key ours_t_mm ours_rot_rad init_t_mm init_rot_rad ours_success`, then mean ± std lines
- `dsim.txt`: per pair `anchor_key ours_ds ours_nc ours_si init_ds init_nc init_si`, then mean ± std lines
- `poses_pred.txt`, `poses_init.txt`, `poses_gt.txt`: `anchor_key tx ty tz qx qy qz qw` (camera-to-world, metres)

and `output/summary.txt` aggregates every evaluated sequence per level. Failed pairs carry `-1` for the *ours* metrics and are excluded from its mean ± std; *init* metrics cover all pairs.

### Evaluation Metrics

The framework evaluates methods using the following metrics:

1. **Disparity Similarity (DS)**: Cosine similarity between normalized depth maps
2. **Normal Correlation (NC)**: Scale-invariant normal correlation between predicted and ground truth depth maps. To stay comparable with the paper, `metrics.normal_correlation` defaults to the intrinsics the published numbers were computed with (`LEGACY_NC_INTRINSICS`, which has `fx = 1e-6` from an argument-order slip in the original code); pass `intrinsics=` for the geometrically correct version.
3. **Scale-Invariant Depth Loss (SI)**: Scale-invariant depth error (Eigen et al. 2014)
4. **Translation Distance**: Euclidean distance between predicted and ground truth camera positions (mm)
5. **Rotation Distance**: Geodesic distance on SO(3) between predicted and ground truth rotations (rad)
6. **Success**: a prediction fails when its camera centre leaves the airway: the SDF barrier of the paper exceeds 1, i.e. the centre is outside the mesh or less than about 0.07 mm inside the wall. The paper evaluated the SDF on a precomputed 160³ grid; `eval_pose.py` evaluates it on the mesh directly (`metrics.MeshSDF`). On one airway the two agreed on 98.9 % of decisions; the disagreements were within about 0.2 mm of the wall.

### Integrating Your Model

Replace the placeholder in `build_model` in `eval_pose.py`:
```python
def build_model(cfg):
    model = YourPoseModel(cfg)
    model.load_state_dict(torch.load(cfg.model.checkpoint, map_location=cfg.device)['model_state_dict'])
    return model
```

The model is called as `rel_pose_pred = model(sim_rgb, rgb)`:
- `sim_rgb`: render of the mesh at the initial pose, `[B,3,224,224]` in [0, 1];
- `rgb`: CycleGAN image of the anchor (target) view, `[B,3,224,224]` in [0, 1];
- returns `T_rel` `[B,4,4]`, applied as `T_pred = T_init @ T_rel`.

## Data Generation

`generation/` contains the scripts used to build the benchmark from the AeroPath airway labels, at the state used for the release; `generation/run_aeropath.sh` runs the whole pipeline per case with the settings in `configs/generation.yaml`:

1. `aeropath_mesh.py`: airway mesh from the label (0.5 mm resampling, marching cubes, Taubin smoothing, frame conversion above).
2. `kimimaro_centerlines.py`, `save_centerlines.py`: skeleton → centerline poses with segment ids, parent ids and generations (`centerlines.txt`, `leaf_ids.npy`, `distance_matrix.npy`).
3. `mesh2sdf.py`: SDF grid (`sdf/`) used to reject poses outside the airway.
4. `training_data_prepare.py`: renders RGB, depth and pose at every centerline node except those on terminal branches, plus 5 perturbed samples per node (up to 10 mm and 25°), rejecting poses outside the airway or too close to the wall and spreading samples across difficulty by quantiles.
5. `depth_filter.py`: removes bad renders (`depth_filter.json`).
6. `third_party/sp_cyclegan/infer.py`: CycleGAN translation of the kept renders (`fake_images/`).
7. `generate_pose_json.py`, `generate_pairs.py`: `pose_dict.json` and the evaluation pairs (one per node, ancestor-hop σ = 1, covisibility ≥ 0.2, 3 levels).
8. `export_release.py`: the dataset layout above.

Extra dependencies: `nibabel`, `SimpleITK`, `scikit-image`, `pyvista`, `kimimaro`, `pandas`, `mesh-to-sdf`, `natsort`, `matplotlib`, and optionally `open3d` (only for the `centerlines.ply` visualisation).

Notes:
- Pose sampling and pair selection are random and unseeded, so a rerun produces a different benchmark. Use the released files when comparing with the paper.
- The image translation uses the structure-preserving CycleGAN of Wang et al., *Structure-preserving Image Translation for Depth Estimation in Colonoscopy*, MICCAI 2024 (built on [CUT](https://github.com/taesungp/contrastive-unpaired-translation); vendored in `third_party/sp_cyclegan` with its license). The release used a ResNet-9-block generator trained on unpaired renders and clinical bronchoscopy frames (checkpoint `rgb_struct_cyclegan_res9`, epoch 100), which is not included in this repository.

## Configuration

The evaluation configuration is managed via Hydra. Key parameters in `configs/config.yaml`:

- `device`: CUDA device to use
- `data_root`: Path to one sequence, or `'all'` for every `bronch_real_aero*` folder under `dataset_root`
- `output`: Output directory for results
- `cam.*`: Renderer camera (see [Conventions](#conventions) for the intrinsics of the rendered images)
- `model.input_dim`: Model input size (default 224 × 224)
- `eval.*`: Levels to evaluate and the slice of pairs per level (`start_idx`, `end_idx`, `subsample`)

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{shu2026bronchopt,
  title={BronchOpt: vision-based pose optimization with fine-tuned foundation models for accurate bronchoscopy navigation},
  author={Shu, Hongchao and Soberanis-Mukul, Roger D and Xu, Jiru and Ding, Hao and Ringel, Morgan and Shen, Mali and Sayed, Saif Iftekar and Rafii-Tari, Hedyeh and Unberath, Mathias},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--9},
  year={2026},
  publisher={Springer}
}
```

## License

See `LICENSE` file for details. `third_party/sp_cyclegan` is distributed under its own license (`third_party/sp_cyclegan/LICENSE`).
