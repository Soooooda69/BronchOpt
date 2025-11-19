# BronchOpt: Vision-Based Pose Optimization with Fine-Tuned Foundation Models for Accurate Bronchoscopy Navigation

Synthetic benchmark dataset for bronchoscopy navigation evaluation. This repository provides the evaluation framework mentioned in the paper for vision-based pose optimization methods in bronchoscopy navigation, addressing the lack of paired CT-endoscopy data for benchmarking.



## Overview

BronchOpt provides a synthetic benchmark dataset for evaluating bronchoscopy navigation algorithms. The dataset includes:

- **Synthetic bronchoscopy sequences**: Multiple sequences (`bronch_real_aero1` through `bronch_real_aero15`) with paired RGB images, poses, and 3D mesh data
- **Evaluation framework**: Tools for evaluating pose estimation methods using multiple metrics
- **Depth rendering**: GPU-accelerated RGB and depth rendering.

## Qualitative Results

Below is an example video showing qualitative results of BronchOpt's pose optimization on the BronchOpt benchmark. The visualization shows the input synthetic bronchoscopy image, the aligned (predicted) pose compared to the ground truth, and the rendered 3D mesh overlay.

https://user-images.githubusercontent.com/18592130/319612447-973af51c-42d1-4aad-b543-8e0ec82ebd91.mp4




## Our Quantitative Results

Below we report the quantitative comparison between OffsetNet and the proposed BronchOpt on the BronchOpt synthetic benchmark across three difficulty levels. The evaluation metrics include Disparity Similarity (DS~$\uparrow$), Normal Correlation (NC~$\uparrow$), Scale-Invariant Depth Error (SI~$\downarrow$), translation error (mm,~$\downarrow$), rotation error (rad,~$\downarrow$), and success rate (\%). 

The initial camera poses (Init) from the robotic system show increasing difficulty across levels, with mean translation/rotation distances of **$1.94 \pm 2.03$ mm / $0.14 \pm 0.11$ rad** for *Easy*, **$4.29 \pm 2.68$ mm / $0.34 \pm 0.15$ rad** for *Medium*, and **$6.96 \pm 3.09$ mm / $0.49 \pm 0.25$ rad** for *Hard* cases. The reported metrics are computed only over successful cases.

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

[BronchOpt Dataset](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/hshu4_jh_edu/ErTy5BR_H3FEiu43q0H0pGUBx5mGTeQub8Mu9yxpAil6rg?e=k2PZZV)

Simply extract the downloaded files into your desired directory. Each sequence (e.g., `bronch_real_aero1`, `bronch_real_aero2`, ...) will be a subdirectory containing the necessary images, pose data, and mesh files as described below.


Each dataset sequence (e.g., `bronch_real_aero1`) contains:
- `eval_src.json`: Evaluation source data with relative pose relationships
- `pose_dict.json`: Ground truth pose dictionary with 7-DOF poses (translation + quaternion)
- `fake_images/`: Synthetic RGB images for evaluation
- `localMedium.ply`: 3D mesh model of the bronchial anatomy

### Example Dataset Structure

Below is the directory and file structure for a typical sequence (e.g., `bronch_real_aero1`) as provided in [`bronch_real_aero1.zip`]:

```
bronch_real_aero1/
├── eval_src.json          # Evaluation source file for paired relative poses
├── pose_dict.json         # Ground-truth pose dictionary (translation + quaternion per frame)
├── fake_images/           # Synthetic RGB images for each frame
│   ├── 0
│   ├── 1
│   └── ...
├── mesh.ply        # 3D mesh model of the bronchial anatomy (PLY format)
└── [other optional files]
```

- Each sequence directory is self-contained.
- `eval_src.json` and `pose_dict.json` are JSON files containing pose graph and absolute pose information.
- `fake_images/` includes one RGB PNG image per frame.
- `mesh.ply` provides a surface mesh of the involved bronchial anatomy.

Each additional sequence (e.g., `bronch_real_aero1`, `bronch_real_aero2`, etc.) follows the same folder and file structure.

The dataset provides both synthetic RGB images and depth maps that are rendered using the ground-truth (GT) camera poses and the corresponding CT mesh for each sequence. The renderer employed in this benchmark uses the GT pose information along with the 3D mesh to generate realistic visual and depth data, ensuring accurate and consistent input for evaluation across all sequences.


## Installation

### Requirements

- Python 3.12
- PyTorch (with CUDA support)
- CUDA-capable GPU

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
cd ..
```

If you encounter CUDA version issues during this step, ensure your environment matches the required CUDA toolkit for nvdiffrast, or consult their repository for troubleshooting instructions.

3. Build the CUDA extensions:
```bash
pip install -e .
```

This will compile the `lietorch_backends` CUDA extension required for SE(3) operations.

## Usage

### Evaluation

The evaluation script (`eval_pose.py`) evaluates pose estimation methods on the benchmark dataset.

1. Configure the evaluation in `configs/config.yaml`:
```yaml
device: 'cuda:0'
data_root: './processed/bronch_real_aero1'
output: 'output'
```

2. Example evaluation code:
```bash
python eval_pose.py
```

Example shell script:
```bash
bash eval_pose.sh
```

### Evaluation Metrics

The framework evaluates methods using the following metrics:

1. **Disparity Similarity (DS)**: Cosine similarity between normalized depth maps
2. **Normal Correlation (NC)**: Scale-invariant normal correlation between predicted and ground truth depth maps
3. **Scale-Invariant Depth Loss (SI)**: Scale-invariant depth error (Eigen et al. 2014)
4. **Translation Distance**: Euclidean distance between predicted and ground truth camera positions
5. **Rotation Distance**: Geodesic distance on SO(3) between predicted and ground truth rotations

Results are saved in the output directory with:
- `poses_pred.txt`: Predicted trajectory (format: `index x y z qw qx qy qz`)
- `poses_init.txt`: Initial pose estimates
- `dsim.txt`: Per-frame disparity similarity, normal correlation, and scale-invariant metrics
- `res.txt`: Translation and rotation distance metrics

### Integrating Your Model

To evaluate your own pose estimation model:

1. Modify `eval_pose.py` to load your model:
```python
# In run_rel_inference function, replace the TODO section:
model = YourPoseModel(cfg)
checkpoint = torch.load(cfg.model.checkpoint_path, map_location=cfg.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

2. Implement the relative pose prediction:
```python
# Replace below line in eval_pose.py:
rel_pose_pred = model(sim_rgb, rgb)  # Your model forward pass
```

## Configuration

The evaluation configuration is managed via Hydra. Key parameters in `configs/config.yaml`:

- `device`: CUDA device to use
- `data_root`: Path to dataset sequence or `'all'` for all sequences
- `output`: Output directory for results
- `cam.*`: Camera intrinsics and rendering parameters
- `eval.*`: Evaluation parameters (start_idx, end_idx, subsample)

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{shu2025bronchoptvisionbasedpose,
      title={BronchOpt : Vision-Based Pose Optimization with Fine-Tuned Foundation Models for Accurate Bronchoscopy Navigation}, 
      author={Hongchao Shu and Roger D. Soberanis-Mukul and Jiru Xu and Hao Ding and Morgan Ringel and Mali Shen and Saif Iftekar Sayed and Hedyeh Rafii-Tari and Mathias Unberath},
      year={2025},
      eprint={2511.09443},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.09443}, 
}
```

## License

See `LICENSE` file for details.