#!/usr/bin/env bash
# Regenerate the BronchOpt synthetic benchmark from the public AeroPath dataset.
# Run from the repository root. Expects AeroPath/<id>/<id>_CT_HR_label_airways.nii.gz.
# Settings come from configs/generation.yaml.
#
# The CycleGAN step needs a checkpoint at ./checkpoints/rgb_struct_cyclegan_res9/100_net_G_A.pth
# (trained on clinical bronchoscopy video; see the README).
set -euo pipefail

ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
for id in "${ids[@]}"; do
    case_dir="./processed/bronch_real_aero${id}"

    # 1) airway mesh (metres; writes localMedium.ply and z_translation.txt)
    python generation/aeropath_mesh.py -i "AeroPath/${id}/${id}_CT_HR_label_airways.nii.gz" -o "${case_dir}"
    # 2) centerlines in the same frame (reads z_translation.txt), then poses / branch labels
    python generation/kimimaro_centerlines.py -i "AeroPath/${id}/${id}_CT_HR_label_airways.nii.gz" -o "${case_dir}" --smooth
    python -m generation.save_centerlines --num "${case_dir}"
    # 3) SDF grid used to reject poses outside the airway
    python -m generation.mesh2sdf data_root="${case_dir}"
    # 4) render centerline nodes + perturbed samples (RGB, depth, poses) into <case>/train
    python -m generation.training_data_prepare data_root="${case_dir}"
    # 5) drop bad renders -> <case>/train/depth_filter.json
    python -m generation.depth_filter data_root="${case_dir}/train"
    # 6) CycleGAN translation of the kept renders -> <case>/train/fake_images
    python third_party/sp_cyclegan/infer.py --dataroot "${case_dir}/train" \
        --gpu_ids 0 --netG resnet_9blocks --name rgb_struct_cyclegan_res9 --epoch 100 \
        --num_test 1000000 --model test --model_suffix _A \
        --input_nc 3 --output_nc 3 --no_dropout --checkpoints_dir ./checkpoints
    # 7) pose_dict.json and the evaluation pairs (train_src.json)
    python -m generation.generate_pose_json data_root="${case_dir}"
    python -m generation.generate_pairs data_root="${case_dir}"
    # 8) benchmark layout (eval_src.json, pose_dict.json, fake_images/, localMedium.ply)
    python -m generation.export_release --case "${case_dir}" --out ./release
done
