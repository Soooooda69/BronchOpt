#!/usr/bin/env bash
# Evaluate on the BronchOpt synthetic benchmark (AeroPath sequences bronch_real_aero1..15).
# Run from the repository root.

# One sequence:
python eval_pose.py device='cuda:0' \
    data_root='./processed/bronch_real_aero1' \
    output='output' \
    eval.start_idx=0 \
    eval.end_idx=-1 \
    eval.subsample=10

# All sequences under dataset_root (writes output/summary.txt over all of them):
# python eval_pose.py device='cuda:0' data_root='all' dataset_root='./processed' output='output'
