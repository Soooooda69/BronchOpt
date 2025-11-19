## AeroPath syn data
# videos=('bronch_real_aero1' 'bronch_real_aero2' 'bronch_real_aero3' 'bronch_real_aero4' 'bronch_real_aero5' 'bronch_real_aero6' 'bronch_real_aero7' 'bronch_real_aero8' 'bronch_real_aero9' 'bronch_real_aero10' 'bronch_real_aero11' 'bronch_real_aero12' 'bronch_real_aero13' 'bronch_real_aero14' 'bronch_real_aero15')
videos=('bronch_real_aero1')
for video in "${videos[@]}"; do
    python eval/eval_pose.py device='cuda:0' \
        data_root="./${video}" \
        eval.start_idx=0 \
        eval.end_idx=-1 \
        eval.subsample=10
done