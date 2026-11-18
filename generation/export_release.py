"""Assemble the benchmark layout from a generated case folder.

    <case>/localMedium.ply              ->  <out>/<name>/localMedium.ply
    <case>/train/pose_dict.json         ->  <out>/<name>/pose_dict.json
    <case>/train/train_src.json         ->  <out>/<name>/eval_src.json
    <case>/train/fake_images/           ->  <out>/<name>/fake_images/

Usage: python -m generation.export_release --case ./processed/bronch_real_aero1 --out ./release
"""
import argparse
import os
import shutil


def export(case, out, name=None):
    name = name or os.path.basename(os.path.normpath(case))
    dst = os.path.join(out, name)
    os.makedirs(dst, exist_ok=True)
    train = os.path.join(case, 'train')
    shutil.copy2(os.path.join(case, 'localMedium.ply'), os.path.join(dst, 'localMedium.ply'))
    shutil.copy2(os.path.join(train, 'pose_dict.json'), os.path.join(dst, 'pose_dict.json'))
    shutil.copy2(os.path.join(train, 'train_src.json'), os.path.join(dst, 'eval_src.json'))
    if os.path.exists(os.path.join(dst, 'fake_images')):
        shutil.rmtree(os.path.join(dst, 'fake_images'))
    shutil.copytree(os.path.join(train, 'fake_images'), os.path.join(dst, 'fake_images'))
    print(f'Exported {case} -> {dst}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True, help='generated case folder, e.g. ./processed/bronch_real_aero1')
    ap.add_argument('--out', required=True, help='output folder for the benchmark layout')
    ap.add_argument('--name', default=None, help='sequence name (default: basename of --case)')
    args = ap.parse_args()
    export(args.case, args.out, args.name)
