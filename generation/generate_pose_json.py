import os
import json

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

def get_pose_from_file(file_path, key_prefix):
    assert file_path.endswith(".txt")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            data = [ float(d) for d in line.split(" ") ]
            key = f"{key_prefix}{int(data[0]):05}" 

            if len(data) > 8:
                yield key, data[1:8], int(data[8]), int(data[9]), int(data[10])
            else:
                yield key, data[1:8], -1, -1, -1

def get_all_poses(root_path):
    poses = {}
    file_p = os.path.join(root_path, "poses.txt")
    for key, pose, branch, parent, gen in get_pose_from_file(file_p, "0_"):
        poses[key] = {
            "pose": pose,
            "branch": branch,
            "parent": parent,
            "gen": gen,
        }

    for dir in tqdm(os.listdir(root_path), desc="Generate pose dictionary "):
        if not os.path.isdir(os.path.join(root_path, dir)):
            continue

        file_p = os.path.join(root_path, dir, "poses.txt")
        prefix = f"1_{int(dir):05}_"
        anchor_key = f"0_{int(dir):05}"
        branch = poses[anchor_key]["branch"]
        parent = poses[anchor_key]["parent"]
        gen = poses[anchor_key]["gen"]

        for key, pose, _, _, _ in get_pose_from_file(file_p, prefix):
            poses[key] = {
                "pose": pose,
                "branch": branch,
                "parent": parent,
                "gen": gen,
            }

    return poses

def generate_pose_dict(root_path, all_keys):
    all_poses = get_all_poses(root_path)
    pose_dict = { k: all_poses[k] for k in all_keys }
    return pose_dict

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(cfg: DictConfig):
    # load json
    json_file = os.path.join(cfg.data_root, "train", "depth_filter.json")
    assert os.path.exists(json_file)

    with open(json_file, "r") as f:
        filter_dict = json.load(f)
        all_keys = [ key for key, i in filter_dict.items() if i ]
        pose_dict = generate_pose_dict(os.path.join(cfg.data_root, "train", "poses"), all_keys)

        with open(os.path.join(cfg.data_root, "train", "pose_dict.json"), "w") as js:
            json.dump(pose_dict, js)

if __name__ == "__main__":
    main()
