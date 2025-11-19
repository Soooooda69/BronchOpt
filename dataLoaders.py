import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

class PoseEvalDataset(Dataset):
    def __init__(self, cfg, level=None):
        self.cfg = cfg
        self.randomize = cfg.data.randomize
        self.rs = transforms.Resize((cfg.model.input_dim.h, cfg.model.input_dim.w))

        # Support multiple data roots
        if cfg.data_root == 'all':
            data_roots = [f'./processed/bronch_real_aero{str(i)}' for i in range(1,15)]
        else:
            data_roots = [cfg.data_root]

        self.f_data_src_list = []
        self.f_pose_list = []
        self.f_fake_rgb_list = []

        for root in data_roots:
            train_root = os.path.join(root, ".")
            self.f_data_src_list.append(os.path.join(train_root, "eval_src.json"))
            self.f_pose_list.append(os.path.join(train_root, "pose_dict.json"))
            # self.f_depth_list.append(os.path.join(train_root, "depths"))
            self.f_fake_rgb_list.append(os.path.join(train_root, "fake_images"))
            # self.f_rgb_list.append(os.path.join(train_root, "images"))

        # Check all paths exist
        for paths in zip(self.f_data_src_list, self.f_pose_list):
            for p in paths:
                assert os.path.exists(p), f"Path does not exist: {p}"

        # Aggregate keys and poses from all data roots
        self.keys = []
        # self.poses = {}
        self.poses: list[dict[str, tuple[torch.Tensor,int,int]]] = []
        self.key_to_dataset_idx = []

        for dataset_idx, (f_data_src, f_pose) in enumerate(zip(self.f_data_src_list,
                                                               self.f_pose_list)):
            # create an empty dict for this dataset
            self.poses.append({})

            # load key relations -------------------------------------------------
            with open(f_data_src) as f:
                rel_keys = json.load(f)["rel"]
                if level is None:
                    rel_keys_level = rel_keys
                else:
                    rel_keys_level = [k for k in rel_keys if k.get('level', 0) == level]
                self.keys.extend(rel_keys_level)
                self.key_to_dataset_idx.extend([dataset_idx] * len(rel_keys_level))

            # load poses ---------------------------------------------------------
            with open(f_pose) as f:
                poses = json.load(f)
                for key, item in poses.items():
                    self.poses[dataset_idx][key] = (
                        torch.tensor(
                            [item["pose"][0],
                             item["pose"][1],
                             item["pose"][2]] + item["pose"][3:]
                        ),                     # 7-vector (xyz [cm] + qwxyz)
                        item["gen"],
                        item["branch"]
                    )

    def _load_data(self, key, dataset_idx):
        f_rgb = self.f_rgb_list[dataset_idx]
        mesh_id = f_rgb.split('/')[-3].split('_')[-1]
        f_fake_rgb = self.f_fake_rgb_list[dataset_idx]

        f = f"{'/'.join([str(int(k)) for k in key.split('_')[1:]])}.npy"
        fake_rgb_img_f = os.path.join(f_fake_rgb, f.replace(".npy", ".png"))

        fake_rgb = cv2.imread(fake_rgb_img_f)
        fake_rgb = cv2.cvtColor(fake_rgb, cv2.COLOR_BGR2RGB)
        fake_rgb = cv2.resize(fake_rgb, (224, 224))
        
        return fake_rgb, mesh_id

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k1 = self.keys[idx]["anchor"]
        k2 = self.keys[idx]["sample"]
        dataset_idx = self.key_to_dataset_idx[idx]

        anchor_fake_rgb, mesh_id = self._load_data(k1, dataset_idx)
        sampled_fake_rgb, mesh_id = self._load_data(k2, dataset_idx)

        anchor_pose  = self.poses[dataset_idx][k1][0]   # only the 7-vector
        sampled_pose = self.poses[dataset_idx][k2][0]
     
        sampled_fake_rgb = torch.from_numpy(sampled_fake_rgb).float().permute(2, 0, 1) / 255.0
        sampled_sim_rgb = torch.from_numpy(sampled_sim_rgb).float().permute(2, 0, 1) / 255.0
        anchor_fake_rgb = torch.from_numpy(anchor_fake_rgb).float().permute(2, 0, 1) / 255.0
        return sampled_fake_rgb, anchor_fake_rgb, sampled_pose, anchor_pose, mesh_id