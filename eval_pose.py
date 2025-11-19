import torch
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig
import os 
import numpy as np
from third_party.lietorch import SE3
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from dataLoaders import PoseEvalDataset
import random
from metrics import dispSimilarity, GeodesicLoss, ScaleInvariantDepthLoss
from depth_renderer import DepthRenderer
from metrics import normal_correlation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def homo2quat(pose):
    assert pose.shape == (4, 4)
    quat = R.from_matrix(pose[:3, :3].cpu().numpy()).as_quat()
    return np.hstack((pose[:3, 3].cpu().numpy(), quat))

def save_pred_trajectory(pose_data, output_file_path):
    with open(output_file_path, 'w') as file:
        for (i, pose, label) in pose_data:
            row = homo2quat(pose)
            file.write(str(i) + ' '+ str(row[0]) + ' ' + str(row[1]) + ' '
                       + str(row[2]) + ' '+ str(row[3]) + ' '
                       + str(row[4]) + ' '+ str(row[5]) + ' '
                       + str(row[6]) + '\n')

def save_dsim(data, output_file_path):
    with open(output_file_path, 'w') as file:
        file.write('# index ours_ds ours_nc ours_si\n')
        for (i, ours_ds, ours_nc, ours_si) in data:
            file.write(
                f"{i} {ours_ds:.2f} {ours_nc:.2f} {ours_si:.2f}\n"
            )

        def valid_stats(arr):
            arr = np.array(arr)
            return np.mean(arr), np.std(arr) if len(arr) > 0 else (float('nan'), float('nan'))

        ours_success_mask = [
            (d[1] != -1 and d[3] != -1 and d[5] != -1)
            for d in data
        ]
        ours_success_count = sum(ours_success_mask)
        ours_total = len(data)

        ours_ds_list = [d[1] for d, ok in zip(data, ours_success_mask) if ok]
        ours_nc_list = [d[3] for d, ok in zip(data, ours_success_mask) if ok]
        ours_si_list = [d[5] for d, ok in zip(data, ours_success_mask) if ok]
     
        ours_ds_mean, ours_ds_std = valid_stats(ours_ds_list)
        ours_nc_mean, ours_nc_std = valid_stats(ours_nc_list)
        ours_si_mean, ours_si_std = valid_stats(ours_si_list)

        file.write(
            f"# ours_ds: {ours_ds_mean:.2f} ± {ours_ds_std:.2f} | "
            f"# ours_nc: {ours_nc_mean:.2f} ± {ours_nc_std:.2f} | "
            f"# ours_si: {ours_si_mean:.2f} ± {ours_si_std:.2f} \n "
        )

def save_res(data, output_file_path):
    import numpy as np
    with open(output_file_path, 'w') as file:
        file.write('# index ours_t_dist ours_rot_dist init_t_dist init_rot_dist\n')
        ours_t_dist_list, ours_rot_dist_list = [], []
        init_t_dist_list, init_rot_dist_list = [], []
        for (index, ours_t_dist, ours_rot_dist, init_t_dist, init_rot_dist) in data:
            file.write(f"{index} {ours_t_dist} {ours_rot_dist} {init_t_dist} {init_rot_dist}\n")
            if ours_t_dist != -1:
                ours_t_dist_list.append(ours_t_dist)
            if ours_rot_dist != -1:
                ours_rot_dist_list.append(ours_rot_dist)
            if init_t_dist != -1:
                init_t_dist_list.append(init_t_dist)
            if init_rot_dist != -1:
                init_rot_dist_list.append(init_rot_dist)
        def fmt(mean, std):
            return f"{mean:.4f} ± {std:.4f}" if not np.isnan(mean) else "N/A"
        file.write(
            "# ours_t_dist: {} | ours_rot_dist: {} \n"
            "| init_t_dist: {} | init_rot_dist: {}\n".format(
                fmt(np.mean(ours_t_dist_list), np.std(ours_t_dist_list)),
                fmt(np.mean(ours_rot_dist_list), np.std(ours_rot_dist_list)),
                fmt(np.mean(init_t_dist_list), np.std(init_t_dist_list)),
                fmt(np.mean(init_rot_dist_list), np.std(init_rot_dist_list))
            )
        )


def run_rel_inference(cfg: DictConfig, model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = model.to(cfg.device)
    renderer = DepthRenderer(cfg).to(cfg.device)
    dispSim = dispSimilarity(cfg)
    rot_dist = GeodesicLoss()
    SI = ScaleInvariantDepthLoss()
    rs = transforms.Compose([
        transforms.Resize((cfg.model.input_dim.h, cfg.model.input_dim.w),
                         interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    # 3 difficulty levels
    for i in range(3):
        save_path = f'{cfg.output}/{os.path.basename(cfg.data_root)}/level_{i}'
        os.makedirs(save_path, exist_ok=True)
        all_pre_poses = []
        eval_dsim = []
        eval_init_poses = []
        eval_res = []

        rel_dataset = PoseEvalDataset(cfg, level=i+1)
        eval_loader = torch.utils.data.DataLoader(rel_dataset, batch_size=1, shuffle=False)
        for j, batch in tqdm(enumerate(eval_loader), total=len(rel_dataset)):
            sampled_fake_rgb, anchor_fake_rgb, \
            sampled_pose, anchor_pose, mesh_id = batch
            rgb = anchor_fake_rgb.to(cfg.device)
            [disp, sim_rgb] = renderer(SE3(anchor_pose.to(cfg.device)), color_render=True)
            disp = rs(disp)
            
            sampled_pose = SE3(sampled_pose.to(cfg.device))
            #TODO: your model here
            # rel_pose_pred = model(sim_rgb, rgb)
            
            # update absolute pose prediction
            pose_pred = sampled_pose.matrix() @ rel_pose_pred
            # render depth with predicted absolute pose
            pred_render_depth = renderer(pose_pred)[0].squeeze(1)
            # compute metrics
            ds = dispSim(rs(pred_render_depth), disp)
            nc = normal_correlation(rs(pred_render_depth).unsqueeze(0), disp)
            si, errmap = SI(rs(pred_render_depth), disp, error_map=True)

            eval_dsim.append((i, ds.item(), nc.item(), si.item()))
            timestamp = rel_dataset.keys[j]["anchor"]
            all_pre_poses.append((timestamp, pose_pred, 0))
            eval_init_poses.append((timestamp, sampled_pose.matrix(), 0))
        
        # save results
        save_res(eval_res, f'{save_path}/res.txt')
        save_dsim(eval_dsim, f'{save_path}/dsim.txt')
        save_pred_trajectory(all_pre_poses, f'{save_path}/poses_pred.txt')
        save_pred_trajectory(eval_init_poses, f'{save_path}/poses_init.txt')

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    ##TODO: initialize model and load checkpoint
    # model = PoseOptimizer(cfg)
    # checkpoint = torch.load(cfg.model.pose_checkpoint, weights_only=False, map_location='cuda:0')
    # state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('module.encoder')}
    # model.load_state_dict(state_dict, strict=False)
    
    run_rel_inference(cfg, model)
    return

if "__main__" == __name__:
    main()
