import json
import os
import numpy as np

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

def iter_all(path: str, ext: str = 'npy', srt_prefix: str = ''):
    """
    Iterate all files with extension |ext| recursively.
    
    Returns: tuple[str, data]
        str: unique id associate with the data, formated as K_N1_N2. K is 0/1,
            0 represents center poses, and 1 represents sampled poses. N1 is 
            the pose group, and N2 is sample number.
        data: path of the data for depth and disparity, python list for poses.
    """
    for f in os.listdir(path):
        subpath = os.path.join(path, f)
        if os.path.isdir(subpath):
            assert f.isdigit()
    
            prefix = '1' if srt_prefix == '' else srt_prefix
            yield from iter_all(subpath, ext, f"{prefix}_{int(f):05}")
    
        elif f.endswith(ext):
            prefix = '0' if srt_prefix == '' else srt_prefix
    
            if ext == 'npy':
                f_name = f.split('.')[0]
                assert f_name.isdigit()
    
                yield (f"{prefix}_{int(f_name):05}", subpath)

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(cfg: DictConfig):
    far_threshold = 0.1
    near_threshold = 0.8
    mean_value_threshold = 1e-3
    far_depth = cfg.cam.zfar
    near_depth = cfg.cam.znear

    depth = { id: data for id, data in iter_all(f"{cfg.data_root}/depths") }
    total = len(depth)
    good_count = 0

    good_key = {}

    for key, item in tqdm(depth.items(), desc="Depth filter "):
        img = np.load(item)
        large_depth_sum = (np.isclose(img, far_depth)).sum()
        small_depth_sum = (np.isclose(img, near_depth)).sum()

        good_key[key] = bool((large_depth_sum) < far_threshold * img.shape[0] * img.shape[1] and 
                             (small_depth_sum) < near_threshold * img.shape[0] * img.shape[1] and 
                             not np.isnan(img).any() and
                             (img.mean() > mean_value_threshold))
        if good_key[key]:
            good_count += 1

    file_name = f"{cfg.data_root}/depth_filter.json"
    with open(file_name, "w") as f:
        json.dump(good_key, f)

    print(good_count / total)
        
if __name__ == "__main__":
    main()
