import os
import torch
from PIL import Image
from options.test_options import TestOptions
# from options.infer_options import InferOptions
from models import create_model
from data.base_dataset import get_transform
from util.util import tensor2im
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import cv2
from natsort import natsorted
import json
from concurrent.futures import ThreadPoolExecutor

class RealDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        dataset_root=opt.dataroot
        image_root = os.path.join(dataset_root, "images")
        self.image_paths = natsorted([os.path.join(dataset_root, "images", path) for path in os.listdir(image_root)])
        # frames, image_paths = read_video_frames(image_path, -1)
        # frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        # frames = frames / 255.0
        # self.frames = frames
        # self.image_paths = image_paths
        self.transform = get_transform(self.opt, grayscale=False)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx], self.transform)
        return self.image_paths[idx], image

class TrainDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        train_root=opt.dataroot
        self.image_path = os.path.join(train_root, "images")
        json_f = os.path.join(train_root, "depth_filter.json")

        self.transform = get_transform(self.opt, grayscale=False)

        with open(json_f, "r") as f:
            self.all_depth = json.load(f)
        self.keys = natsorted([key for key, value in self.all_depth.items() if value])
        
    def __len__(self):
        return len(self.keys)

    def read_image(self, file_path):
        # image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # BGR
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(file_path).convert('RGB')
        image = self.transform(image)
        # image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return image
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        file_path = f"{self.image_path}/{'/'.join([str(int(k)) for k in key.split('_')[1:]])}.png"
        image = self.read_image(file_path)
        assert image is not None
        return file_path, image

def read_video_frames(video_path, process_length):
    if os.path.isdir(video_path) is True:
        image_paths = natsorted([os.path.join(video_path, i) for i in os.listdir(video_path) if i.endswith('.png') or i.endswith('.npy')])
        if process_length != -1 and process_length < len(image_paths):
            image_paths = image_paths[:process_length]
        
        images = []
        for image_path in image_paths:
            if image_path.endswith('.npy'):
                image = np.load(image_path)
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        frames = np.stack(images, axis=0)
        return frames, image_paths

def load_image(image_path, transform):
    if image_path.endswith('.npy'):
        # Load depth map directly, keep float32 range
        depth = np.load(image_path).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        return depth
    else:
        image = Image.open(image_path).convert('RGB')
        return transform(image)


def save_image(img_tensor, save_path, save_as_npy=False, scale_to_16bit=False):
    # img_numpy = img_tensor.detach().cpu().squeeze().numpy()  # [H, W]
    # if len(img_numpy.shape) == 3:
    #     save_as_npy = False
    #     img_numpy = img_tensor.detach().squeeze().permute(1,2,0).cpu().squeeze().numpy()
    # if save_as_npy:
    #     np.save(save_path.replace(".png", ".npy"), img_numpy)
    # elif scale_to_16bit:
    #     # Scale to 16-bit PNG
    #     img_16bit = (img_numpy * 1000).clip(0, 65535).astype(np.uint16)
    #     Image.fromarray(img_16bit).save(save_path)
    # else:
    #     # Normalize and save as 8-bit PNG
    #     img_norm = (img_numpy - img_numpy.min()) / (np.ptp(img_numpy) + 1e-6)
    #     img_8bit = (img_norm * 255).astype(np.uint8)
    #     Image.fromarray(img_8bit).save(save_path)

    img_numpy = img_tensor.detach().permute(1,2,0).cpu().numpy()  # [H, W]
    # img_numpy = img_tensor.detach().squeeze().permute(1,2,0).cpu().squeeze().numpy()
    # print('1', img_numpy.shape)
    # Normalize and save as 8-bit PNG
    img_norm = (img_numpy - img_numpy.min()) / (np.ptp(img_numpy) + 1e-6)
    img_8bit = (img_norm * 255).astype(np.uint8)
    Image.fromarray(img_8bit).save(save_path)


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    real_data = (opt.dataroot.split('/')[-1] != 'train')
    if real_data:
        testSet = RealDataset(opt)
    else:
        testSet = TrainDataset(opt)

    testLoader = torch.utils.data.DataLoader(
        testSet, batch_size=1, shuffle=False, num_workers=12
    )

    futures = []
    with ThreadPoolExecutor(max_workers=12) as executor:  # ensures shutdown/join
        with torch.no_grad():
            for image_path, image in tqdm(testLoader):
                data = {'A': image, 'A_paths': image_path}
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                fake_B = visuals['fake']  # assume [B, C, H, W]

                for fake_B_i, path in zip(fake_B, image_path):
                    # <dataroot>/images/<rel> -> <dataroot>/fake_images/<rel>
                    rel = os.path.relpath(path, os.path.join(opt.dataroot, 'images'))
                    path = os.path.join(opt.dataroot, 'fake_images', rel)
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    infer_image = fake_B_i.detach().cpu()
                    # side-by-side with input for quick QA
                    # infer_image = torch.cat((image.detach().squeeze().cpu(), infer_image), dim=-1)

                    futures.append(executor.submit(save_image, infer_image, path))

        # (optional) wait and raise if any save failed
        # for f in as_completed(futures):
        #     f.result()

    print(f"Inference complete. Results saved.")

if __name__ == '__main__':
    main()