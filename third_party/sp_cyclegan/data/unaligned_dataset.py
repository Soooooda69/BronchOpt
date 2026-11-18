import os.path
from data.base_dataset import BaseDataset, get_transform, dual_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

import numpy as np


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # shuxian: add depth directory
        self.depth_dir_A = os.path.join(opt.dataroot, 'depthA')
        self.A_depth_paths = sorted(make_dataset(self.depth_dir_A, opt.max_dataset_size))

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.opt.phase = opt.phase

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # shuxian: load depths
        # A_depth_path = self.A_depth_paths[index % self.A_size]
        # A_depth_img = np.array(Image.open(A_depth_path)).astype(np.float32) / (2**16 - 1)  # convert to [0, 1] float32
        # A_depth_img = Image.fromarray(A_depth_img) # convert to Image
        # Load depth from .npy file and normalize to [0, 1]
        A_depth_path = self.A_depth_paths[index % self.A_size]
        A_depth_img = np.load(A_depth_path).astype(np.float32)
        A_depth_img = (A_depth_img - A_depth_img.min()) / (A_depth_img.max() - A_depth_img.min() + 1e-8)
        A_depth_img = Image.fromarray(A_depth_img)

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        B = transform(B_img)

        if self.opt.phase == 'train':
            A, A_depth = dual_transform(A_img, A_depth_img, modified_opt)
        else:
            A, A_depth = transform(A_img), None

        # shuxian: add depth to dict
        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        return {'A': A, 'B': B, 'A_depth': A_depth, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
