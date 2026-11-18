"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchvision.transforms.functional as tf


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        if "gta2cityscapes" in opt.dataroot:
            osize[0] = opt.load_size // 2
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_shortside' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))

    if 'zoom' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if 'patch' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))

    if 'trim' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))

    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif 'flip' in params:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

def __make_power_2_dual(img, depth, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img, depth

    return img.resize((w, h), method), depth.resize((w,h), method)

def __random_zoom(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img

def __random_zoom_dual(img, depth, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    depth = depth.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img, depth


def __scale_shortside(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)

def __scale_shortside_dual(img, depth, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method), depth.resize((round(ow * scale), round(oh * scale)), method)

def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))

def __trim_dual(img, depth, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend)), depth.crop((xstart, ystart, xend, yend))

def __scale_width(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method)

def __scale_width_dual(img, depth, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method), depth.resize((w,h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __crop_dual(img, depth, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th)), depth.crop((x1, y1, x1+tw, y1+th))
    return img, depth

def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))

def __patch_dual(img, depth, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size)), depth.crop((gridx, gridy, gridx + size, gridy + size))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __flip_dual(img, depth, flip):
    if flip is None:
        flip = np.random.rand() < 0.5
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


# shuxian: add extra functions
def __grayscale_dual(img, depth):
    img = transforms.Grayscale(1)(img)
    return img, depth

def __resize_dual(img, depth, size, method):
    img = tf.resize(img, size, interpolation=method)
    depth = tf.resize(depth, size, interpolation=method)
    return img, depth

def __normalize_dual(img, depth, grayscale):
    if grayscale:
        img = transforms.Normalize((0.5,), (0.5,))(img)
    else:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img, depth

def __random_crop_dual(img, depth, crop_size):
    params = transforms.RandomCrop(crop_size).get_params(img, output_size=(crop_size, crop_size))
    img = tf.crop(img, *params)
    depth = tf.crop(depth, *params)
    return img, depth

def __tensor_dual(img, depth):
    return tf.to_tensor(img), tf.to_tensor(depth)

# shuxian: duplicate transforms between img + depth
def dual_transform(img, depth, opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    if grayscale:
        img, depth = __grayscale_dual(img, depth)
    if 'fixsize' in opt.preprocess:
        img, depth = __resize_dual(img, depth, params["size"], method)
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        if "gta2cityscapes" in opt.dataroot:
            osize[0] = opt.load_size // 2
        img, depth = __resize_dual(img, depth, osize, method)
    elif 'scale_width' in opt.preprocess:
        img, depth = __scale_width_dual(img, depth, opt.load_size, opt.crop_size, method)
    elif 'scale_shortside' in opt.preprocess:
        img, depth = __scale_shortside_dual(img, depth, opt.load_size, opt.crop_size, method)

    if 'zoom' in opt.preprocess:
        if params is None:
            img, depth = __random_zoom_dual(img, depth, opt.load_size, opt.crop_size, method)
        else:
            img, depth = __random_zoom_dual(img, depth, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            img, depth = __random_crop_dual(img, depth, opt.crop_size)
        else:
            img, depth = __crop_dual(img, depth, params['crop_pos'], opt.crop_size)

    if 'patch' in opt.preprocess:
        img, depth = __patch_dual(img, depth, params['patch_index'], opt.crop_size)

    if 'trim' in opt.preprocess:
        img, depth = __trim_dual(img, depth, opt.crop_size)

    # if opt.preprocess == 'none':
    img, depth = __make_power_2_dual(img, depth, base=4, method=method)

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            img, depth = __flip_dual(img, depth, None)
        elif 'flip' in params:
            img, depth = __flip_dual(img, depth, params['flip'])

    if convert:
        img, depth = __tensor_dual(img, depth)
        img, depth = __normalize_dual(img, depth, grayscale)

    return img, depth