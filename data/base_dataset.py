"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


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


def get_transfroms(opt):
    transform_list = []
    transform_list.append(transforms.Resize([opt.load_size, opt.load_size], Image.BICUBIC))
    if opt.preprocess == "none":
        transform_list.append(transforms.CenterCrop([opt.crop_size, opt.crop_size]))
    elif opt.preprocess == "light":
        aug_list = [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop([opt.crop_size, opt.crop_size]),
                    transforms.Lambda(__gaussian_blur),
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.75, 1.5), saturation=0., hue=0.)]
        transform_list.extend(aug_list)
    elif opt.preprocess == "common":
        # if random.random() <= 0.1:
        #     transform_list.append(transforms.CenterCrop([opt.crop_size, opt.crop_size]))
        # else:
        aug_list = [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop([opt.crop_size, opt.crop_size]),
                    transforms.Lambda(__gaussian_blur),
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.75, 1.5), saturation=0., hue=0.),
                    transforms.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-8, 8))]
        random.shuffle(aug_list)
        transform_list.extend(aug_list)
    else:
        raise Exception("The preprocessing method is not yet implemented, double check.")

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def __gaussian_blur(img):
    if random.random() > 0.5:
        radius = random.uniform(0.1, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        return img
