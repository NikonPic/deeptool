# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\00_dataload.ipynb (unless otherwise specified).

__all__ = ['MAX_PIXEL_VAL', 'MEAN', 'STDDEV', 'INPUT_DIM', 'RandomCrop', 'TriplePrep', 'MiddleCrop', 'Rescale',
           'ToTensor', 'Normalize']

# Cell
import os  # file and path management
import torch  # machine learning
import numpy as np  # matrix multiplication
from torch.utils.data import Dataset, DataLoader  # Structure for Dataloader
from torchvision import transforms  # Predefined transformations
from skimage import transform  # Rescale 2d images
import random
import matplotlib.pyplot as plt

# Cell

# Dataset specific parameters
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

# only takes the middle range...
INPUT_DIM = 224

# Cell

class RandomCrop(object):
    """Randomly crop to only some slices of the image"""

    def __init__(self, args):
        """define the number of slices which will be cropped"""
        assert isinstance(args.crop_size, int)
        assert args.crop_size < 17  # watch out if sizes get too large
        self.crop_size = args.crop_size

    def __call__(self, mr_scan):
        """apply the cropping"""
        # get the depth of the mr scan
        depth = mr_scan.shape[0]
        # choose random range within the depth
        cr_start = np.random.randint(depth - self.crop_size)
        # crop by returning only a part of the true scan
        return mr_scan[cr_start:cr_start+self.crop_size]

# Cell

class TriplePrep(object):
    """
    Randomly crop to only some slices of the image
    Further pad the picture to the 224 size!
    """

    def __init__(self, args):
        """define the number of slices which will be cropped"""
        self.crop_percent = args.crop_percent
        self.pic_size = 224
        self.pad = int((args.pic_size - self.pic_size)/2)
        self.p_h = 0.5

    def h_flip(self, mr_scan, depth):
        """apply horizontal flipping"""
        # random check
        if random.random() < self.p_h:
            # flip
            mr_scan = torch.flip(mr_scan, (2,))
        return mr_scan

    def __call__(self, mr_scan):
        """apply the cropping"""
        mr_scan = torch.FloatTensor(mr_scan)
        # get the depth of the mr scan
        depth = mr_scan.shape[0]
        # get the current crop-percentage:
        cr_layers = int(depth * self.crop_percent)
        # choose random range within the depth
        cr_start = np.random.randint(depth - cr_layers)
        # choose random range within width:
        pad_w_b = np.random.randint(2*self.pad)
        pad_w_e = 2 * self.pad - pad_w_b
        # choose random range within height:
        pad_h_b = np.random.randint(2*self.pad)
        pad_h_e = 2 * self.pad - pad_h_b
        # crop by returning only a part of the true scan and pad picture
        mr_scan = mr_scan[cr_start:cr_start+cr_layers,
                          pad_w_b:-pad_w_e, pad_h_b:-pad_h_e]
        # randonly flip the picture
        mr_scan = self.h_flip(mr_scan, depth)
        return mr_scan

# Cell

class MiddleCrop(object):
    """Crop to only some slices of the image from the middle"""

    def __init__(self, args):
        """define the number of slices which will be cropped"""
        assert isinstance(args.crop_size, int)
        assert args.crop_size < 17  # watch out if sizes get too large
        self.crop_size = args.crop_size

    def __call__(self, mr_scan):
        """apply the cropping"""
        # get the depth of the mr scan
        depth = mr_scan.shape[0]
        # choose the middle range within the scan
        cr = int(depth / 2) - int(self.crop_size / 2)
        # crop by returning only a part of the true scan
        return mr_scan[cr:cr+self.crop_size]

# Cell

class Rescale(object):
    """Rescale a whole MR set to a new size"""

    def __init__(self, args, dtype=np.float32):
        self.output_size = args.pic_size
        self.dim = args.dim
        self.dtype = dtype

        self.shorten = False
        self.model = args.model_type

        if self.model == "triplenet":
            self.shorten = True

    def __call__(self, mr_scan):
        """Apply rescaling"""

        if self.shorten:
            return mr_scan

        # get old dimensions
        depth, h, w = mr_scan.shape

        # get new dimensions
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # if the resolution is the same, avoid this in the future
        if (new_h, new_w) != (h, w):
            # this resets this to a numpy array
            mr_scan = transform.resize(mr_scan, (depth, new_h, new_w))

        # rescale to required input dimension
        if self.dim == 3 and self.shorten == False:
            mr_scan = mr_scan.reshape((1, depth, new_h, new_w))

        return mr_scan

# Cell

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, type=torch.FloatTensor):
        self.type = type

    def __call__(self, mr_scan):
        # apply transformation to Float32 tensor
        return self.type(mr_scan)

# Cell

class Normalize(object):
    """Normalize the array"""

    def __init__(self, usegpu=True):
        """save mean and std"""

        if usegpu:
            self.min = torch.min
            self.max = torch.max
        else:
            self.min = np.min
            self.max = np.max

    def __call__(self, mr_scan):
        """apply normalization on scan"""
        # preselect min and max
        min_mr = self.min(mr_scan)
        max_mr = self.max(mr_scan)

        # standardize
        mr_scan = (mr_scan - min_mr) / (max_mr - min_mr) * MAX_PIXEL_VAL
        # normalize
        mr_scan = (mr_scan - MEAN) / STDDEV

        return mr_scan