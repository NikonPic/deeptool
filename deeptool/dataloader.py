# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\00_dataloader.ipynb (unless otherwise specified).

__all__ = ['MAX_PIXEL_VAL', 'MEAN', 'STDDEV', 'INPUT_DIM', 'MRNetDataset', 'RandomCrop', 'TriplePrep', 'MiddleCrop',
           'Rescale', 'ToTensor', 'Normalize', 'load_datasets']

# Cell
import os  # file and path management
import torch  # machine learning
import numpy as np
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

class MRNetDataset(Dataset):
    """
    Magnetic Resonance Imaging Dataset of 1129 knee joints in training data and 119 for validation.
    The Dataset contains information about:
        root_dir: General file directory
        contains two dictionarys with the corresponding infos:
        dirs[{"train", "valid"}][{"abn", "acl", "men"}] -> abnormal, acl-rupture, meniscus
    """

    def __init__(self, args, transform=None, mode="train"):
        """
        Args:
            root_dir(string): Path of the Dataset
            transform: Transformations applied to a sample
            mode: "train" or "valid"
        """
        # lets assume the filepackage is correctly formatted: see order below
        super(MRNetDataset, self).__init__()
        self.root_dir = args.root_dir
        self.naming = args.perspectives
        self.mode = mode
        self.transform = transform

        self.subpaths = os.listdir(self.root_dir)
        assert len(
            self.subpaths) > 8, "Not enough files in directory! - Check directory"

        self.dirs = {}
        self.traindir = os.path.join(self.root_dir, "train")
        self.dirs["train"] = self.read_dir(self.traindir)

        self.validdir = os.path.join(self.root_dir, "valid")
        self.dirs["valid"] = self.read_dir(self.validdir)

        self.labels = {}
        self.weights = {}
        self.labels["train"] = {}

        # start with abnormal labels
        labels = self.read_labels(os.path.join(
            self.root_dir, "train-abnormal.csv"))
        self.labels["train"]["abn"] = labels
        neg_weight = np.mean(self.labels["train"]["abn"])
        self.weights["abn"] = [neg_weight, 1-neg_weight]

        # acl labels
        labels = self.read_labels(
            os.path.join(self.root_dir, "train-acl.csv"))
        self.labels["train"]["acl"] = labels
        temp_labels = [labels[i] for i in range(
            len(labels)) if self.labels["train"]["abn"][i] == 1]
        neg_weight = np.mean(temp_labels)
        self.weights["acl"] = [neg_weight, 1-neg_weight]

        # men labels and weights
        labels = self.read_labels(
            os.path.join(self.root_dir, "train-meniscus.csv"))
        self.labels["train"]["men"] = labels
        temp_labels = [labels[i] for i in range(
            len(labels)) if self.labels["train"]["abn"][i] == 1]
        neg_weight = np.mean(temp_labels)
        self.weights["men"] = [neg_weight, 1-neg_weight]

        # validation labels without weights
        self.labels["valid"] = {}
        self.labels["valid"]["abn"] = self.read_labels(
            os.path.join(self.root_dir, "valid-abnormal.csv"))
        self.labels["valid"]["acl"] = self.read_labels(
            os.path.join(self.root_dir, "valid-acl.csv"))
        self.labels["valid"]["men"] = self.read_labels(
            os.path.join(self.root_dir, "valid-meniscus.csv"))

        # length of the dataset
        self.len = len(self.labels[self.mode]["acl"])

        # define how the output should be formated
        if args.model_type == "diagnosis":
            self.get_img = self.image_stack
        else:
            self.get_img = self.volume_stack

    def image_stack(self, mr_data):
        """
        keep the pictures independent
        """
        return mr_data

    def volume_stack(self, mr_data):
        """
        concatenate the pictures together
        """
        return torch.cat([mr_data[name] for name in self.naming], dim=0)

    def read_labels(self, datadir):
        """
        Read the csv files and store as a single tensor
        """
        labels = []
        for i, line in enumerate(open(datadir).readlines()):
            line = line.strip().split(',')
            label = int(line[1])
            labels.append(label)
        return labels

    def read_dir(self, dir_files):
        """
        Read the directory and determine all files within
        typically include all files from naming
        Args:
            dir_files: filepath for the folders with the categories
        Return:
            filenames: dict of filenames with their corresponding category(axial...)
        """
        filenames = {}
        # go trough ac, cor, sag
        for subfolder in os.listdir(dir_files):
            if subfolder in self.naming:
                # get the local folder name
                subname = os.path.join(dir_files, subfolder)
                subfiles = []

                # collect all filenames
                for subfile in sorted(os.listdir(subname)):
                    # exclude the .DS_STore file
                    if subfile != ".DS_Store":
                        subfiles.append(os.path.join(subname, subfile))

                filenames[subfolder] = subfiles

        return filenames

    def __len__(self):
        """Number of Knees"""
        return self.len

    def __getitem__(self, idx, apply_transform=True):
        """
        Select an MRI dataset with the corresponding information by index.
        Preprocess the data accordingly.
        return:
            data["vol"] -> volumentric data
            data[{"abn","acl","men"}] -> labels
        """
        # load the corresponding files (input data)
        mr_data = {}
        for name in self.naming:
            mr_data[name] = np.load(self.dirs[self.mode][name][idx])

        # Apply transformations on the files here!
        if self.transform and apply_transform:
            for name in self.naming:
                mr_data[name] = self.transform(mr_data[name])

        # collect the csv data for the index (output data)
        data = {}
        data["img"] = self.get_img(mr_data)
        data["abn"] = self.labels[self.mode]["abn"][idx]
        data["acl"] = self.labels[self.mode]["acl"][idx]
        data["men"] = self.labels[self.mode]["men"][idx]

        return data

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

        if self.model == "diagnosis":
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

# Cell

def load_datasets(args):
    """Retruns the training and validation dataset with the corresponding loader"""
    # Select the right cropping
    Crop = RandomCrop if args.rand_crop == True else MiddleCrop

    # special case with triplenet -> we can have more layers -> take percentage
    if args.model_type == "diagnosis":
        args.batch_size = 1
        Crop = TriplePrep

    # Transformations for training
    transform = transforms.Compose([
        Normalize(usegpu=False),
        Crop(args),
        Rescale(args),
        ToTensor(),
    ])

    # Create Datasets
    train_dataset = MRNetDataset(args, mode="train", transform=transform)

    store = args.crop_percent
    args.crop_percent = 0.99

    # Transformations for validation
    transform = transforms.Compose([
        Normalize(usegpu=False),
        Crop(args),
        Rescale(args),
        ToTensor(),
    ])

    args.crop_percent = store

    valid_dataset = MRNetDataset(args, mode="valid", transform=transform)

    # Create the Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_worker)

    return train_dataset, valid_dataset, train_loader, valid_loader