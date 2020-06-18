# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_parameters.ipynb (unless otherwise specified).

__all__ = ['get_dataset_args', 'get_dataset_mrnet_args', 'get_dataset_kneexray_args', 'get_model_args',
           'get_architecture_args', 'get_opt_args', 'get_vis_args', 'get_introvae_args', 'get_dcgan_args',
           'get_bigan_args', 'get_vqvae_args', 'get_mrnet_args', 'get_rnn_vae_args', 'get_mocoae_args', 'compat_args',
           'get_all_args']

# Cell

# Load the general modules
import torch.nn as nn
import numpy as np
from torchvision import models
import argparse
import torch
from tqdm import tqdm
import platform

# Cell


def get_dataset_args(parser, args=[]):
    """
    Get all relevant parameters to handle the dataset
    """
    # Dataset:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="MRNet",
        help='Dataset: "MRNet, KneeXray, ..."',
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="Number of GPUs on the PC",
    )
    parser.add_argument(
        "--crop_size", type=int, default=16, help="Depth of the MR-Scan",
    )
    parser.add_argument(
        "--rot_max", type=int, default=30, help="Maximum random rotation angle",
    )
    parser.add_argument(
        "--crop_percent",
        type=float,
        default=0.7,
        help="if we use triplenet take percentage of av. layers",
    )
    parser.add_argument(
        "--rand_crop", type=bool, default=True, help="Decide whether to crop randomly",
    )
    parser.add_argument(
        "--pic_size", type=int, default=256, help="Picture size of the MR-Scan",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        default=0,
        help="Number of workers for loading the dataset",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_dataset_mrnet_args(parser, args=[]):
    """
    Get all relevant parameters to handle the dataset
    -> here: MRNET
    """

    # determine path
    if platform.system() == "Linux":
        path = "/home/biomech/Documents/OsteoData/MRNet-v1.0/"
    else:
        path = "C:/Users/Niko/Documents/data/MRNet-v1.0/MRNet-v1.0"
        # path = "C:/Users/ga46yeg/data/MRNet-v1.0"

    # Dataset MRNet:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--root_dir_mrnet", type=str, default=path, help="Directory of the dataset"
    )
    parser.add_argument(
        "--perspectives",
        type=list,
        default=["axial", "coronal", "sagittal"],
        help="Perspectives of the Mr Scans",
    )
    parser.add_argument(
        "--classes",
        type=list,
        default=["abn", "acl", "men"],
        help="Classify for these classes",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_dataset_kneexray_args(parser, args=[]):
    """
    Get all relevant parameters to handle the dataset
    -> here: KneeXray
    """

    # determine path
    if platform.system() == "Linux":
        path = "/home/biomech/Documents/OsteoData/KneeXrayData/ClsKLData/kneeKL224/"
    else:
        path = "C:/Users/Niko/Documents/data/KneeXrayData/ClsKLData/kneeKL224"

    # Dataset KneeXray:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--root_dir_kneexray", type=str, default=path, help="Directory of the dataset"
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_model_args(parser, args=[]):
    """
    parameters determing the network architecture
    -> model_type: Which model should be applied?
    -> load_model: Use pretrained model?
    -> model_path: pretrained from which path?
    """
    # Model:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--model_type",
        type=str,
        default="diagnosis",
        help='Model: "introvae", "dcgan", "bigan", "vqvae", "diagnosis", "mocoae", "rnnvae"',
    )
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        help="Determine whether to load pretrained model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./data/src/_model",
        help="Path to the model parameters",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_architecture_args(parser, args=[]):
    """
    Parameters determing the network architecture
    """
    # Architecture:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--dim", type=int, default=3, help="Dimension of Network: -2- or -3- "
    )
    parser.add_argument(
        "--n_fea_down",
        type=int,
        default=16,
        help="Number of Conv Features for Downscale Network",
    )
    parser.add_argument(
        "--n_fea_up",
        type=int,
        default=16,
        help="Number of Conv Features for Upscale Network",
    )
    parser.add_argument("--n_z", type=int, default=100, help="Size of latent layer")
    parser.add_argument(
        "--min_size",
        type=int,
        default=4,
        help="Minimum encoding dimension for convolutions",
    )
    parser.add_argument(
        "--scale2d", type=int, default=2, help="Feature factor for 2d downscale"
    )
    parser.add_argument(
        "--scale3d", type=int, default=2, help="Feature factor for 3d downscale"
    )
    parser.add_argument(
        "--n_res2d",
        type=int,
        default=0,
        help="Number of residual layers between striding -2d",
    )
    parser.add_argument(
        "--n_res3d",
        type=int,
        default=0,
        help="Number of residual layers between striding -3d",
    )
    parser.add_argument(
        "--evo_on",
        type=bool,
        default=False,
        help="Define whether to use the advancements of Evolving Normalization-Activation Layers",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_opt_args(parser, args=[]):
    """
    parameters determing the VQVAE parameters
    """
    # Optimizers:
    # ------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--n_epochs", type=int, default=500, help="number of training epochs"
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_vis_args(parser, args=[]):
    """
    parameters for visualization
    """
    # Visualization:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--watch_batch", type=int, default=1130, help="batchnumber for visualization"
    )
    parser.add_argument(
        "--track", type=bool, default=True, help="track the training progress"
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_introvae_args(parser, args=[]):
    """
    parameters determing the IntroVae parameters
    """
    # IntroVAE:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--alpha", type=float, default=1, help="Factor for adversarial leaning"
    )
    parser.add_argument(
        "--beta", type=float, default=0.01, help="Factor for autoencoder leaning"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        help="Factor for variational autoencoder leaning",
    )
    parser.add_argument(
        "--m", type=float, default=500, help="Positive margin for valid learning"
    )
    parser.add_argument(
        "--n_pretrain",
        type=int,
        default=5,
        help="training epochs for pretraining autoencoder",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_dcgan_args(parser, args=[]):
    """
    parameters determing the DCGAN parameters
    """
    # DCGAN:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--lam", type=float, default=10, help="Factor for scaling gradient penalty"
    )
    parser.add_argument(
        "--wgan",
        type=bool,
        default=False,
        help="Determine if WGAN training should be activated",
    )
    parser.add_argument(
        "--p_drop",
        type=float,
        default=0.1,
        help="Dropout probability for the Discriminator network",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_bigan_args(parser, args=[]):
    """
    parameters determing the BiGAN parameters
    """
    # BiGAN:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--bi_extension",
        type=bool,
        default=False,
        help="extending bigan to have sx, and sz (like in BigBiGAN)",
    )
    parser.add_argument(
        "--bi_ae_scale",
        type=float,
        default=0.9,
        help="exponential decaying ae-part of the bigan network",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_vqvae_args(parser, args=[]):
    """
    parameters determing the VQVAE parameters
    """
    # VQVAE2:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--vq_beta",
        type=float,
        default=0.25,
        help="change embedding to keep distance small",
    )
    parser.add_argument(
        "--vq_gamma",
        type=float,
        default=0.99,
        help="discale factor for embedding learning",
    )
    parser.add_argument(
        "--vq_layers",
        type=list,
        default=[8, 16],
        help="Dimensions of layers with quantization",
    )
    parser.add_argument(
        "--vq_n_embed",
        type=int,
        default=512,
        help="Dimensions of layers with quantization",
    )
    parser.add_argument(
        "--vq_classify",
        type=bool,
        default=False,
        help="Activate classifier within vq training",
    )
    parser.add_argument(
        "--vq_class", type=float, default=1.0, help="Factor for classifier learning"
    )
    parser.add_argument(
        "--vq_gan", type=bool, default=False, help="Activate GAN within vq training"
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_mrnet_args(parser, args=[]):
    """
    parameters determing the MRNet parameters
    """
    # MRNet Diagnosis:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--mrnet_backbone",
        type=str,
        default="alexnet",
        help='Choose Backbone: "resnet18" or "alexnet", "vgg" or "squeeze"',
    )
    parser.add_argument(
        "--mrnet_label_smoothing",
        type=float,
        default=0.0,
        help="Smooth labels for potential better generalisation",
    )
    parser.add_argument(
        "--mrnet_batch_update",
        type=int,
        default=32,
        help="Batchcount before updating the network",
    )
    parser.add_argument(
        "--mrnet_hidden_dim",
        type=int,
        default=256,
        help="Size of the hidden layer for the RNN part",
    )
    parser.add_argument(
        "--mrnet_rnn_gap",
        type=bool,
        default=False,
        help="Use an GRU Unit instead of adaptive average gap",
    )
    parser.add_argument(
        "--mrnet_singlestream",
        type=bool,
        default=True,
        help="Use only a single Backbone and Gap",
    )
    parser.add_argument(
        "--mrnet_moco",
        type=bool,
        default=True,
        help = "Contrastive enhanced learning"
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_rnn_vae_args(parser, args=[]):
    """
    parameters determing the Recurrent Varaitional Autoencoder parameters
    """
    # RNN VAE:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--rnn_transition",
        type=str,
        default="cnn",
        help="Define whether the transition layer is active! -> 2D or Pseudo 3D!",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="ae",
        help="Define the Type: ae / vae / intro /bigan",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def get_mocoae_args(parser, args=[]):
    """
    parameters determing the VQVAE parameters
    """
    # MoCoAE:
    # ------------------------------------------------------------------------
    parser.add_argument(
        "--moco_tau",
        type=float,
        default=1, # 0.07 in standard form
        help="Temperature parameter for the logit loss",
    )
    parser.add_argument(
        "--moco_K",
        type=int,
        default=16384,
        help="The Dictionary Size / anz of negative logits",
    )
    parser.add_argument(
        "--moco_m",
        type=list,
        default=[0.999, 0.99],
        help="The Momentum for the key Network update encoder[0] and decoder[1]",
    )
    parser.add_argument(
        "--moco_aemode",
        type=bool,
        default=False,
        help="Define whether the normale autoencoder loss should be considered during training",
    )
    parser.add_argument(
        "--moco_ganmode",
        type=bool,
        default=False,
        help="Define whether the a Generative Adversarial approach should be taken during training",
    )
    # ------------------------------------------------------------------------
    return parser

# Cell


def compat_args(args):
    """
    make arguments compatible with each others by applying all necessary logic
    - so far only crop_size and dim can be uncompatible!
    """
    if args.model_type == 'diagnosis':
        # always use max pic size, all perspectives and limit to MRNet
        args.pic_size = 256
        args.perspectives = ["axial", "coronal", "sagittal"]
        args.dataset_type = "MRNet"
        args.batch_size = 1

    if args.dim == 2:
        # change to 1
        args.crop_size = 1

    if args.dataset_type in [
        "KneeXray",
    ]:
        args.dim = 2
        args.crop_size = 1
        args.perspectives = list(range(3))

    return args

# Cell


def get_all_args(args=[]):
    """
    return all predefined arguments using the combined parameter getters
    """
    # setup
    parser = argparse.ArgumentParser(description="Parameters for training")

    # dataset args
    parser = get_dataset_args(parser, args)
    parser = get_dataset_mrnet_args(parser, args)
    parser = get_dataset_kneexray_args(parser, args)

    # general args
    parser = get_model_args(parser, args)
    parser = get_architecture_args(parser, args)
    parser = get_opt_args(parser, args)
    parser = get_vis_args(parser, args)

    # Model specific args
    parser = get_introvae_args(parser, args)
    parser = get_dcgan_args(parser, args)
    parser = get_bigan_args(parser, args)
    parser = get_vqvae_args(parser, args)
    parser = get_mrnet_args(parser, args)
    parser = get_rnn_vae_args(parser, args)
    parser = get_mocoae_args(parser)

    # transform
    args = parser.parse_args(args=args)

    # correct crop size
    args = compat_args(args)

    return args