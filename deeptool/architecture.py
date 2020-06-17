# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_architecture.ipynb (unless otherwise specified).

__all__ = ['instance_std', 'group_std', 'EvoNorm2D', 'weights_init', 'Quantize', 'ResNetBlock', 'ConvBn', 'ConvTpBn',
           'LinearSigmoid', 'DownUpConv', 'Encoder', 'Decoder', 'Discriminator']

# Cell
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Cell
def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    return torch.sqrt(var + eps)


def group_std(x, groups=32, eps=1e-5):
    N, C, H, W = x.size()
    groups = C if groups > C else groups
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

# Cell
class EvoNorm2D(nn.Module):
    def __init__(
        self, input, non_linear=True, version="S0", momentum=0.9, training=True
    ):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
        if self.non_linear:
            self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.register_buffer("running_var", torch.ones(1, self.insize, 1, 1))
        self.reset_parameters()

        # prepare the functionality in advance
        if self.version == "S0":
            self.forward = self.forward_S0_nl if self.non_linear else self.forward_S0_l

        if self.version == "B0":
            self.forward = self.forward_B0

    def reset_parameters(self):
        self.running_var.fill_(1)

    def forward_S0_nl(self, x):
        num = x * torch.sigmoid(self.v * x)
        return num / group_std(x) * self.gamma + self.beta

    def forward_S0_l(self, x):
        return x * self.gamma + self.beta

    def forward_B0_nl(self, x):
        if self.training:
            var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True).reshape(
                1, x.size(1), 1, 1
            )
            with torch.no_grad():
                self.running_var.copy_(
                    self.momentum * self.running_var + (1 - self.momentum) * var
                )
        else:
            var = self.running_var

        if self.non_linear:
            den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x))
            return x / den * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta

# Cell


def weights_init(m):
    """
    Define the weight parameters depending on the type:
    Conv or Batchnorm
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Cell


class Quantize(nn.Module):
    """
    Quantization 'Layer'
    inspired by: https://github.com/deepmind/sonnet
    modified from: https://github.com/rosinality/vq-vae-2-pytorch
    """

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        """
        Setup the embedding Matrix (dim x n_embed)
        """
        # This is equal to the feature number at the corresponding layer
        self.dim = dim
        # This is the discretization level for each pixel
        self.n_embed = n_embed
        # Learning parameters
        self.decay = decay
        self.eps = eps
        # Init matrix of available embeddings
        embed = torch.randn(dim, n_embed)
        # Register to avoid gradients
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x, update=True):
        """
        Apply the Quantization on the input:
        -> Update Embeddings
        -> Remain the Gradient flow
        Return:
            Quantized input
            Difference between true and
        """
        # Reshape input
        flatten = x.reshape(-1, self.dim)
        # Calculate L2-distance between each pixel / voxel and embedding
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # Select the closest pairs
        _, embed_ind = (-dist).max(1)
        # Construct matrix of matching pairs
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # Select the correct dimensions here!
        embed_ind = embed_ind.view(*x.shape[:-1])
        # Apply quantization
        quantize = self.embed_code(embed_ind)

        # Only for training -> Update Embeddings with moving average
        if update:
            # N_i = N_(i-1) * gamma + (1-gamma) * n_i
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            # Sum(E(x))
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            # m_i = m_(i-1) *gamma + (1-gamma) * Sum(E(x))
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            # N_i
            n = self.cluster_size.sum()
            # norm N_i
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            # e_i = m_i / N_i
            self.embed.data.copy_(embed_normalized)

        # Loss between Original to Quantization
        diff = (quantize.detach() - x).pow(2).mean()
        # Get Output, while enabling to copy gradients
        quantize = x + (quantize - x).detach()
        # Return results
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """
        Perform the quantization by selecting all embedings from the ids
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))

# Cell


class ResNetBlock(nn.Module):
    """
    An individually designalble ResNet Block for 3 Dimensional Convoluions
    based on: https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        n_chan,
        convsize=3,
        activation=nn.ReLU(inplace=True),
        init_w=weights_init,
        dim=3,
        evo_on=False,
    ):
        """setup the general architecture"""
        super(ResNetBlock, self).__init__()

        if dim == 3:
            conv = nn.Conv3d
            batchnorm = nn.BatchNorm3d
            evo = nn.BatchNorm3d

        else:
            conv = nn.Conv2d
            batchnorm = nn.BatchNorm2d
            evo = EvoNorm2D if evo_on else nn.BatchNorm2d

        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

        self.conv1 = conv(n_chan, n_chan, convsize, stride=1, padding=1, bias=False)
        self.conv1.apply(init_w)

        self.conv2 = conv(n_chan, n_chan, convsize, stride=1, padding=1, bias=False)
        self.conv2.apply(init_w)

        if evo_on:
            self.bn1 = evo(n_chan)
            self.bn2 = evo(n_chan)
        else:
            self.bn1 = batchnorm(n_chan)
            self.bn2 = batchnorm(n_chan)

    def forward(self, x):
        """Calculate the forward pass"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation(out)

        return out

# Cell


class ConvBn(nn.Module):
    """
    An individually designalble Block for 3 Dimensional Convoluions with Batchnorm and Dropout
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        convsize=3,
        stride=2,
        activation=nn.LeakyReLU(0.2, inplace=True),
        init_w=weights_init,
        padding=1,
        dim=3,
        p_drop=0,
        evo_on=False
    ):
        """setup the general architecture"""
        super(ConvBn, self).__init__()

        if dim == 3:
            conv = nn.Conv3d
            batchnorm = nn.BatchNorm3d
            dropout = nn.Dropout3d
        else:
            conv = nn.Conv2d
            batchnorm = EvoNorm2D if evo_on else nn.BatchNorm2d
            dropout = nn.Dropout2d

            # Check convsize and stride
            if type(convsize) == tuple and len(convsize) > 2:
                convsize = convsize[1:]

            if type(stride) == tuple and len(stride) > 2:
                stride = stride[1:]

        self.main_part = nn.Sequential(
            conv(
                in_chan, out_chan, convsize, stride=stride, padding=padding, bias=False
            ),
            batchnorm(out_chan),
            activation,
            dropout(p=p_drop),
        )
        self.main_part.apply(init_w)

    def forward(self, x):
        """Calculate the forward pass"""
        return self.main_part(x)

# Cell


class ConvTpBn(nn.Module):
    """
    An individually designalble Block for 3 Dimensional Transposed Convoluions with Batchnorm
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        convsize=3,
        stride=2,
        activation=nn.ReLU(inplace=True),
        init_w=weights_init,
        padding=1,
        dim=3,
        evo_on=False
    ):
        """setup the general architecture"""
        super(ConvTpBn, self).__init__()
        if dim == 3:
            convtranspose = nn.ConvTranspose3d
            batchnorm = nn.BatchNorm3d
        else:
            convtranspose = nn.ConvTranspose2d
            batchnorm = EvoNorm2D if evo_on else nn.BatchNorm2d

            # Check convsize and stride
            if type(convsize) == tuple and len(convsize) > 2:
                convsize = convsize[1:]
            if type(stride) == tuple and len(stride) > 2:
                stride = stride[1:]

        self.main_part = nn.Sequential(
            convtranspose(
                in_chan, out_chan, convsize, stride=stride, padding=padding, bias=False
            ),
            batchnorm(out_chan),
            activation,
        )
        self.main_part.apply(init_w)

    def forward(self, x):
        """Calculate the forward pass"""
        return self.main_part(x)

# Cell


class LinearSigmoid(nn.Module):
    """
    Helper class to provide a simple ending with linear layer and Sigmoid
    """

    def __init__(self, hidden_dim, y_dim, bias=False):
        """setup network"""
        super(LinearSigmoid, self).__init__()
        # store inputs
        self.hidden_dim = hidden_dim
        self.y_dim = y_dim
        self.bias = bias
        # make network
        self.main_part = nn.Sequential(
            nn.Linear(self.hidden_dim, self.y_dim, bias=self.bias), nn.Sigmoid()
        )

    def forward(self, x):
        """reformat then apply fc part"""
        x = x.view((-1, self.hidden_dim))
        x = self.main_part(x)
        return x

# Cell
class DownUpConv(nn.Module):
    """
    A helper Type, which contains the generic conv network
    for 3d up- or downsclaing depending on "move"
    """

    def __init__(
        self, args, n_fea_in, n_fea_next, pic_size, depth, move="down", p_drop=0
    ):
        """Setup the conv-network"""
        super(DownUpConv, self).__init__()
        # Convolutions in 2d / 3d
        self.dim = args.dim
        self.evo_on = args.evo_on
        # Input dim
        self.pic_size = pic_size
        self.depth = depth
        self.n_fea_in = n_fea_in
        # Number of feature channels
        self.n_fea_next = n_fea_next
        # Output dim
        self.min_size = args.min_size  # when to stop reducing
        # Scaling
        self.scale2d = args.scale2d
        self.scale3d = args.scale3d
        self.n_res2d = args.n_res2d
        self.n_res3d = args.n_res3d
        # Dropout
        self.p_drop = p_drop
        # Add the relevant quantization layers if required
        self.vq_layers = args.vq_layers if args.model_type == "vqvae" else []
        # Direction
        self.move = move  # Define whether to scale up or down
        self.main, output_tuple = self.generic_conv_init()
        self.max_fea, self.max_fea_next, self.pic_out, self.final_depth = output_tuple

    def add_layers(
        self, conv_layers, fea_in, fea_out, n_res, pic_size, convsize=4, stride=2
    ):
        """
        Add layers according to number of residual blocks and down/upscale mode
        """
        # Downsample
        # ---------------------------------------------------
        if self.move == "down":
            # 1. Downsample
            conv_layers.extend(
                [
                    ConvBn(
                        fea_in,
                        fea_out,
                        dim=self.dim,
                        convsize=convsize,
                        stride=stride,
                        p_drop=self.p_drop,
                        evo_on=self.evo_on
                    ),
                ]
            )
            # 2. Add the residual blocks:
            for _ in range(n_res):
                conv_layers.extend(
                    [
                        ResNetBlock(
                            fea_out, convsize=3, dim=self.dim, evo_on=self.evo_on,
                        ),
                    ]
                )

        # Upsample
        # ---------------------------------------------------
        else:
            # Attention! This layer can be a quantization layer!
            special_pic = pic_size / 2
            if special_pic > self.min_size and special_pic in self.vq_layers:
                fea_out *= 2  # so it has a doubled feature size

            # 1. Add the residual blocks:
            for _ in range(n_res):
                conv_layers[:0] = [
                    ResNetBlock(fea_in, convsize=3, dim=self.dim, evo_on=self.evo_on),
                ]

            # 2. Upsample
            conv_layers[:0] = [
                ConvTpBn(
                    fea_out, fea_in, dim=self.dim, convsize=convsize, stride=stride, evo_on=self.evo_on,
                ),
            ]

    def generic_conv_init(self):
        """
        Initialise the convolution layers generically:
        Down:
        Idea: -> Half picsize until all 3 dims are equal
              -> Then reduce all dims until min_size.
        Up:
        Idea: -> Double all dims until z-limit is reached
              -> Then double picsize until output-dim is reached.
        """

        # Init the conv_layer list
        conv_layers = []
        # Current z-dim of the picture
        cur_pic_dim = self.pic_size
        # Current anz of features at input size
        cur_fea_in = self.n_fea_in
        # Current anz of features at output size
        cur_fea_out = self.n_fea_next
        # Current depth of the picture
        cur_depth = self.depth
        # Summarize current values
        output_tuple = (cur_fea_in, cur_fea_out, cur_pic_dim, cur_depth)

        # Until the limiting z_dim occurs or picsize too small
        # ---------------------------------------------------------------
        while cur_pic_dim > self.depth and cur_pic_dim > self.min_size:
            # Add layers
            self.add_layers(
                conv_layers,
                cur_fea_in,
                cur_fea_out,
                self.n_res2d,
                cur_pic_dim,
                convsize=(3, 4, 4),
                stride=(1, 2, 2),
            )
            # Update input size
            cur_fea_in = cur_fea_out
            # Features are doupled
            cur_fea_out *= self.scale2d
            cur_pic_dim /= 2  # dimension is halved

            # Store current values
            output_tuple = (cur_fea_in, cur_fea_out, cur_pic_dim, cur_depth)

            # CASE: Layer is a quantization layer!!
            if cur_pic_dim in self.vq_layers:
                # Return current Network and relevant parameters
                return nn.Sequential(*conv_layers), output_tuple

        # Limit reached, now continue until min-picsize reached
        # ---------------------------------------------------------------
        while cur_pic_dim > self.min_size:
            # Add layers
            self.add_layers(
                conv_layers,
                cur_fea_in,
                cur_fea_out,
                self.n_res3d,
                cur_pic_dim,
                convsize=4,
                stride=2,
            )
            # Update input size
            cur_fea_in = cur_fea_out
            # Features are doupled
            cur_fea_out *= self.scale3d
            cur_pic_dim /= 2  # Dimension is halved
            cur_depth /= 2  # Depth is also halfed

            # Store current values
            output_tuple = (cur_fea_in, cur_fea_out, cur_pic_dim, cur_depth)

            # CASE: Layer is a quantization layer!!
            if cur_pic_dim in self.vq_layers:
                # Return current Network and relevant parameters
                return nn.Sequential(*conv_layers), output_tuple

        # Finally return the network at minimum size
        return nn.Sequential(*conv_layers), output_tuple

    def forward(self, x):
        """Calculate the forward pass"""
        return self.main(x)

# Cell


class Encoder(nn.Module):
    """Encoder with 3dimensional conv setup"""

    def __init__(self, args, init_w=weights_init, vae_mode=True):
        """
        Setup the Architecture:
        Args:
            ngpu = Number of GPUs available
            init_w = Function for initialisation of weights
            n_chan = Number of input channels: batch x n_chan x depth x size x size
            n_d_fea = Number of feature channels within network
            n_z = Latent space dimension
        """
        super(Encoder, self).__init__()
        # vae / ae definitions
        self.n_z = args.n_z * 2 if vae_mode else args.n_z
        self.forward = self.forward_vae if vae_mode else self.forward_ae

        # Convolutional network
        self.conv_part = DownUpConv(
            args,
            n_fea_next=args.n_fea_down,
            move="down",
            pic_size=args.pic_size,
            depth=args.crop_size,
            n_fea_in=len(args.perspectives),
        )
        self.max_fea = self.conv_part.max_fea
        self.hidden_dim = self.max_fea * args.min_size ** (args.dim)

        # Finish with fully connected layers
        self.fc_part = nn.Sequential(
            # State size batch x (cur_fea*4*4*4)
            nn.Linear(self.hidden_dim, self.n_z, bias=False),
            nn.ReLU(inplace=True),
            # Output size batch x n_z
        )
        # Initialise (conv part is already)
        self.fc_part.apply(init_w)

    def forward_vae(self, x):
        """calculate output, return mu and sigma"""
        # Apply convolutions
        x = self.conv_part(x)
        # Resize
        x = x.view((-1, self.hidden_dim))
        # Apply fully connected part
        x = self.fc_part(x)
        # Separate mu and sigma
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def forward_ae(self, x):
        """calculate output, return mu and sigma"""
        # Apply convolutions
        x = self.conv_part(x)
        # Resize
        x = x.view((-1, self.hidden_dim))
        # Apply fully connected part
        x = self.fc_part(x)
        return x

# Cell


class Decoder(nn.Module):
    """Decoder class (also a Generator)"""

    def __init__(self, args, init_w=weights_init):
        """
        Setup the Architecture:
        Args:
            init_w = Function for initialisation of weights
            n_chan = Number of input channels: batch x n_chan x depth x size x size
            n_d_fea = Number of feature channels within network
            n_z = Latent space dimension
        """
        super(Decoder, self).__init__()

        # Convolutional network
        self.conv_part = DownUpConv(
            args,
            n_fea_next=args.n_fea_up,
            move="up",
            pic_size=args.pic_size,
            depth=args.crop_size,
            n_fea_in=len(args.perspectives),
        )
        self.max_fea = self.conv_part.max_fea
        self.hidden_dim = self.max_fea * args.min_size ** (args.dim)
        self.view_arr = [-1, self.max_fea]
        self.view_arr.extend([args.min_size for _ in range(args.dim)])

        self.fc_part = nn.Sequential(
            # Input is batch x n_z
            nn.Linear(args.n_z, self.hidden_dim, bias=False),
            # nn.Tanh(),
        )
        # Initialise (conv part is already)
        self.fc_part.apply(init_w)

    def forward(self, x):
        """calculate output"""
        # Apply fully connected part
        x = self.fc_part(x)
        # Resize
        x = x.view(self.view_arr)
        # Apply convolutions
        x = self.conv_part(x)
        return x

# Cell


class Discriminator(nn.Module):
    """
    Discriminator class, only for true/fake differences
    Classifier for Determinig between several classes
    """

    def __init__(self, args, diag_dim=1, init_w=weights_init, wgan=False):
        """Setup the Architecture:"""
        super(Discriminator, self).__init__()

        # Convolutional network
        self.conv_part = DownUpConv(
            args,
            n_fea_next=args.n_fea_down,
            move="down",
            pic_size=args.pic_size,
            depth=args.crop_size,
            n_fea_in=len(args.perspectives),
            p_drop=args.p_drop,
        )
        self.max_fea = self.conv_part.max_fea
        self.hidden_dim = self.max_fea * args.min_size ** (args.dim)

        # Finish with fully connected layers
        self.fc_part = nn.Sequential(
            # State size batch x (cur_fea*4*4*4)
            nn.Linear(self.hidden_dim, diag_dim, bias=False),
        )
        # Initialise (conv part is already)
        self.fc_part.apply(init_w)
        self.forward = self.forward_wgan if wgan else self.forward_dcgan

    def forward_wgan(self, x):
        """calculate output, return prob real / fake"""

        # Apply convolutions
        x = self.conv_part(x)
        # Resize
        x = x.view((-1, self.hidden_dim))
        # Apply fully connected part
        x = self.fc_part(x)
        return x

    def forward_dcgan(self, x):
        """calculate output, return prob real / fake"""

        # Apply Network
        x = self.forward_wgan(x)
        torch.sigmoid_(x)
        return x

# Cell
