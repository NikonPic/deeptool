# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/22_vqvae.ipynb (unless otherwise specified).

__all__ = ['EncQuantDec', 'VQVAE2']

# Cell

# General defines
import torch
from torch import nn, optim
from torch import autograd

# Personal imports
from ..architecture import DownUpConv, Quantize, LinearSigmoid, Discriminator
from ..abs_model import AbsModel

# Cell


class EncQuantDec(nn.Module):
    """
    Helper Class for the generic generated Network with variable number of Quantization Layers
    It Contains:
        Enc   <- List of Encoders
        Dec   <- List of Decoders
        Quant <- List of Quantizations
    If Required:
        Cla   <- List of Classifiers
    """

    def __init__(self, args):
        """
        Idea: Create as many networks, until the minimum dimension is reached
        Add them to a list and construct Quant and Decoders accordingly
        """
        super(EncQuantDec, self).__init__()

        # Initialise empty list of networks -> using Module List to track parameters
        self.Enc = nn.ModuleList()
        self.Dec = nn.ModuleList()
        self.Quant = nn.ModuleList()

        # The Classify / Discriminator part:
        self.classify = False
        if args.vq_classify:
            self.Cla = nn.ModuleList()
            self.classify = True
            self.y_dim = len(args.classes)

        # Gan simply decides between true / false
        if args.vq_gan and False:  # deacivated for now!!
            self.Cla = nn.ModuleList()
            self.classify = True
            self.y_dim = 1

        # Current dimensions
        n_chan = len(args.perspectives)
        arr_pic_size = [args.pic_size]
        arr_fea_in = [n_chan]
        arr_fea_out = [args.n_fea_up]
        arr_depth = [args.crop_size]

        # Create encoder networks until the minimum size is reached
        # Also Create the Classification layers
        count = 0
        while arr_pic_size[-1] > args.min_size:
            # Create next Network
            cur_net = DownUpConv(
                args,
                n_fea_in=arr_fea_in[-1],
                n_fea_next=arr_fea_out[-1],
                pic_size=arr_pic_size[-1],
                depth=arr_depth[-1],
                move="down",
            )

            # Save the current dimensions
            arr_fea_in.append(cur_net.max_fea)
            arr_fea_out.append(cur_net.max_fea_next)
            arr_depth.append(cur_net.final_depth)
            arr_pic_size.append(cur_net.pic_out)
            # Append this net to the list of Encoding networks
            self.Enc.append(cur_net)

            # Add the classification layers as well
            if self.classify:
                # determin the factor of features for classification
                if count == 0:
                    # first layer
                    fea_fac = 1
                    count += 1
                else:
                    # all middler layers
                    fea_fac = 3

                # create classification networks
                cur_net = DownUpConv(
                    args,
                    n_fea_in=arr_fea_in[-2] * fea_fac,
                    n_fea_next=arr_fea_out[-2],
                    pic_size=arr_pic_size[-2],
                    depth=arr_depth[-2],
                    move="down",
                )
                self.Cla.append(cur_net)

                # add the final layer if min size is reached:
                if arr_pic_size[-1] <= args.min_size:
                    # output dim
                    y_dim = self.y_dim
                    # input dim (Class + Quant-single input)
                    hidden_dim = 2 * arr_fea_in[-1] * args.min_size ** (args.dim)
                    # add a simple Linear Net with Sigmoid activation
                    cur_net = LinearSigmoid(hidden_dim, y_dim)
                    # include at end of network
                    self.Cla.append(cur_net)

        # Add the Quantization layers starting from smallest dimension
        for fea_in, pic_size in zip(
            reversed(arr_fea_in[1:]), reversed(arr_pic_size[1:])
        ):
            # Double the feature input if we dont deal with the last layer
            if pic_size != args.min_size:
                fea_in *= 2
            # The last layer differs:
            quant = Quantize(fea_in, args.vq_n_embed, decay=args.vq_gamma)
            self.Quant.append(quant)

        # Reverse the lists for Decoder construction, last elements are not needed
        arr_fea_in = reversed(arr_fea_in[:-1])
        arr_fea_out = reversed(arr_fea_out[:-1])
        arr_pic_size = reversed(arr_pic_size[:-1])
        arr_depth = reversed(arr_depth[:-1])

        # Now add the corresponding decoders to the list
        count = 0
        for fea_in, fea_out, pic_size, depth in zip(
            arr_fea_in, arr_fea_out, arr_pic_size, arr_depth
        ):
            # Create Next Decoder (pay attention to special case in "add_layers" of DownUpConv)
            cur_net = DownUpConv(
                args,
                n_fea_in=fea_in,
                n_fea_next=fea_out,
                pic_size=pic_size,
                depth=depth,
                move="up",
            )
            # Append this to the list of networks
            self.Dec.append(cur_net)

        self.len = len(self.Enc)

        # Finally the required Reshaping dimensions depending on 2D / 3D
        if args.dim == 2:
            # 2D Conv
            self.reshape_q_pre = (0, 2, 3, 1)
            self.reshape_q_pos = (0, 3, 1, 2)
        else:
            # 3D Conv
            self.reshape_q_pre = (0, 2, 3, 4, 1)
            self.reshape_q_pos = (0, 4, 1, 2, 3)

    def classification(self, arr_q, update=True):
        """
        Apply Classification on the quantization steps
        """
        count = 0
        res = 0

        for cla, q in zip(self.Cla, reversed(arr_q)):
            # concatenate the input
            if count > 0:
                q = torch.cat([res, q], 1)

            # calculate output
            res = cla(q)
            count += 1

        return res

    def encode(self, x):
        """
        Use the Encoder Networks to encode all and save all steps in between
        """
        # init empty array
        arr_x = []
        # Go over all Encoders
        for enc in self.Enc:
            x = enc(x)
            arr_x.append(x)

        return arr_x

    def decode_quant(self, arr_x, update=True):
        """
        Decode the input while performing quantizations:
        Return Decoded Picture and the latent difference
        """
        # Inits
        count = 0
        latent_diff = 0
        q_i = 0
        arr_q = []

        # Quantizize and Decode
        for dec, quant, x_i in zip(self.Dec, self.Quant, reversed(arr_x)):
            # Concatenate the Arrays
            if count > 0:
                x_i = torch.cat([q_i, x_i], 1)

            # Reformat for Quantization
            x_i = x_i.permute(self.reshape_q_pre)
            # Perform Quantization and update embeddings
            q_i, diff_i, _ = quant(x_i, update)
            # Reformat to old shape
            q_i = q_i.permute(self.reshape_q_pos)
            # add the quantization to the list of quantizations
            if self.classify:
                arr_q.append(q_i)
            # Decode the quant result
            q_i = dec(q_i)
            # Increase count
            count += 1
            # Save latent losses
            latent_diff += diff_i

        # the output:
        out = q_i

        if self.classify:
            return out, latent_diff, arr_q

        return out, latent_diff

# Cell


class VQVAE2(AbsModel):
    """
    Vector Quantized Variational AutoEncoder
    based on https://arxiv.org/abs/1906.00446
    adapted from https://github.com/rosinality/vq-vae-2-pytorch
    """

    def __init__(self, device, args):
        """Network and parameter definitions"""
        super(VQVAE2, self).__init__(args)

        # Initialise all networks within the Enc-Dec List
        self.device = device
        self.EncQuantDec = EncQuantDec(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.vq_beta = args.vq_beta
        self.optimizer = optim.Adam(self.EncQuantDec.parameters(), lr=args.lr)

        # define forward function
        self.forward = self.forward_normal

        self.vq_class = 0

        # choose the right forward function
        if args.vq_classify:
            # get the classifier specific part
            self.class_criterion = nn.BCELoss()
            self.y_labels = args.classes
            self.y_len = len(self.y_labels)
            self.forward = self.forward_class

        if args.vq_gan:
            # get the gan specific part
            self.gan_loss = nn.BCELoss()
            self.gan_real = 1
            self.gan_fake = 0
            self.forward = self.forward_gan
            self.lam = args.lam

            # get a single classifier:
            # deactivate the model type to not have vq layers
            args.model_type = ""
            self.Cla = Discriminator(args, 1).to(self.device)
            self.optimizer_Cla = optim.Adam(self.Cla.parameters(), lr=args.lr)
            # reactivate model
            args.model_type = "vqvae"

    def set_parameters(self, args):
        """reset the intern parameters to allow pretraining"""
        self.vq_class = args.vq_class

    def calc_gradient_penalty(self, real_data, fake_data):
        """
        Apply the gradient Penalty for Discriminator training
        This is responsible for ensuring the Lipschitz constraint,
        which is required to ensure the Wasserstein distance.
        modified from: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
        """
        # Asssign random factor alpha between 0 and 1
        sh = real_data.shape
        b_size = sh[0]
        alpha = torch.rand(b_size, 1)
        alpha = (
            alpha.expand(b_size, int(real_data.nelement() / b_size))
            .contiguous()
            .view(sh)
        )
        alpha = alpha.to(self.device)

        # interpolating as disc input
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        # evaluate discriminator
        disc_interpolates = self.Cla(interpolates)

        # calculate gradients
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)

        # constrain gradients
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lam

        return gradient_penalty

    def forward_gan(self, data, update=True):
        """
        Determin the training as GAN oriented:
        The Classifier becomes a Dterminator as well
        no sampling.. but reconstructed images are marked as fake
        """
        # Move img-input on GPU
        inp = data["img"].to(self.device)

        # (1) Train Dicriminator
        # 1.1 Train with all-real batch
        self.Cla.zero_grad()
        output = self.Cla(inp).view(-1)
        errD_real = -output.mean()
        # store output
        D_x = output.mean().item()

        # 1.2 Train with all-fake batch
        # build fake:
        arr_x = self.EncQuantDec.encode(inp)
        arr_x = [ele.detach() for ele in arr_x]
        x_r, latent_diff = self.EncQuantDec.decode_quant(arr_x, update=False)
        output = self.Cla(x_r.detach()).view(-1)
        errD_fake = output.mean()
        # store output
        D_G_z1 = output.mean().item()

        gradient_penalty = self.calc_gradient_penalty(inp, x_r.detach())
        errD = errD_fake + errD_real + gradient_penalty

        if update:
            errD.backward()
            self.optimizer_Cla.step()

        # (2) Train Encoder Quant / Discriminator
        self.EncQuantDec.zero_grad()
        # Gan part
        output = self.Cla(x_r).view(-1)
        errG = -output.mean().tanh()
        if update and self.vq_class > 0:
            errG.backward()

        # vqvae part
        arr_x = self.EncQuantDec.encode(inp)
        x_r, latent_diff = self.EncQuantDec.decode_quant(arr_x, update=True)
        recon_loss = self.criterion(x_r, inp)
        latent_loss = latent_diff.mean()

        # final loss
        loss = recon_loss + self.vq_beta * latent_loss

        # Update Generator
        if update:
            loss.backward()
            # composed of Enc/Dec/Quant
            self.optimizer.step()

        else:
            # Track all relevant losses
            tr_data = {}
            tr_data["l_all"] = loss.item()
            tr_data["l_dis"] = errD.item()
            tr_data["l_gen"] = errG.item()
            tr_data["l_recon"] = recon_loss.item()
            tr_data["l_latent"] = latent_loss.item()
            tr_data["D_real(1)"] = D_x
            tr_data["D_fake(0)"] = D_G_z1
            # Return losses and fake data
            return x_r.detach(), tr_data

    def forward_class(self, data, update=True):
        """
        With included classification
        Encode-Quantize-Decode and update
        """
        # Move img-input on GPU
        inp = self.prep(data).to(self.device)
        # Reset Gradients
        self.EncQuantDec.zero_grad()
        # Encode
        arr_x = self.EncQuantDec.encode(inp)
        # Decode and Quantizice - update Embeddings
        x_re, latent_diff, arr_q = self.EncQuantDec.decode_quant(arr_x, update)
        # append x_re to arr_q
        arr_q.append(x_re)
        # Classify
        res = self.EncQuantDec.classification(arr_q, update)
        # Get the true labels
        y = torch.zeros(inp.shape[0], self.y_len)  # init
        for i, cl in enumerate(self.y_labels):
            y[:, i] = data[cl]  # fill
        y = y.to(self.device)  # send to device
        # get the classification loss
        class_loss = self.class_criterion(res, y)

        # Calculate the reconstruction loss
        recon_loss = self.criterion(x_re, inp)
        # Calculate the latent loss
        latent_loss = latent_diff.mean()

        # Get the final loss
        loss = recon_loss + self.vq_beta * latent_loss + self.vq_class * class_loss

        # Backpropagate and Update:
        if update:
            loss.backward()
            self.optimizer.step()
            # Return the output
            return x_re

        # Return a dictionary of data to track
        else:
            tr_data = {}
            tr_data["l_all"] = loss.item()
            tr_data["l_class"] = class_loss.item()
            tr_data["l_recon"] = recon_loss.item()
            tr_data["l_latent"] = latent_loss.item()

            # Return output and losses
            return x_re, tr_data

    def forward_normal(self, data, update=True):
        """Encode-Quantize-Decode and update"""
        # Move img-input on GPU
        inp = self.prep(data).to(self.device)
        # Reset Gradients
        self.EncQuantDec.zero_grad()
        # Encode
        arr_x = self.EncQuantDec.encode(inp)
        # Decode and Quantizice - update Embeddings
        x_re, latent_diff = self.EncQuantDec.decode_quant(arr_x, update)
        # Calculate the reconstruction loss
        recon_loss = self.criterion(x_re, inp)
        # Calculate the latent loss
        latent_loss = latent_diff.mean()
        # Get the final loss
        loss = recon_loss + self.vq_beta * latent_loss

        # Backpropagate and Update:
        if update:
            loss.backward()
            self.optimizer.step()

            # Return the output
            return x_re.detach()

        # Return a dictionary of data to track
        else:
            tr_data = {}
            tr_data["l_all"] = loss.item()
            tr_data["l_recon"] = recon_loss.item()
            tr_data["l_latent"] = latent_loss.item()

            # Return output and losses
            return x_re.detach(), tr_data