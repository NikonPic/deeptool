# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\33_rnn_vae.ipynb (unless otherwise specified).

__all__ = ['mod_batch', 'RNN_AE', 'RNN_VAE', 'RNN_INTROVAE', 'RNN_BIGAN', 'Creator_RNN_AE']

# Cell

import torch
from torch import nn, optim
import torch.nn.functional as F
from ..architecture import Encoder, Decoder, DownUpConv
from ..utils import Tracker

# Cell


def mod_batch(batch, key="img"):
    """
    transform the batch to be compatible with the network by permuting
    """
    if len(batch[key].shape) > 4:
        batch[key] = batch[key][0, :, :, :, :]
        batch[key] = batch[key].permute(1, 0, 2, 3)
    return batch

# Cell


class RNN_AE(nn.Module):

    def __init__(self, device, args):
        """
        The recurrent autoencoder for compressing 3d data.
        It compresses in 2d while (hopefully) maintaining the spatial relation between layers
        """
        super(RNN_AE, self).__init__()
        self.device = device

        # 1. create the convolutional Encoder
        args.dim = 2
        self.conv_part_enc = DownUpConv(args, pic_size=args.pic_size, n_fea_in=len(
            args.perspectives), n_fea_next=args.n_fea_up, depth=1).to(self.device)

        # save important features
        max_fea, min_size = self.conv_part_enc.max_fea, self.conv_part_enc.min_size
        self.n_z, self.max_fea, self.min_size = args.n_z, max_fea, min_size

        self.view_arr = [-1, max_fea * min_size**2]  # as flat vector
        self.view_conv = [-1, max_fea, min_size, min_size]  # as conv block
        self.view_track = [1, len(args.perspectives), -1,
                           args.pic_size, args.pic_size]

        # 2. Apply FC- Encoder Part
        self.fc_part_enc = nn.Sequential(
            nn.Linear(max_fea*min_size*min_size, max_fea*min_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea*min_size, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea, args.n_z),
        ).to(self.device)

        # 3. Transition Layer:
        if args.rnn_active:
            # Apply a RECURRENCE
            self.transition = nn.GRU(args.n_z, args.n_z, 1).to(self.device)
        else:
            # simple Identity
            self.transition = nn.Sequential()

        # 4. Apply FC-Decoder Part
        self.fc_part_dec = nn.Sequential(
            nn.Linear(args.n_z, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea, max_fea*min_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea*min_size, max_fea*min_size*min_size),
        ).to(self.device)

        # 5. create the convolutional Decoder
        self.conv_part_dec = DownUpConv(
            args, pic_size=args.pic_size, n_fea_in=len(
                args.perspectives), n_fea_next=args.n_fea_down, depth=1, move='up').to(self.device)

        # the standard loss
        self.mse_loss = nn.MSELoss(reduction='sum')

        # the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        # reset the dimension
        args.dim = 3

        # Setup the tracker to visualize the progress
        if args.track:
            self.tracker = Tracker(args)

    def watch_progress(self, test_data, iteration):
        """Outsourced to Tracker"""
        self.tracker.track_progress(self, test_data, iteration)

    def rnn_transition(self, x):
        """
        take the matrix of encoded input slices and apply the RNN part
        """
        # reshape
        x = x.reshape([-1, 1, self.n_z])
        # apply GRU layer
        x, _ = self.transition(x)
        # reshape
        x = x.reshape([-1, self.n_z])
        return x

    def encode(self, x):
        x = self.conv_part_enc(x)
        x = x.reshape(self.view_arr)
        x = self.fc_part_enc(x)
        return x

    def decode(self, x):
        # apply transition
        x = self.rnn_transition(x)
        # decode
        x = self.fc_part_dec(x)
        x = x.reshape(self.view_conv)
        x = self.conv_part_dec(x)
        return x

    def prep_input(self, batch):
        self.zero_grad()
        batch = mod_batch(batch)
        img = batch['img'].to(self.device)
        return img

    def ae_forward(self, img):
        # encode:
        x = self.encode(img)
        # decode
        x = self.decode(x)
        # calc loss
        loss = self.mse_loss(img, x)
        return loss, x

    def forward(self, batch, update=True):
        """
        calculate the forward pass
        """
        # prepare
        img = self.prep_input(batch)
        # autoencoder
        loss, x = self.ae_forward(img)

        if update:
            loss.backward()
            self.optimizer.step()
            return x

        else:
            tr_data = {}
            tr_data["loss"] = loss.item()

        return x, tr_data

# Cell


class RNN_VAE(RNN_AE):
    """
    inherit from RNN_AE and add the variational part
    """

    def __init__(self, device, args):
        #super(RNN_AE, self).__init__(device, args)
        RNN_AE.__init__(self, device, args)
        # 2. rewrite FC- Encoder Part
        max_fea, min_size = self.max_fea, self.min_size
        self.fc_part_enc = nn.Sequential(
            nn.Linear(max_fea*min_size*min_size, max_fea*min_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea*min_size, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max_fea, 2 * args.n_z),
        ).to(self.device)
        # get the kl facor
        self.gamma = args.gamma

        # reset the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def vae_sampling(self, x):
        mu, log_sig2 = x.chunk(2, dim=1)
        # get random matrix
        eps = torch.rand_like(
            mu, device=self.device)
        # sample together
        z = mu + torch.exp(torch.mul(0.5, log_sig2)) * eps
        return z, mu, log_sig2

    def kl_loss(self, mu, log_sig2):
        return -0.5 * torch.sum(1 - torch.pow(mu, 2) - torch.exp(log_sig2) + log_sig2)

    def forward(self, batch, update=True):
        # prepare
        img = self.prep_input(batch)
        # encode
        x = self.encode(img)
        # apply the vae sampling
        x, mu, log_sig2 = self.vae_sampling(x)
        # decode
        x = self.decode(x)

        # get loss
        ae_loss = self.mse_loss(img, x)
        vae_loss = self.kl_loss(mu, log_sig2)
        loss = ae_loss + self.gamma * vae_loss

        if update:
            loss.backward()
            self.optimizer.step()
            return x

        else:
            tr_data = {}
            tr_data["loss"] = loss.item()
            tr_data["ae_loss"] = ae_loss.item()
            tr_data["vae_loss"] = vae_loss.item()

        return x, tr_data

# Cell


class RNN_INTROVAE(RNN_VAE):
    """
    inherit from RNN_VAE and add the GAN part
    """

    def __init__(self, device, args):
        # super(RNN_AE, self).__init__(device, args)
        RNN_VAE.__init__(self, device, args)
        # add extra parameters
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.m = args.m
        self.n_pretrain = args.n_pretrain

        # reset the optimizer
        self.optimizer = None

        enc_params = list(self.conv_part_enc.parameters(
        )) + list(self.fc_part_enc.parameters()) + list(self.transition.parameters())
        self.optimizerEnc = optim.Adam(enc_params, lr=args.lr)

        dec_params = list(self.conv_part_dec.parameters(
        )) + list(self.fc_part_dec.parameters()) + list(self.transition.parameters())
        self.optimizerDec = optim.Adam(dec_params, lr=args.lr)

    def forward(self, batch, update=True):

        # prepare
        self.optimizerEnc.zero_grad()
        self.optimizerDec.zero_grad()
        img = self.prep_input(batch)

        # (1st) Pass Original
        # --------------------------------------
        # encode
        z = self.encode(img)
        z, mu, log_sig2 = self.vae_sampling(z)

        # decode
        x_re = self.decode(z)

        # Losses
        ae_loss = self.beta * self.mse_loss(img, x_re)
        kl_loss = self. gamma * self.kl_loss(mu, log_sig2)

        # (2nd) Pass Reconstruct Original (Enc)
        # --------------------------------------
        # encode
        z_re_1 = self.encode(x_re.detach())
        z_re_1, mu_re_1, log_sig2_re_1 = self.vae_sampling(z_re_1)

        # Losses
        kl_loss_re_e = self.kl_loss(mu_re_1, log_sig2_re_1)

        # (3rd) Pass Generate Fake imgs (Enc)
        # --------------------------------------
        # generate fake samples
        z_p = torch.randn_like(z, device=self.device)

        # decode
        x_p = self.decode(z_p)

        # encode (xp stopped!)
        z_p_re_1 = self.encode(x_p.detach())
        z_p_re_1, mu_p_re_1, log_sig2_re_1 = self.vae_sampling(z_p_re_1)

        # Losses
        kl_loss_p_e = self.kl_loss(mu_p_re_1, log_sig2_re_1)

        # -------
        l_adv_e = self.alpha * \
            0.5 * (torch.clamp(self.m - kl_loss_re_e, min=0) +
                   torch.clamp(self.m - kl_loss_p_e, min=0))
        L_e = ae_loss + kl_loss + l_adv_e

        if update:
            L_e.backward(retain_graph=True)
            self.optimizerEnc.step()
        # ------

        # (4th) Pass Reconstruct Original (Dec)
        # --------------------------------------
        # encode (x_re free)
        z_re_2 = self.encode(x_re)
        z_re_2, mu_re_2, log_sig2_re_2 = self.vae_sampling(z_re_2)

        # Losses
        kl_loss_re_d = self.kl_loss(mu_re_2, log_sig2_re_2)

        # (5th) Pass Generate Fake imgs (Dec)
        # --------------------------------------
        # encode (xp free)
        z_p_re_2 = self.encode(x_p)
        z_p_re_2, mu_p_re_2, log_sig2_re_2 = self.vae_sampling(z_p_re_2)

        # Losses
        kl_loss_p_d = self.kl_loss(mu_p_re_1, log_sig2_re_1)

        L_d = self.alpha * 0.5 * (kl_loss_re_d + kl_loss_p_d)

        # ------
        if update:
            L_d.backward()
            self.optimizerDec.step()
            return x_re
        # ------

        else:
            tr_data = {}
            tr_data["L_encoder"] = L_e.item()
            tr_data["L_decoder"] = L_d.item() + ae_loss.item() + kl_loss.item()
            tr_data["ae_loss"] = ae_loss.item()
            tr_data["vae_loss"] = kl_loss.item()
            tr_data["l_adv_e"] = l_adv_e.item()
            tr_data["l_adv_d"] = L_d.item()

        return x_re, tr_data

# Cell


class RNN_BIGAN(RNN_VAE):
    """
    apply the Bidirectional-GAN part, inherit from the normal autoencoder
    """

    def __init__(self, device, args):
        """
        init the networks and the discriminator
        """
        # init the vae architecture
        RNN_VAE.__init__(self, device, args)
        # we ned a dicriminator!
        # switch to 2 dim for the init:
        # -----------
        args.dim = 2
        self.conv_part_dis = DownUpConv(args, pic_size=args.pic_size, n_fea_in=len(
            args.perspectives), n_fea_next=args.n_fea_up, depth=1).to(self.device)
        args.dim = 3
        # -----------

        # take saved params
        max_fea, min_size, n_z = self.max_fea, self.min_size, args.n_z

        # add the fc part(s)
        self.fc_part_dis_x = nn.Sequential(
            # layer 1
            nn.Linear(max_fea*min_size*min_size, max_fea*min_size),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 2
            nn.Linear(max_fea*min_size, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 3
            nn.Linear(max_fea, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 4
            nn.Linear(max_fea, 1),
        ).to(self.device)

        self.fc_part_dis_z = nn.Sequential(
            # layer 1
            nn.Linear(n_z, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 2
            nn.Linear(max_fea, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 3
            nn.Linear(max_fea, 1),
        ).to(self.device)

        self.fc_part_dis_xz = nn.Sequential(
            # layer 1
            nn.Linear(n_z+max_fea*min_size*min_size, max_fea*min_size),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 2
            nn.Linear(max_fea*min_size, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 3
            nn.Linear(max_fea, max_fea),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 4
            nn.Linear(max_fea, 1),
        ).to(self.device)

        # reset the optimizer
        self.optimizer = None

        enc_params = list(self.conv_part_enc.parameters(
        )) + list(self.fc_part_enc.parameters())
        self.optimizerEnc = optim.Adam(enc_params, lr=args.lr)

        dec_params = list(self.conv_part_dec.parameters(
        )) + list(self.fc_part_dec.parameters()) + list(self.transition.parameters())
        self.optimizerDec = optim.Adam(dec_params, lr=args.lr)

        dis_params = list(self.conv_part_dis.parameters()) + list(self.fc_part_dis_x.parameters()) + \
            list(self.fc_part_dis_z.parameters()) + \
            list(self.fc_part_dis_xz.parameters())
        self.optimizerDis = optim.Adam(dis_params, lr=args.lr)

        # parameters
        self.lam = args.lam
        self.bi_ae_scale = args.bi_ae_scale

    def encode_non_d(self, x):
        """non-deterministic encoding from the paper"""
        x = self.encode(x)
        # get mu and sigma
        mu, sig_hat = x.chunk(2, dim=1)
        sig = torch.log(1 + torch.exp(sig_hat))
        # get random matrix
        eps = torch.rand_like(mu, device=self.device)
        # sample together
        z = mu + sig * eps
        return z

    def get_s(self, x, z):
        """apply discriminator and output sx, sz and sxz"""
        # shape inputs
        x = self.conv_part_dis(x)
        x = x.reshape(self.view_arr)
        xz = torch.cat([x, z], dim=1)

        # apply fc decisions to generate out-dis
        s_x = self.fc_part_dis_x(x).view(-1)
        s_z = self.fc_part_dis_z(z).view(-1)
        s_xz = self.fc_part_dis_xz(xz).view(-1)

        return s_x, s_z, s_xz

    def hinge(self, x):
        """the hinge loss: max(0, 1-x)"""
        return F.relu(1-x)

    def decide(self, x, z, y, ed=False):
        """
        generate dis-loss
        ed -> ENCODE-DECODE Learning
        """
        # get decisions from Discriminator
        s_x, s_z, s_xz = self.get_s(x, z)

        # apply y for encoder-decoder
        if ed:
            return y * (s_x + s_z + s_xz)

        # apply hinge losses for discriminator
        hs_x = self.hinge(y * s_x)
        hs_z = self.hinge(y * s_z)
        hs_xz = self.hinge(y * s_xz)

        return hs_x + hs_z + hs_xz

    def ae_part(self, x, update):
        """simple forward pass of autoencoder"""
        ae_loss, _ = self.ae_forward(x)
        ae_loss *= self.bi_ae_scale

        if update:
            ae_loss.backward()
        return ae_loss.mean().item()

    def forward(self, batch, update=True):
        """main function"""
        # zero all gradients
        self.optimizerEnc.zero_grad()
        self.optimizerDec.zero_grad()
        self.optimizerDis.zero_grad()

        # load batch
        x = self.prep_input(batch)

        # (0) Train Autoencoder
        # -------------------------------
        ae_loss = 0
        #ae_loss = self.ae_part(x, update)

        # (1) Train Discriminator
        # -------------------------------
        # load batch
        x = self.prep_input(batch)
        # generate original z
        z = self.encode_non_d(x)

        # fake
        z_p = torch.randn_like(z, device=self.device)
        # decode
        x_p = self.decode(z_p)

        # fill the labels
        b_size = x.size(0)

        # real
        errD_real = self.decide(x.detach(), z.detach(), +1).mean()
        if update:
            errD_real.backward()

        # fake
        errD_fake = self.decide(x_p.detach(), z_p.detach(), -1).mean()

        if update:
            errD_fake.backward()
            self.optimizerDis.step()

        errD = (errD_real + errD_fake).mean().item()

        # (2) Train Encoder / Decoder
        # -------------------------------
        # encode
        z = self.encode_non_d(x)

        # decode
        x_p = self.decode(z_p)

        # real
        errEnc = self.decide(x, z, +1, ed=True).mean()

        if update:
            errEnc.backward()

        # fake
        errDec = self.decide(x_p, z_p, -1, ed=True).mean()

        errEncDec = (errEnc + errDec).mean().item()

        # Update Generator
        if update:
            errDec.backward()
            self.optimizerEnc.step()
            self.optimizerDec.step()
            return x_p

        else:
            # Track all relevant losses
            tr_data = {}
            tr_data["ae_loss"] = ae_loss
            tr_data["errDis"] = errD
            tr_data["errEncDec"] = errEncDec

            tr_data["errD_real"] = errD_real.mean().item()
            tr_data["errD_fake"] = errD_fake.mean().item()

            tr_data["errEnc"] = errEnc.mean().item()
            tr_data["errDec"] = errDec.mean().item()

            # generate the autoencoder output:
            x_r = self.decode(z)

            # Return losses and reconstruction data
            return x_r, tr_data

# Cell


def Creator_RNN_AE(device, args):
    """
    return an instance of the class depending on the mode set in args
    """
    switcher = {
        "ae": RNN_AE,
        "vae": RNN_VAE,
        "introvae": RNN_INTROVAE,
        "bigan": RNN_BIGAN,
    }
    print(args.rnn_type)
    # Get the model_creator
    model_creator = switcher.get(args.rnn_type, lambda: "Invalid Model Type")
    return model_creator(device, args)