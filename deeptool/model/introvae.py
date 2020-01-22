# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\21_introvae.ipynb (unless otherwise specified).

__all__ = ['IntroVAE']

# Cell

import torch
from torch import nn, optim
from ..architecture import Encoder, Decoder
from ..utils import Tracker

# Cell

class IntroVAE(nn.Module):
    """
    Modification of the IntroVAE-Paper for 3-Dimensional tasks in MR-Imaging
    based on: https://arxiv.org/abs/1807.06358
    modified from: https://github.com/woxuankai/IntroVAE-Pytorch
    """

    def __init__(self, device, args):
        """
        Setup the general architecture for the IntroVAE model, composed of:
        >Encoder, Decoder<
        """
        super(IntroVAE, self).__init__()
        # gpu / cpu
        self.device = device

        # Encoder
        self.encoder = Encoder(args).to(self.device)
        # Decoder
        self.decoder = Decoder(args).to(self.device)

        # add further training params here...
        self.alpha = 0  # GAN
        self.beta = args.beta  # AE
        self.gamma = args.gamma  # VAE
        self.m = args.m  # margin for stopping gae learning if too far apart

        # without mean -> squarred error
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

        # optimizers
        self.optimizerEnc = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.optimizerDec = optim.Adam(self.decoder.parameters(), lr=args.lr)

        # Setup the tracker to visualize the progress
        self.tracker = Tracker(args)

    def watch_progress(self, test_data, iteration):
        """
        Outsourced to Tracker
        """
        self.tracker.track_progress(self, test_data, iteration)

    def reparametrisation(self, mu, log_sig2):
        """Apply the reparametrisation trick for VAE."""

        eps = torch.rand_like(
            mu, device=self.device)  # uniform distributed matrix
        # mean + sigma * eps
        z_latent = mu + torch.exp(torch.mul(0.5, log_sig2)) * eps
        return z_latent

    def kl_loss(self, mu, log_sig2):
        """
        KL-Divergence between two univariate gaussian distributions
        special case: compared to uniform distribution: mu2 = 0, sig2= 1
        """
        return -0.5 * torch.sum(1 - torch.pow(mu, 2) - torch.exp(log_sig2) + log_sig2)

    def ae_loss(self, x_hat, x):
        """
        sqrt(sum_i sum_j (x_ij - x_hat_ij)^2)
        pixelwise mean squared error! (sum requires to sum over one picture and the mean!)
        """
        return self.mse_loss(x_hat, x).mean()

    def set_parameters(self, args):
        """
        Control training by setting the parameters:
        alpha, beta, gamma, m
        """
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.m = args.m

    def forward(self, data, update=True):
        """
        Get the different relevant outputs for Intro VAE training
        update=True to allow updating, update=False to keep networs constant
        return x_re (reconstructed) and x_p (sampled)
        """
        # 1. Send data to device
        x = data["img"].to(self.device)

        # 2. Go trough the networks
        # Reset Gradients
        self.optimizerEnc.zero_grad()
        self.optimizerDec.zero_grad()

        # Encode
        z_mu, z_log_sig2 = self.encoder(x)
        # Apply reparametrisation and obtain z_enc
        z_enc = self.reparametrisation(z_mu, z_log_sig2)
        # Decode to reconstruct x
        x_re = self.decoder(z_enc)
        # Encode again to obtain z_re, while stopping gradient of x_re
        z_re_mu, z_re_log_sig2 = self.encoder(x_re.detach())

        # Take random z-vector
        z_p = torch.randn_like(z_enc, device=self.device)
        # Reconstruct random vector
        x_p = self.decoder(z_p)
        # Encode again to obtain z_pp, while stopping gradient of x_p
        z_pp_mu, z_pp_log_sig2 = self.encoder(x_p.detach())

        # 3. Determine the losses

        # Autoencoder loss -> AE
        l_rec = self.beta * self.ae_loss(x_re, x)

        # Regression loss -> VAE
        l_kl_z = self.gamma * self.kl_loss(z_mu, z_log_sig2)

        # Adversarial Part: define regressions
        l_kl_z_re = self.kl_loss(z_re_mu, z_re_log_sig2)
        l_kl_z_pp = self.kl_loss(z_pp_mu, z_pp_log_sig2)

        # Adversarial part for Encoder -> GAN
        l_adv_enc = self.alpha * 0.5 * \
            (torch.clamp(self.m - l_kl_z_re, min=0) +
             torch.clamp(self.m - l_kl_z_pp, min=0))

        # Set loss of Enc
        L_enc = l_rec + l_kl_z + l_adv_enc

        # Update if necessary
        if update:
            # 4. Update Encoder
            # ---------------------
            # Backpropagate Enc loss while saving the losses
            L_enc.backward(retain_graph=True)
            # Update Enc
            self.optimizerEnc.step()

        # Encode again to obtain z_re, without stopping gradient of x_re
        z_re_mu, z_re_log_sig2 = self.encoder(x_re)
        # Encode again to obtain z_pp, without stopping gradient of x_p
        z_pp_mu, z_pp_log_sig2 = self.encoder(x_p)

        # recalculate losses
        l_kl_z_re = self.kl_loss(z_re_mu, z_re_log_sig2)
        l_kl_z_pp = self.kl_loss(z_pp_mu, z_pp_log_sig2)

        # Adversarial part for Decoder -> GAN
        l_adv_dec = self.alpha * 0.5 * (l_kl_z_re + l_kl_z_pp)

        # Set loss of Dec
        L_dec = 0
        L_dec += l_adv_dec  # L_ae exists from backprop of previous branch already

        # Update if necessary
        if update:
            # 5. Update Decoder
            # ---------------------
            L_dec.backward()
            # Update Dec
            self.optimizerDec.step()
            # Return the Output
            return x_re

        else:
            # Track the current losses
            L_dec += l_rec + l_kl_z  # Add to watch true loss

            # setup dictionary for Tracking
            tr_data = {}
            tr_data["l_rec"] = l_rec.item()
            tr_data["l_kl_zec"] = l_kl_z.item()
            tr_data["l_adv_enc"] = l_adv_enc.item()
            tr_data["l_adv_dec"] = l_adv_dec.item()
            tr_data["L_enc"] = L_enc.item()
            tr_data["L_dec"] = L_dec.item()

            # Return output and tracking data
            return x_re, tr_data