# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/24_mocoae.ipynb (unless otherwise specified).

__all__ = ['MoCoAE', 'concat_all_gather', 'copy_q2k_params', 'momentum_update', 'MomentumContrastiveLoss', 'ce_loss']

# Cell
import torch
from torch import nn, optim
import torch.nn.functional as F
from ..architecture import Encoder, Decoder, DownUpConv
from ..abs_model import AbsModel
import numpy as np

# Cell


class MoCoAE(AbsModel):
    """
    The MoCoAE contains the Autoencoder based Architecture and the modified Pretext task
    """

    def __init__(self, device, args):
        """init the network"""
        super(MoCoAE, self).__init__(args)
        self.device = device  # GPU
        self.dim = args.dim  # 2/3 Dimensional input
        self.n_z = args.n_z  # Compression

        ### MoCo specific args
        self.K = args.moco_K  # limit of the queue
        self.tau = args.moco_tau  # temperature
        self.m = args.moco_m  # momentum

        # Modes:
        self.ae_mode = args.moco_aemode
        self.gan_mode = args.moco_ganmode

        # Encoder
        self.enc_q = Encoder(args, vae_mode=False).to(self.device)  # query encoder
        self.enc_k = Encoder(args, vae_mode=False).to(self.device)  # key encoder

        #balancing parameter for MOCO
        self.W = nn.Parameter(torch.randn([args.n_z, args.n_z], device=self.device))

        # set the params of the k-network to be equal to the q-network:
        copy_q2k_params(self.enc_q, self.enc_k)

        # Initialise the randomised Queues for Momentum Contrastive Learning
        self.register_queue("enc_queue")

        # Save the pointer position as well
        self.register_buffer(
            "ptr_enc", torch.zeros(1, dtype=torch.long).to(self.device)
        )

        # optimizers
        self.optimizerEnc = optim.Adam(self.enc_q.parameters(), lr=args.lr)
        self.optimizerW = optim.Adam([self.W], lr=args.lr)

        # Autoencoder init?
        self.init_ae(args) if self.ae_mode else None

        # GAN init?
        self.init_gan(args) if self.gan_mode else None

        # override prep and take
        self.prep = self.prep_3D if args.dataset_type == "MRNet" else self.prep_2D
        self.take = self.take_3D if args.dataset_type == "MRNet" else self.take_2D

    def init_ae(self, args):
        """Init the Autoencoder specific parts"""

        # AE -> as augmentation
        self.mse_loss = nn.MSELoss()

        self.ae_enc = Encoder(args, vae_mode=False).to(self.device)  # ae encoder
        self.ae_dec = Decoder(args).to(self.device)  # ae decoder

        ae_params = (
            list(self.ae_enc.parameters())
            + list(self.ae_dec.parameters())
        )
        self.optimizerAE = optim.Adam(ae_params, lr=args.lr)

    def init_gan(self, args):
        """Init the GAN specific parts"""
        self.gan_head = nn.Sequential(
            nn.Linear(args.n_z, args.n_z),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.n_z, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.gen = Decoder(args).to(self.device)  # generator
        self.real_label = 1
        self.fake_label = 0
        self.gan_loss = nn.BCELoss()
        self.optimizerGen = optim.Adam(self.gen.parameters(), lr=args.lr)
        self.optimizerGan = optim.Adam(self.gan_head.parameters(), lr=args.lr)

    def prep_2D(self, data):
        return data[0][0]

    def prep_3D(self, data, key="img"):
        return data[key]

    def take_2D(self, data):
        return data[0][0], data[0][1]

    def take_3D(self, data, key="img"):
        return data[key], data[key]

    @torch.no_grad()
    def register_queue(self, name: str):
        """
        Register the queue as a buffer with no parameters in the state dict
        """
        # create the queue
        self.register_buffer(name, torch.randn(self.n_z, self.K).to(self.device))
        setattr(self, name, nn.functional.normalize(getattr(self, name), dim=0))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the Queue and Pointer ->
        available in mode 'enc' and 'dec'
        """
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(getattr(self, "ptr_enc"))
        indixes = list(range(ptr, ptr + batch_size))

        if ptr + batch_size > self.K:
            ind1 = list(range((ptr + batch_size) % self.K))
            ind2 = list(range(ptr, self.K))
            indixes = ind1 + ind2

        # replace the keys at ptr (dequeue and enqueue)
        self.enc_queue[:, indixes] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.ptr_enc[0] = ptr

    def ae_forward(self, x, update):
        """
        Classic regression part of a normal Autoencoder
        """
        self.optimizerAE.zero_grad()

        ############################
        # Autoencoder Training
        ###########################
        z = self.enc_q(x)
        z = nn.functional.normalize(z, dim=1)
        x_r = self.ae_dec(z)

        # loss
        ae_loss = self.mse_loss(x_r, x)
        ae_loss.backward(retain_graph=True) if update else None

        return x_r, ae_loss.item()

    def gan_forward(self, x_q, k, update):
        """
        Gan part taking the original image and the key to determine between true / fake
        """
        b_size = k.size(0)
        self.optimizerGan.zero_grad()

        ############################
        # (1) Discriminator Training
        ###########################

        # true
        label = torch.full((b_size,), self.real_label, device=self.device)
        q_real = self.enc_q(x_q)
        output = self.gan_head(q_real).view(-1)
        d_loss_real = self.gan_loss(output, label)
        d_loss_real.backward() if update else None

        # fake
        label.fill_(self.fake_label)
        x_a = self.gen(k.detach())
        q_fake = self.enc_q(x_a.detach())
        output = self.gan_head(q_fake).view(-1)
        d_loss_fake = self.gan_loss(output, label)
        d_loss_fake.backward() if update else None
        d_loss = d_loss_fake.item() + d_loss_real.item()

        # update now before adding wrong gradients!
        if update:
            self.optimizerEnc.step()
            self.optimizerGan.step()

        ############################
        # (2) Generator Training
        ###########################
        self.optimizerGen.zero_grad()

        # fake
        label.fill_(self.real_label)
        x_a = self.gen(k.detach())
        q_fake = self.enc_q(x_a)
        output = self.gan_head(q_fake).view(-1)
        g_loss = self.gan_loss(output, label)
        g_loss.backward() if update else None
        g_loss = g_loss.item()

        if update:
            self.optimizerGen.step()

        return x_a, d_loss, g_loss


    def forward(self, data, update=True):
        """
        Perform forward computation and update
        """
        # Reset Gradients
        self.optimizerEnc.zero_grad()
        self.optimizerW.zero_grad()

        # 1. Get the augmented data
        x_q, x_k = self.take(data)

        # 2. Send pictures to device
        x_q = x_q.to(self.device)
        x_k = x_k.to(self.device)

        # ae part if on
        loss_ae = 0
        if self.ae_mode:
            x_q, loss_ae = self.ae_forward(x_k, update)

        # 3. Encode
        q = self.enc_q(x_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.enc_k(x_k)
            k = nn.functional.normalize(k, dim=1)

        # Get the InfoNCE loss:
        loss_InfoNCE = MomentumContrastiveLoss(
            k, self.W, q, self.enc_queue, device=self.device, tau=self.tau
        )
        loss_InfoNCE.backward() if update else None
        loss_InfoNCE = loss_InfoNCE.item()
        # append keys to the queue
        self._dequeue_and_enqueue(k)

        # gan part if on
        d_loss, g_loss = 0, 0
        if self.gan_mode:
            x_q, d_loss, g_loss = self.gan_forward(fac, x_q, k, update)

        # Perform encoder update
        if update:
            # MOCO
            self.optimizerEnc.step() if not self.gan_mode else None
            momentum_update(self.enc_q, self.enc_k, self.m[0])

            # AE
            self.optimizerAE.step() if self.ae_mode else None

            # W update
            self.optimizerW.step() if self.gan_mode or self.ae_mode else None

            return x_q.detach()

        else:
            tr_data = {
                "loss_ae": loss_ae,
                "loss_InfoNCE": loss_InfoNCE,
                "d_loss": d_loss,
                "g_loss": g_loss,
            }
            return x_q.detach(), tr_data

# Cell
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# Cell
@torch.no_grad()
def copy_q2k_params(Q_network: nn.Module, K_network: nn.Module):
    """
    Helper function to Copy parameters from Network Q to network K.
    Further deactive gradient computation on k
    """
    for param_q, param_k in zip(Q_network.parameters(), K_network.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not updated by gradient

# Cell
@torch.no_grad()
def momentum_update(Q_network: nn.Module, K_network: nn.Module, m: float):
    """Momentum update of the key network based on the query network"""
    for param_q, param_k in zip(Q_network.parameters(), K_network.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1.0 - m)

# Cell
ce_loss = nn.CrossEntropyLoss()


def MomentumContrastiveLoss(k, W, q, queue, device, tau=1):
    """
    Calculate the loss of the network depending on the current key(k), the query(q)
    and the overall queue(queue)
    We follow the suggestion of the paper, Algorithm 1:
    https://arxiv.org/pdf/1911.05722.pdf
    """
    # positive logits: Nx1
    l_pos = torch.einsum("nc,cc,nc->n", [q, W, k]).unsqueeze(-1)

    # negative logits: NxK
    l_neg = torch.einsum("nc,cc,ck->nk", [q, W, queue.clone().detach()])

    # logits: Nx(1+K) with temperature
    logits = torch.cat([l_pos, l_neg], dim=1) / tau
    # substract max for stability
    logits.sub_(torch.max(logits, axis=1).values.unsqueeze(1))

    # positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # calculate the crossentropyloss
    loss = ce_loss(logits, labels)

    return loss