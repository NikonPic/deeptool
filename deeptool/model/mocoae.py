# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/24_mocoae.ipynb (unless otherwise specified).

__all__ = ['MoCoAE', 'concat_all_gather', 'copy_q2k_params', 'momentum_update', 'aug', 'MomentumContrastiveLoss',
           'ce_loss']

# Cell
import torch
from torch import nn, optim
import torch.nn.functional as F
from ..architecture import Encoder, Decoder, DownUpConv
from ..abs_model import AbsModel

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

        # AE Loss
        self.ae_mode = args.moco_aemode
        self.mse_loss = nn.MSELoss(reduction="mean")

        # Encoder
        self.enc_q = Encoder(args, vae_mode=False).to(self.device)  # query encoder
        self.enc_k = Encoder(args, vae_mode=False).to(self.device)  # key encoder

        # Decoder
        self.dec_q = Decoder(args).to(self.device)  # query decoder
        self.dec_k = Decoder(args).to(self.device)  # key decoder

        # set the params of the k-network to be equal to the q-network:
        copy_q2k_params(self.enc_q, self.enc_k)
        copy_q2k_params(self.dec_q, self.dec_k)

        # Initialise the randomised Queues for Momentum Contrastive Learning
        self.register_queue("enc_queue")
        self.register_queue("dec_queue")

        # Save the pointer position as well
        self.register_buffer(
            "ptr_enc", torch.zeros(1, dtype=torch.long).to(self.device)
        )
        self.register_buffer(
            "ptr_dec", torch.zeros(1, dtype=torch.long).to(self.device)
        )

        # optimizers
        self.optimizerEnc = optim.Adam(self.enc_q.parameters(), lr=args.lr)
        self.optimizerDec = optim.Adam(self.dec_q.parameters(), lr=args.lr)

        # override prep
        self.prep = self.prep_3D if args.dataset_type == "MRNet" else self.prep_2D
        self.take = self.take_3D if args.dataset_type == "MRNet" else self.take_2D

    @torch.no_grad()
    def register_queue(self, name: str):
        """
        Register the queue as a buffer with no parameters in the state dict
        """
        # create the queue
        self.register_buffer(name, torch.randn(self.n_z, self.K).to(self.device))
        setattr(self, name, nn.functional.normalize(getattr(self, name), dim=0))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, mode="enc"):
        """
        Update the Queue and Pointer ->
        available in mode 'enc' and 'dec'
        """
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(getattr(self, f"ptr_{mode}"))
        indixes = list(range(ptr, ptr + batch_size))

        if ptr + batch_size > self.K:
            ind1 = list(range((ptr + batch_size) % self.K))
            ind2 = list(range(ptr, self.K))
            indixes = ind1 + ind2

        # replace the keys at ptr (dequeue and enqueue)
        if mode == "enc":
            self.enc_queue[:, indixes] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.ptr_enc[0] = ptr

        # mode is 'dec'
        else:
            self.dec_queue[:, indixes] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.ptr_dec[0] = ptr

    def ae_forward(self, x, update):
        """
        Classic regression part of a normal Autoencoder
        """
        # encode
        z = self.enc_q(x)
        z = nn.functional.normalize(z, dim=1)
        # decode
        x_r = self.dec_q(z)
        # loss
        ae_loss = self.mse_loss(x_r, x)
        # backprop
        ae_loss.backward() if update else None
        return ae_loss

    def prep_2D(self, data):
        return data[0][0]

    def prep_3D(self, data, key="img"):
        return data[key]

    def take_2D(self, data):
        return data[0][0], data[0][1]

    def take_3D(self, data, key="img"):
        return data[key], data[key]

    def forward(self, data, update=True):
        """
        Perform forward computaion and update
        """
        # Reset Gradients
        self.optimizerEnc.zero_grad()
        self.optimizerDec.zero_grad()

        # 1. Get the augmented data
        x_q, x_k = self.take(data)

        # 2. further we will apply additional augmentation to the picture!
        x_q = x_q.to(self.device)
        x_k = x_k.to(self.device)

        # ae part if on
        ae_loss = self.ae_forward(x_q, update) if self.ae_mode else None

        # 3. Encode
        q = self.enc_q(x_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.enc_k(x_k)
            k = nn.functional.normalize(k, dim=1)

        # Get the InfoNCE loss:
        loss_enc = MomentumContrastiveLoss(
            k, q, self.enc_queue, self.tau, device=self.device
        )

        # Perform encoder update
        if update:
            loss_enc.backward()

            # update the Query Encoder
            self.optimizerEnc.step()

            # update the Key Encoder with Momentum update
            momentum_update(self.enc_q, self.enc_k, self.m[0])

        # 4. Decode
        x_re = self.dec_q(k.detach())

        # 5. Encode again using the k-network to focus on decoder only!:
        kk = self.enc_k(x_re)
        kk = nn.functional.normalize(kk, dim=1)

        # Get the InfoNCE loss:
        loss_dec = MomentumContrastiveLoss(
            kk, k, self.enc_queue, self.tau, device=self.device
        )

        # perform decoder update
        if update:
            loss_dec.backward()

            # update the Query Decoder
            self.optimizerDec.step()

        # append keys to the queue
        self._dequeue_and_enqueue(k, mode="enc")

        if update:
            return x_re.detach()

        else:
            tr_data = {
                "loss_enc": loss_enc.item(),
                "loss_dec": loss_dec.item(),
            }
            if self.ae_mode:
                tr_data["ae_loss"] = ae_loss.item()
            return x_re.detach(), tr_data

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
import torch
import torchvision
from ..dataloader import RandomCrop, Rescale


@torch.no_grad()
def aug(x):
    """perform random data augmentation on an image batch"""
    # ToDo
    return x

# Cell
ce_loss = nn.CrossEntropyLoss()


def MomentumContrastiveLoss(k, q, queue, tau, device):
    """
    Calculate the loss of the network depending on the current key(k), the query(q)
    and the overall queue(queue)
    We follow the suggestion of the paper, Algorithm 1:
    https://arxiv.org/pdf/1911.05722.pdf
    """
    N, C = q.shape
    K = k.shape[1]

    # positive logits: Nx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

    # logits: Nx(1+K) with temperature
    logits = torch.cat([l_pos, l_neg], dim=1) / tau

    # positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # calculate the crossentropyloss
    loss = ce_loss(logits, labels)

    return loss