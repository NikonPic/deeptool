# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20_dcgan.ipynb (unless otherwise specified).

__all__ = ['DCGAN']

# Cell

# General Includes
import torch
from torch import nn, optim
from torch import autograd

# Personal includes
from ..architecture import Decoder, Discriminator
from ..abs_model import AbsModel

# Cell


class DCGAN(AbsModel):
    """
    Modification of the DCGAN-Paper https://arxiv.org/pdf/1511.06434.pdf for 3-Dimensional tasks in MR-Imaging
    oriented on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    def __init__(self, device, args):
        """
        Setup the general architecture for the DCGAN model, composed of:
        Generator, Discriminator
        """
        super(DCGAN, self).__init__(args)
        # dimension of networks (2d conv or 3d conv)
        self.dim = args.dim

        # Number of channels depending on perspectives
        self.device = device
        self.n_chan = len(args.perspectives)

        # Generator
        self.generator = Decoder(args).to(self.device)

        # Encoding dimension
        self.n_z = args.n_z
        self.batch_size = args.batch_size

        # Fixed noise to visualize progression
        self.fixed_noise = torch.randn(self.batch_size, self.n_z, device=self.device)

        # lambda factor for gradient penatly
        self.lam = args.lam

        # Loss to be optimized for dcgan
        self.loss = nn.BCELoss()

        self.real_label = 1
        self.fake_label = 0

        if args.wgan == True:
            self.name = "wgan"
            self.forward = self.forward_wgan
            # Discriminator
            self.discriminator = Discriminator(args, wgan=True).to(self.device)

        else:
            self.name = "dcgan"
            self.forward = self.forward_dcgan
            # Discriminator
            self.discriminator = Discriminator(args, wgan=False).to(self.device)

        # Optimizers
        self.optimizerGen = optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizerDis = optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

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
        disc_interpolates = self.discriminator(interpolates)

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

    def sample_noise(self, batch_size, update):
        """
        Sample the latent noise required for training
        """
        if update:
            return torch.randn(batch_size, self.n_z, device=self.device)
        return self.fixed_noise

    def forward_wgan(self, data, update=True):
        """
        Calculate output and update networks with wgan
        """
        # get the image data
        real_gpu = self.prep(data).to(self.device)

        # (1) Update D network: maximize D(x) - D(G(z))
        # 1.1 Train with all-real batch
        self.discriminator.zero_grad()
        b_size = real_gpu.size(0)
        output = self.discriminator(real_gpu).view(-1)
        errd_real = -torch.mean(output)

        # 1.2 Train with all-fake batch
        noise = self.sample_noise(b_size, update)
        fake = self.generator(noise)
        output = self.discriminator(fake.detach()).view(-1)
        errd_fake = torch.mean(output)

        # 1.3 assign Gradient penalty
        gradient_penalty = (
            self.calc_gradient_penalty(real_gpu, fake.detach()) if update else 0
        )

        # sum the losses up
        errd = errd_fake + errd_real + gradient_penalty

        # Update Discriminator
        if update:
            errd.backward()
            self.optimizerDis.step()

        # (2) Update G network: maximize D(G(z))
        self.generator.zero_grad()
        output = self.discriminator(fake).view(-1)
        errg = -torch.mean(output)

        # Update Generator
        if update:
            errg.backward()
            self.optimizerGen.step()
            return fake.detach()

        else:
            # Track all relevant losses
            tr_data = {}
            tr_data["errD"] = errd.item()
            tr_data["errG"] = errg.item()
            tr_data["D_x"] = errd_real.item()
            tr_data["D_G_z1"] = errd_fake.item()
            tr_data["D_G_z2"] = output.mean().item()
            # Return losses and fake data
            return fake.detach(), tr_data

    def forward_dcgan(self, data, update=True):
        """
        Calculate output and update networks with dcgan
        """
        # get the image data
        real_gpu = self.prep(data).to(self.device)

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        # 1.1 Train with all-real batch
        self.discriminator.zero_grad()
        b_size = real_gpu.size(0)
        label = torch.full((b_size,), self.real_label, device=self.device, dtype=torch.float32)
        output = self.discriminator(real_gpu).view(-1)
        errd_real = self.loss(output, label)
        errd_real.backward() if update else None
        d_x = output.mean().item()

        # 1.2 Train with all-fake batch
        noise = self.sample_noise(b_size, update)
        fake = self.generator(noise)
        output = self.discriminator(fake.detach()).view(-1)
        label.fill_(self.fake_label)
        errd_fake = self.loss(output, label)
        errd_fake.backward() if update else None
        self.optimizerDis.step() if update else None

        d_g_z1 = output.mean().item()
        errd = errd_fake.item() + errd_real.item()

        # (2) Update G network: maximize 1 - log(D(G(z)))
        self.generator.zero_grad()
        label.fill_(self.real_label)
        output = self.discriminator(fake).view(-1)
        errg = self.loss(output, label)

        # Update Generator
        if update:
            errg.backward()
            self.optimizerGen.step()
            return fake.detach()

        else:
            # Track all relevant losses
            tr_data = {}
            tr_data["errD"] = errd
            tr_data["errG"] = errg.item()
            tr_data["D_x"] = d_x
            tr_data["D_G_z1"] = d_g_z1
            tr_data["D_G_z2"] = output.mean().item()
            # Return losses and fake data
            return fake.detach(), tr_data