# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/23_bigan.ipynb (unless otherwise specified).

__all__ = ['DisBiGan', 'BiGAN']

# Cell

import torch
from torch import nn, optim
from ..architecture import Encoder, Decoder, DownUpConv, weights_init
from ..abs_model import AbsModel

# Cell


class DisBiGan(nn.Module):
    """
    The redefined Discriminator for Bigan:
    Contains the classic discriminator and concatenates the input with the hiddem dimension
    """

    def __init__(self, args):
        """The Discriminator for BiGan: will include Conv and fully part"""
        super(DisBiGan, self).__init__()

        # convolutional neural network
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
        self.last_dim = int(args.n_z / 2)

        # Finish with fully connected layers
        self.fc_part_sxz = nn.Sequential(
            # State size batch x (cur_fea*4*4*4)
            nn.Linear(self.hidden_dim + args.n_z, self.last_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=args.p_drop),
            # Output size batch x n_z
            nn.Linear(self.last_dim, 1, bias=False),
            nn.Sigmoid(),
            # Output size batch x 1
        )
        self.fc_part_sxz.apply(weights_init)

        self.forward = self.forward_normal
        # add two further fully connected parts if the extension is set to "TRUE"
        if args.bi_extension:
            self.forward = self.forward_extension
            # picture extension
            self.fc_part_sx = nn.Sequential(
                # State size batch x (cur_fea*4*4*4)
                nn.Linear(self.hidden_dim, self.last_dim, bias=False),
                nn.Dropout(p=args.p_drop),
                nn.LeakyReLU(0.2, inplace=True),
                # Output size batch x n_z
                nn.Linear(self.last_dim, 1, bias=False),
                nn.Sigmoid(),
                # Output size batch x 1
            )
            self.fc_part_sx.apply(weights_init)

            # latent extension
            self.fc_part_sz = nn.Sequential(
                # State size batch x (cur_fea*4*4*4)
                nn.Linear(args.n_z, self.last_dim, bias=False),
                nn.Dropout(p=args.p_drop),
                nn.LeakyReLU(0.2, inplace=True),
                # Output size batch x n_z
                nn.Linear(self.last_dim, 1, bias=False),
                nn.Sigmoid(),
                # Output size batch x 1
            )
            self.fc_part_sz.apply(weights_init)

    def forward_normal(self, inp):
        """
        Perform forward calculation
        input is tuple of: picture x and laten z
        """
        x, z = inp
        # first apply convolutions on picture x
        x = self.conv_part(x)
        # Resize
        x = x.view((-1, self.hidden_dim))
        # now concatenate to get the extend dimension
        x = torch.cat([x, z], dim=1)
        # now apply the fully connected part on these
        x = self.fc_part_sxz(x)
        # return the determined value
        return x

    def forward_extension(self, inp):
        """
        Perform forward calculation
        input is tuple of: picture x and laten z
        """
        x, z = inp
        x = self.conv_part(x)
        x = x.view((-1, self.hidden_dim))
        # get the value for sxz
        s_xz = torch.cat([x, z], dim=1)
        s_xz = self.fc_part_sxz(x)
        # get the value for sx
        s_x = self.fc_part_sx(x)
        # get the value for sz
        s_z = self.fc_part_sx(z)
        return s_xz, s_x, s_z

# Cell


class BiGAN(AbsModel):
    """
    The Bidirectional Generative adversarial network
    based on https://arxiv.org/abs/1605.09782
    extension based on: https://arxiv.org/abs/1907.02544
    """

    def __init__(self, device, args):
        """
        network architecture
        """
        super(BiGAN, self).__init__(args)
        self.device = device
        self.dim = args.dim
        self.n_z = args.n_z

        # Loss to be optimized for dcgan
        self.loss = nn.BCELoss()

        # labeling
        self.real_label = 1
        self.fake_label = 0

        # the three relevant networks
        self.decoder = Decoder(args).to(self.device)
        self.encoder = Encoder(args, vae_mode=False).to(self.device)
        self.discriminator = DisBiGan(args).to(self.device)

        # parameters
        self.lam = args.lam

        # the optimizers
        self.optimizerDec = optim.Adam(
            self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999)
        )
        self.optimizerEnc = optim.Adam(
            self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999)
        )
        self.optimizerDis = optim.Adam(
            self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
        )

        # Fixed noise to visualize progression
        self.batch_size = args.batch_size
        self.fixed_noise = torch.randn(self.batch_size, self.n_z, device=self.device)

    def sample_noise(self, batch_size, update):
        """
        Sample the latent noise required for training
        """
        if update:
            return torch.randn(batch_size, self.n_z, device=self.device)
        return self.fixed_noise

    def calc_gradient_penalty(self, real, fake):
        """
        Apply the gradient Penalty for Discriminator training
        This is responsible for ensuring the Lipschitz constraint,
        which is required to ensure the Wasserstein distance.
        modified from: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
        """
        # Asssign random factor alpha between 0 and 1
        sh = real.shape
        b_size = sh[0]
        alpha = torch.rand(b_size, 1)
        alpha = (
            alpha.expand(b_size, int(real.nelement() / b_size)).contiguous().view(sh)
        )
        alpha = alpha.to(self.device)

        # interpolating as disc input
        interpolates = (alpha * real + ((1 - alpha) * fake)).to(self.device)
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

    def forward(self, data, update=True):
        """
        Calculate output and update networks with dcgan
        """
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # ------------------------------------------------------------
        # 1.1 Train with all-real batch
        # Get the true data
        real_x = self.prep(data).to(self.device)
        real_z = self.encoder(real_x)
        real = (real_x, real_z.detach())

        self.discriminator.zero_grad()

        # fill the labels
        b_size = real_x.size(0)
        label = torch.full((b_size,), self.real_label, device=self.device)

        # True set
        output = self.discriminator(real).view(-1)
        label.fill_(self.real_label)
        errD_real = self.loss(output, label)
        errD_real.backward() if update else None
        D_x = output.mean().item()

        # 1.2 Train with all-fake batch
        fake_z = self.sample_noise(b_size, update)
        fake_x = self.decoder(fake_z)
        fake = (fake_x.detach(), fake_z.detach())

        # Fake set
        output = self.discriminator(fake).view(-1)
        label.fill_(self.fake_label)
        errD_fake = self.loss(output, label)
        errD_fake.backward() if update else None

        D_G_z1 = output.mean().item()

        # final discriminatro loss
        errD = errD_fake.item() + errD_real.item()

        # Update Discriminator
        if update:
            self.optimizerDis.step()

        # (2) Update G network: maximize 1 - log(D(G(z)))
        # ------------------------------------------------------------
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        # True set
        real = (real_x, real_z)
        output = self.discriminator(real).view(-1)
        label.fill_(self.fake_label)
        errE = self.loss(output, label)
        errE.backward() if update else None

        # Fake set
        fake = (fake_x, fake_z)
        output = self.discriminator(fake).view(-1)
        label.fill_(self.real_label)
        errD = self.loss(output, label)
        errD.backward() if update else None

        # Update Generator
        if update:
            self.optimizerEnc.step()
            self.optimizerDec.step()
            return fake_x

        else:
            # Track all relevant losses
            tr_data = {}
            tr_data["errD"] = errD
            tr_data["errG"] = errD.item()
            tr_data["D_x"] = D_x
            tr_data["D_G_z1"] = D_G_z1
            tr_data["D_G_z2"] = output.mean().item()

            # generate the autoencoder output:
            x_r = self.decoder(real_z)

            # Return losses and reconstruction data
            return x_r, tr_data