# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_diagnosis.ipynb (unless otherwise specified).

__all__ = ['Classify', 'ClassifyRNN', 'Compressor', 'TripleMRNet']

# Cell

import pdb
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import models
from ..abs_model import AbsModel

# Cell


class Classify(nn.Module):
    """
    The Classifier on top of the triplenet network
    """

    def __init__(self, in_dim, mid_dim, out_dim, p_drop=0.5):
        """init the classifier"""
        super(Classify, self).__init__()
        # reduction block
        self.reduce = nn.Sequential(
            nn.Linear(in_dim, mid_dim), nn.Dropout(p=p_drop), nn.ReLU(inplace=True),
        )
        # final block
        self.fin_block = nn.Sequential(nn.Linear(mid_dim, out_dim))

    def forward(self, x):
        """perform forward calculation"""
        # reduce
        x = self.reduce(x)
        x = self.fin_block(x)
        return x

# Cell

class ClassifyRNN(nn.Module):
    """
    The Classifier on top of the triplenet network
    """

    def __init__(self, device, input_size, hidden_size, n_layers=1):
        """init the classifier"""
        super(ClassifyRNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers).to(self.device)

    def init_hidden(self):
        """create zeros for hidden layer"""
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    def forward(self, x):
        """perform forward calculation"""
        x = x.view(-1, 1, self.input_size)
        # create hidden dimension
        hidden = self.init_hidden()
        # apply GRU
        _, hidden = self.gru(x, hidden)
        # use hidden for final linear layer (equal to last output)
        return hidden.view(1, -1)

# Cell

class Compressor(nn.Module):
    """
    This class compresses the data from all slices to be only a vector.
    Contains the backbone and the pooling
    """

    def __init__(
        self,
        device,
        args,
        train_data=None,
        backbone="resnet18",
        training=True,
        document=True,
    ):
        super(Compressor, self).__init__()

        # general defines
        self.device = device
        self.backbone = args.mrnet_backbone
        self.y_labels = args.classes
        self.y_len = len(self.y_labels)
        self.naming = args.perspectives

        # average together
        self.rnn_gap = args.mrnet_rnn_gap
        self.hidden_dim = args.mrnet_hidden_dim

        # define whether to use one network or multiple ones
        self.single_stream = args.mrnet_singlestream

        # build backbone networks
        self.axial_net = self.build_backbone(training)
        self.sagit_net = self.build_backbone(training)
        self.coron_net = self.build_backbone(training)

        # build gap and classifier
        # apply average pooling
        self.gap_axial = nn.AdaptiveAvgPool2d(1).to(self.device)
        self.gap_sagit = nn.AdaptiveAvgPool2d(1).to(self.device)
        self.gap_coron = nn.AdaptiveAvgPool2d(1).to(self.device)

        if self.rnn_gap:
            # only for the avergae case
            if self.backbone in ("resnet18", "vgg", "squeeze"):
                self.input_rnn = 512
            else:
                self.input_rnn = 256

            # the RNN gapping
            self.max_axial = ClassifyRNN(self.device, self.input_rnn, self.hidden_dim)
            self.max_sagit = ClassifyRNN(self.device, self.input_rnn, self.hidden_dim)
            self.max_coron = ClassifyRNN(self.device, self.input_rnn, self.hidden_dim)

        # redefine if single stream
        if self.single_stream:
            # make them all reference the same network
            self.sagit_net = self.axial_net
            self.coron_net = self.axial_net

            if self.rnn_gap:
                self.max_sagit = self.max_axial
                self.max_coron = self.max_axial

    def build_backbone(self, training):
        """
        Builds the desired backbone
        """
        if self.backbone == "resnet18":
            resnet = models.resnet18(pretrained=training)
            modules = list(resnet.children())[:-1]
            local_net = nn.Sequential(*modules)
            for param in local_net.parameters():
                param.requires_grad = False

        elif self.backbone == "resnet34":
            resnet = models.resnet34(pretrained=training)
            modules = list(resnet.children())[:-1]
            local_net = nn.Sequential(*modules)
            for param in local_net.parameters():
                param.requires_grad = False

        elif self.backbone == "alexnet":
            local_net = models.alexnet(pretrained=training)
            local_net = local_net.features

        elif self.backbone == "vgg":
            local_net = models.vgg11(pretrained=training)
            local_net = local_net.features

        elif self.backbone == "squeeze":
            local_net = models.squeezenet1_1(pretrained=training)
            local_net = local_net.features

        return local_net

    def apply_gap(self, vol_axial, vol_sagit, vol_coron):
        """
        applies the average / rnn gap
        """
        vol_axial = self.gap_axial(vol_axial).view(vol_axial.size(0), -1)
        vol_sagit = self.gap_sagit(vol_sagit).view(vol_sagit.size(0), -1)
        vol_coron = self.gap_coron(vol_coron).view(vol_coron.size(0), -1)

        if self.rnn_gap:
            # idea add spatial relation here
            x = self.max_axial(vol_axial)[0]
            y = self.max_sagit(vol_sagit)[0]
            z = self.max_coron(vol_coron)[0]

            w = torch.cat((x, y, z), 0)

        else:
            x = torch.max(vol_axial, 0, keepdim=True)[0]
            y = torch.max(vol_sagit, 0, keepdim=True)[0]
            z = torch.max(vol_coron, 0, keepdim=True)[0]

            w = torch.cat((x, y, z), 1)[0]

        return w

    def forward(self, vol_axial, vol_sagit, vol_coron):
        # apply the main networks
        vol_axial = self.axial_net(vol_axial)
        vol_sagit = self.sagit_net(vol_sagit)
        vol_coron = self.coron_net(vol_coron)

        # apply the gap
        w = self.apply_gap(vol_axial, vol_sagit, vol_coron)

        return w

# Cell
from .mocoae import momentumContrastiveLoss, momentum_update, copy_q2k_params, concat_all_gather

class TripleMRNet(AbsModel):
    """
    adapted from https://github.com/yashbhalgat/MRNet-Competition
    with the knowledge of: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699
    """

    def __init__(
        self,
        device,
        args,
        train_data=None,
        backbone="resnet18",
        training=True,
        document=True,
    ):
        super(TripleMRNet, self).__init__(args)

        # mid layer outcome
        self.rnn_gap = args.mrnet_rnn_gap
        self.hidden_dim = args.mrnet_hidden_dim

        # general defines
        self.device = device
        self.backbone = args.mrnet_backbone
        self.y_labels = args.classes
        self.y_len = len(self.y_labels)
        self.naming = args.perspectives

        # depending on whether train_data is specified
        if train_data == None:
            self.weights = {}
            self.weights["abn"] = [0.81, 0.19]
            self.weights["acl"] = [0.23, 0.77]
            self.weights["men"] = [0.43, 0.57]
        else:
            self.weights = train_data.weights

        # picture center cropping to 224 resolution
        self.pic_size = 224
        self.pad = int((args.pic_size - self.pic_size) / 2)
        self.factor = 1  # 1130 / 208  # inverse factor

        # internal count for updating
        self.int_count = 0
        self.batch_update = args.mrnet_batch_update
        self.label_smoothing = args.mrnet_label_smoothing

        # the actual big backbone network
        self.compressor = Compressor(device, args, train_data, backbone, training, document).to(self.device)

        # the Momentum Contrastive bool
        self.moco = args.mrnet_moco

        self.forward= self.forward_moco if self.moco else self.forward_normal

        # add classifier
        self.add_classifier()

        # final sigmoid layer
        self.sigmoid = nn.Sigmoid().to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        # init all parts relevant for moco
        self.init_moco(args, backbone, training) if self.moco else None

    def init_moco(self, args, backbone, training):
        """initialise all parts connected to MomentumContrastiveLearning"""
        # add a second compressor if moco is active
        self.m_compressor = Compressor(self.device, args, None, backbone, training, document).to(self.device)

    def add_classifier(self):
        """
        add the final classification part
        """
        if self.rnn_gap:
            self.classifier = Classify(3 * self.hidden_dim, self.hidden_dim, self.y_len).to(self.device)
        else:
            # only for the average case
            if self.backbone in ("resnet18", "vgg", "squeeze", "resnet34"):
                self.classifier = Classify(3 * 512, self.hidden_dim, self.y_len).to(
                    self.device
                )
            elif self.backbone == "alexnet":
                self.classifier = Classify(3 * 256, self.hidden_dim, self.y_len).to(
                    self.device
                )

    def weighted_loss(self, prediction, target, cl):
        """
        Calculate the weighted loss with label smoothing
        """
        # determin the weights
        weights_npy = np.array([self.weights[cl][int(target)]])
        weights_tensor = torch.FloatTensor(weights_npy)
        weights_tensor = weights_tensor.to(self.device)[0]

        # smooth the labels
        if self.label_smoothing > 0:
            target = target.add(self.label_smoothing).div(2)

        # calculate binary cross entropy
        loss = F.binary_cross_entropy_with_logits(
            prediction, target, weight=Variable(weights_tensor)
        )
        return loss

    @torch.no_grad()
    def watch_progress(self, valid_loader, iteration):
        """Outsourced to Tracker"""
        self.tracker.get_accuracy(self, valid_loader, iteration)

    @torch.no_grad()
    def get_vol(self, data, name):
        """
        helper func to load values
        """
        vol = data['img'][name]

        # two volumes if moco is active
        if self.moco:
            vol0 = vol[0][0, :, :, :].to(self.device)
            vol0 = torch.stack((vol0,) * 3, axis=1)

            vol1 = vol[1][0, :, :, :].to(self.device)
            vol1 = torch.stack((vol1,) * 3, axis=1)
            return vol0, vol1

        else:
            # no stacking necessary
            vol = vol[0, :, :, :].to(self.device)
            vol = torch.stack((vol,) * 3, axis=1)
            return vol

    @torch.no_grad()
    def get_input_image(self, data):
        """
        take the input from the stack and give the single volumes
        """
        # get the three volumes from the dictionary
        # data["img"]["axial"] -> shape = batch x depth x pic x pic
        vol_axial = self.get_vol(data, self.naming[0])
        vol_sagit = self.get_vol(data, self.naming[1])
        vol_coron = self.get_vol(data, self.naming[2])

        label = torch.zeros(vol_axial.shape[0], self.y_len)  # init
        for i, cl in enumerate(self.y_labels):
            label[:, i] = data[cl]
        label = label[0, :].to(self.device)

        return vol_axial, vol_sagit, vol_coron, label

    def forward_moco(self, data, update=True):
        """
        perform forward pass in moco mode
        """
        # get the required input
        vol_axial, vol_sagit, vol_coron, label = self.get_input_image(data)

        # compress to the (256) vector
        w = self.compressor(vol_axial[0], vol_sagit[0], vol_coron[0])



    def forward_normal(self, data, update=True):
        """
        perform the forward pass and update in normal mode
        """
        # get the required input
        vol_axial, vol_sagit, vol_coron, label = self.get_input_image(data)

        # compress to the (256) vector
        w = self.compressor(vol_axial, vol_sagit, vol_coron)

        logit = self.classifier(w)
        # accumulate losses
        loss = 0
        for i, cl in enumerate(self.y_labels):
            loss += self.weighted_loss(logit[i], label[i], cl)
        loss /= self.batch_update

        out = self.sigmoid(logit)

        if update:
            loss.backward()
            self.int_count += 1

            if self.int_count > self.batch_update:
                # finally take the update step
                self.int_count = 0
                self.optimizer.step()
                self.zero_grad()

            return out
        else:
            tr_data = {}
            tr_data["loss"] = loss.item()
            return out, label, tr_data