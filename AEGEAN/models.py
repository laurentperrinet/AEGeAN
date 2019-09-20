import os
import numpy as np

import itertools

# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# from torch.utils.tensorboard import SummaryWriter
#
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# see research/UniStellar/AlignAndStack/2019-06-27-B1c_linear_inverter-decorrelation.ipynb
from torch.nn.functional import conv2d
KW = torch.zeros(size=(1, 1, 3, 3))
KW[0, 0, :, 0] = torch.Tensor([0, -1, 0])
KW[0, 0, :, 1] = torch.Tensor([-1, 4, -1])
KW[0, 0, :, 2] = torch.Tensor([0, -1, 0])
# KW = KW * torch.ones(1, 3, 1, 1)
KW = KW * torch.eye(3).view(3, 3, 1, 1)
KW /= np.sqrt(4. * 3)
Kinv = torch.zeros((1, 1, 3, 3))
Kinv[0, 0, :, 0] = torch.Tensor([.75, 1.5, .75])
Kinv[0, 0, :, 1] = torch.Tensor([1.5, 4.5, 1.5])
Kinv[0, 0, :, 2] = torch.Tensor([.75, 1.5, .75])
# Kinv = Kinv * torch.ones(1, 3, 1, 1)
Kinv = Kinv * torch.eye(3).view(3, 3, 1, 1)
Kinv /= np.sqrt(4. * 3)
# print(conv2d(KW, Kinv, padding=1))
# print(conv2d(Kinv, KW, padding=1))
# print(KW, Kinv, conv2d(KW, Kinv, padding=1), conv2d(Kinv, KW, padding=1))

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.opt = opt
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, stride=opt.stride,
                         padding=opt.padding, padding_mode='zeros')
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]


        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum))
            return block

        # use a different layer in the encoder using similarly max_filters
        # self.channels[3] = 512

        self.conv1 = nn.Sequential(*encoder_block(opt.channels, self.channels[0], bn=False),)
        self.conv2 = nn.Sequential(*encoder_block(self.channels[0], self.channels[1]),)
        self.conv3 = nn.Sequential(*encoder_block(self.channels[1], self.channels[2]),)
        self.conv4 = nn.Sequential(*encoder_block(self.channels[2], self.channels[3]),)

        self.init_size = opt.img_size // opts_conv['stride']**4
        self.vector = nn.Linear(self.channels[3] * self.init_size ** 2, opt.latent_dim)
        # self.sigmoid = nn.Sequential(nn.Sigmoid(),)

    def forward(self, img):
        if self.opt.verbose: print("Encoder")
        if self.opt.verbose: print("Image shape : ",img.shape)
        out = img #.copy()
        if self.opt.do_whitening:
            # for i in range(self.opt.channels):
            #     out[:, i, :, :] = conv2d(out[:, i, :, :], KW, padding=1)
            out = conv2d(out, KW, padding=1)
        if self.opt.verbose: print("WImage shape : ",out.shape)
        out = self.conv1(img)
        if self.opt.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        if self.opt.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        if self.opt.verbose: print("Conv3 out : ",out.shape)
        out = self.conv4(out)
        if self.opt.verbose: print("Conv4 out : ",out.shape, " init_size=", self.init_size)

        out = out.view(out.shape[0], -1)
        if self.opt.verbose: print("View out : ",out.shape)
        z = self.vector(out)
        # z = self.sigmoid(z)
        if self.opt.verbose: print("Z : ",z.shape)

        return z

    def _name(self):
        return "Encoder"

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, stride=opt.stride,
                         padding=opt.padding, padding_mode='zeros')
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]


        def generator_block(in_filters, out_filters):
            block = [nn.UpsamplingNearest2d(scale_factor=opts_conv['stride']), nn.Conv2d(in_filters, out_filters, kernel_size=opts_conv['kernel_size'], stride=1, padding=opts_conv['padding'], padding_mode=opts_conv['padding_mode']), nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum), NL]

            return block

        self.opt = opt
        self.init_size = opt.img_size // opts_conv['stride']**3
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.channels[3] * self.init_size ** 2), NL)


        self.conv1 = nn.Sequential(*generator_block(self.channels[3], self.channels[2]),)
        self.conv2 = nn.Sequential(*generator_block(self.channels[2], self.channels[1]),)
        self.conv3 = nn.Sequential(*generator_block(self.channels[1], self.channels[0]),)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.channels[0], opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        if self.opt.verbose: print("G")
        # Dim : opt.latent_dim
        out = self.l1(z)
        if self.opt.verbose: print("l1 out : ",out.shape)
        out = out.view(out.shape[0], self.channels[3], self.init_size, self.init_size)
        # Dim : (self.channels[3], opt.img_size/8, opt.img_size/8)
        if self.opt.verbose: print("View out : ",out.shape)

        out = self.conv1(out)
        # Dim : (self.channels[3]/2, opt.img_size/4, opt.img_size/4)
        if self.opt.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        # Dim : (self.channels[3]/4, opt.img_size/2, opt.img_size/2)
        if self.opt.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        # Dim : (self.channels[3]/8, opt.img_size, opt.img_size)
        if self.opt.verbose: print("Conv3 out : ",out.shape)

        img = self.conv_blocks(out)
        # Dim : (opt.chanels, opt.img_size, opt.img_size)
        if self.opt.verbose: print("img out : ", img.shape)

        if self.opt.do_whitening:
            img = conv2d(img, Kinv, padding=1)

        return img

    def _name(self):
        return "Generator"

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, stride=opt.stride,
                         padding=opt.padding, padding_mode='zeros')
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]


        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]#, nn.Dropout2d(0.25)
            if bn:
                # https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d
                block.append(nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum))
            return block

        self.opt = opt

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, self.channels[0], bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(self.channels[0], self.channels[1]),)
        self.conv3 = nn.Sequential(*discriminator_block(self.channels[1], self.channels[2]),)
        self.conv4 = nn.Sequential(*discriminator_block(self.channels[2], self.channels[3]),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opts_conv['stride']**4
        self.adv_layer = nn.Sequential(nn.Linear(self.channels[3] * self.init_size ** 2, 1))#, nn.Sigmoid()

    def forward(self, img):
        if self.opt.verbose:
            print("D")
            print("Image shape : ",img.shape)
            # Dim : (opt.chanels, opt.img_size, opt.img_size)
        out = img
        if self.opt.do_whitening:
            out = conv2d(out, KW, padding=1)

        out = self.conv1(img)
        if self.opt.verbose:
            print("Conv1 out : ",out.shape)

        out = self.conv2(out)
        if self.opt.verbose:
            print("Conv2 out : ",out.shape)

        out = self.conv3(out)
        if self.opt.verbose:
            print("Conv3 out : ",out.shape)

        out = self.conv4(out)
        if self.opt.verbose:
            print("Conv4 out : ",out.shape)

        out = out.view(out.shape[0], -1)
        if self.opt.verbose:
            print("View out : ",out.shape)

        validity = self.adv_layer(out)
        if self.opt.verbose:
            print("Val out : ",validity.shape)
            # Dim : (1)

        return validity

    def _name(self):
        return "Discriminator"
