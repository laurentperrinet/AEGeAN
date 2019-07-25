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

opts_conv = dict(kernel_size=9, stride=2, padding=4, padding_mode='zeros')
channels = [64, 128, 256, 512]


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.opt = opt
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        # use a different layer in the encoder using similarly max_filters
        # channels[3] = 512

        self.conv1 = nn.Sequential(*encoder_block(opt.channels, channels[0], bn=False),)
        self.conv2 = nn.Sequential(*encoder_block(channels[0], channels[1]),)
        self.conv3 = nn.Sequential(*encoder_block(channels[1], channels[2]),)
        self.conv4 = nn.Sequential(*encoder_block(channels[2], channels[3]),)

        self.init_size = opt.img_size // opts_conv['stride']**4
        self.vector = nn.Linear(channels[3] * self.init_size ** 2, opt.latent_dim)
        # self.sigmoid = nn.Sequential(nn.Sigmoid(),)

    def forward(self, img):
        if self.opt.verbose: print("Encoder")
        if self.opt.verbose: print("Image shape : ",img.shape)
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


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)

        def generator_block(in_filters, out_filters):
            block = [nn.UpsamplingNearest2d(scale_factor=opts_conv['stride']), nn.Conv2d(in_filters, out_filters, kernel_size=opts_conv['kernel_size'], stride=1, padding=opts_conv['padding'], padding_mode=opts_conv['padding_mode']), nn.BatchNorm2d(out_filters, opt.eps), NL]

            return block

        self.opt = opt
        self.init_size = opt.img_size // opts_conv['stride']**3
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, channels[3] * self.init_size ** 2), NL)


        self.conv1 = nn.Sequential(*generator_block(channels[3], channels[2]),)
        self.conv2 = nn.Sequential(*generator_block(channels[2], channels[1]),)
        self.conv3 = nn.Sequential(*generator_block(channels[1], channels[0]),)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels[0], opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        if self.opt.verbose: print("G")
        # Dim : opt.latent_dim
        out = self.l1(z)
        if self.opt.verbose: print("l1 out : ",out.shape)
        out = out.view(out.shape[0], channels[3], self.init_size, self.init_size)
        # Dim : (channels[3], opt.img_size/8, opt.img_size/8)
        if self.opt.verbose: print("View out : ",out.shape)

        out = self.conv1(out)
        # Dim : (channels[3]/2, opt.img_size/4, opt.img_size/4)
        if self.opt.verbose: print("Conv1 out : ",out.shape)
        out = self.conv2(out)
        # Dim : (channels[3]/4, opt.img_size/2, opt.img_size/2)
        if self.opt.verbose: print("Conv2 out : ",out.shape)
        out = self.conv3(out)
        # Dim : (channels[3]/8, opt.img_size, opt.img_size)
        if self.opt.verbose: print("Conv3 out : ",out.shape)

        img = self.conv_blocks(out)
        # Dim : (opt.chanels, opt.img_size, opt.img_size)
        if self.opt.verbose: print("img out : ", img.shape)

        return img

    def _name(self):
        return "Generator"

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), NL]#, nn.Dropout2d(0.25)
            if bn:
                block.append(nn.BatchNorm2d(out_filters, opt.eps))
            return block

        self.opt = opt

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, channels[0], bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(channels[0], channels[1]),)
        self.conv3 = nn.Sequential(*discriminator_block(channels[1], channels[2]),)
        self.conv4 = nn.Sequential(*discriminator_block(channels[2], channels[3]),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opts_conv['stride']**4
        self.adv_layer = nn.Sequential(nn.Linear(channels[3] * self.init_size ** 2, 1))#, nn.Sigmoid()

    def forward(self, img):
        if self.opt.verbose:
            print("D")
            print("Image shape : ",img.shape)
            out = self.conv1(img)
            print("Conv1 out : ",out.shape)
            out = self.conv2(out)
            print("Conv2 out : ",out.shape)
            out = self.conv3(out)
            print("Conv3 out : ",out.shape)
            out = self.conv4(out)
            print("Conv4 out : ",out.shape)

            out = out.view(out.shape[0], -1)
            print("View out : ",out.shape)
            validity = self.adv_layer(out)
            print("Val out : ",validity.shape)
        else:
            # Dim : (opt.chanels, opt.img_size, opt.img_size)
            out = self.conv1(img)
            # Dim : (channels[3]/8, opt.img_size/2, opt.img_size/2)
            out = self.conv2(out)
            # Dim : (channels[3]/4, opt.img_size/4, opt.img_size/4)
            out = self.conv3(out)
            # Dim : (channels[3]/2, opt.img_size/4, opt.img_size/4)
            out = self.conv4(out)
            # Dim : (channels[3], opt.img_size/8, opt.img_size/8)

            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)
            # Dim : (1)

        return validity

    def _name(self):
        return "Discriminator"
