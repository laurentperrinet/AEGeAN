import os
import numpy as np

import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, stride=opt.stride,
                         padding=opt.padding, padding_mode='zeros', bias=opt.do_bias)
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), ]
            if bn and (not opt.bn_eps == np.inf):
                block.append(nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum))
            block.append(NL)
            return block

        self.conv1 = nn.Sequential(*encoder_block(opt.channels, self.channels[0], bn=False),)
        self.conv2 = nn.Sequential(*encoder_block(self.channels[0], self.channels[1]),)
        self.conv3 = nn.Sequential(*encoder_block(self.channels[1], self.channels[2]),)
        self.conv4 = nn.Sequential(*encoder_block(self.channels[2], self.channels[3]),)

        self.init_size = opt.img_size // opt.stride**4
        # self.vector = nn.Linear(self.channels[3] * self.init_size ** 2, opt.latent_dim)
        self.vector = nn.Sequential(
            nn.Linear(self.channels[3] * self.init_size ** 2, opt.latent_dim),
            nn.Tanh(),
        )

        self.opt = opt

    def forward(self, img):
        if self.opt.verbose:
            print("Encoder")
        if self.opt.verbose:
            print("Image shape : ", img.shape)
        out = img
        # if self.opt.do_whitening:
        #     out = conv2d(out, KW, padding=1)
        if self.opt.verbose:
            print("WImage shape : ", out.shape)
        out = self.conv1(out)
        if self.opt.verbose:
            print("Conv1 out : ", out.shape)
        out = self.conv2(out)
        if self.opt.verbose:
            print("Conv2 out : ", out.shape)
        out = self.conv3(out)
        if self.opt.verbose:
            print("Conv3 out : ", out.shape)
        out = self.conv4(out)
        if self.opt.verbose:
            print("Conv4 out : ", out.shape)

        out = out.view(out.shape[0], -1)
        if self.opt.verbose:
            print("View out : ", out.shape, " init_size=", self.init_size)
        out = self.vector(out)
        if self.opt.verbose:
            print("Z : ", out.shape)

        return out

    def _name(self):
        return "Encoder"


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, bias=opt.do_bias,
                         padding=opt.padding, padding_mode='zeros')
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]

        def generator_block(in_filters, out_filters, bn=True):
            block = [nn.UpsamplingNearest2d(scale_factor=opt.stride),
                     nn.Conv2d(in_filters, out_filters, **opts_conv),
                     #nn.ConvTranspose2d(in_filters, out_filters, stride=opt.stride, **opts_conv),
                     ]
            if bn and (not opt.bn_eps == np.inf):
                block.append(nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum))
            block.append(NL)
            return block

        self.init_size = opt.img_size // opt.stride**3
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, self.channels[3] * self.init_size ** 2), NL)

        self.conv1 = nn.Sequential(*generator_block(self.channels[3], self.channels[2], bn=False),)
        self.conv2 = nn.Sequential(*generator_block(self.channels[2], self.channels[1]),)
        self.conv3 = nn.Sequential(*generator_block(self.channels[1], self.channels[0]),)

        self.conv_blocks = nn.Conv2d(self.channels[0], opt.channels, kernel_size=3, stride=1, padding=1)
        # self.conv_blocks = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=opt.stride), #opts_conv['stride']),
        #     nn.Conv2d(self.channels[0], opt.channels, kernel_size=3, stride=1, padding=1) #, bias=opts_conv['bias']),
        #     # nn.ConvTranspose2d(self.channels[0], opt.channels, kernel_size=3, stride=opt.stride, padding=2, bias=opt.do_bias),
        #     # nn.Tanh(),
        #     # nn.Sigmoid()
        #     # nn.ReLU(inplace=True)
        # )
        self.out_NL = nn.Tanh()


        self.opt = opt

    def forward(self, z):
        if self.opt.verbose:
            print("Generator")
        # Dim : opt.latent_dim
        out = self.l1(z)
        if self.opt.verbose:
            print("l1 out : ", out.shape)
        out = out.view(out.shape[0], self.channels[3], self.init_size, self.init_size)
        # Dim : (self.channels[3], opt.img_size/8, opt.img_size/8)
        if self.opt.verbose:
            print("View out : ", out.shape)

        out = self.conv1(out)
        # Dim : (self.channels[3]/2, opt.img_size/4, opt.img_size/4)
        if self.opt.verbose:
            print("Conv1 out : ", out.shape)
        out = self.conv2(out)
        # Dim : (self.channels[3]/4, opt.img_size/2, opt.img_size/2)
        if self.opt.verbose:
            print("Conv2 out : ", out.shape)

        out = self.conv3(out)
        # Dim : (self.channels[3]/8, opt.img_size, opt.img_size)
        if self.opt.verbose:
            print("Conv3 out : ", out.shape)

        out = self.conv_blocks(out)
        # out = self.out_NL(out)
        # Dim : (opt.chanels, opt.img_size, opt.img_size)
        if self.opt.verbose:
            print("img out : ", out.shape)

        # if self.opt.do_whitening:
        #     out = conv2d(out, Kinv, padding=1)
        # return F.tanh(out)
        return out


    def _name(self):
        return "Generator"


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        NL = nn.LeakyReLU(opt.lrelu, inplace=True)
        opts_conv = dict(kernel_size=opt.kernel_size, stride=opt.stride,
                         padding=opt.padding, padding_mode='zeros',
                         bias=opt.do_bias)
        self.channels = [opt.channel0, opt.channel1, opt.channel2, opt.channel3]

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, **opts_conv), ]  # , nn.Dropout2d(0.25)
            if bn and (not opt.bn_eps == np.inf):
                # https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d
                block.append(nn.BatchNorm2d(out_filters, eps=opt.bn_eps, momentum=opt.bn_momentum))
            block.append(NL)
            return block

        self.opt = opt

        self.conv1 = nn.Sequential(*discriminator_block(opt.channels, self.channels[0], bn=False),)
        self.conv2 = nn.Sequential(*discriminator_block(self.channels[0], self.channels[1]),)
        self.conv3 = nn.Sequential(*discriminator_block(self.channels[1], self.channels[2]),)
        self.conv4 = nn.Sequential(*discriminator_block(self.channels[2], self.channels[3]),)

        # The height and width of downsampled image
        self.init_size = opt.img_size // opt.stride**4
        # self.adv_layer = nn.Sequential(nn.Linear(self.channels[3] * self.init_size ** 2, 1))#, nn.Sigmoid()
        self.adv_layer = nn.Linear(self.channels[3] * self.init_size ** 2, 1)

    def forward(self, img):
        if self.opt.verbose:
            print("D")
            print("Image shape : ", img.shape)
            # Dim : (opt.chanels, opt.img_size, opt.img_size)

        out = img
        if self.opt.D_noise > 0:
            n = self.opt.D_noise * torch.randn(img.shape)
            if cuda:
                n = n.to('cuda')
            out += n
        # if self.opt.do_whitening:
        #     out = conv2d(out, KW, padding=1)

        out = self.conv1(out)
        if self.opt.verbose:
            print("Conv1 out : ", out.shape)

        out = self.conv2(out)
        if self.opt.verbose:
            print("Conv2 out : ", out.shape)

        out = self.conv3(out)
        if self.opt.verbose:
            print("Conv3 out : ", out.shape)

        out = self.conv4(out)
        if self.opt.verbose:
            print("Conv4 out : ", out.shape)

        out = out.view(out.shape[0], -1)
        if self.opt.verbose:
            print("View out : ", out.shape)

        validity = self.adv_layer(out)
        if self.opt.verbose:
            print("Val out : ", validity.shape)
            # Dim : (1)

        return validity

    def _name(self):
        return "Discriminator"
