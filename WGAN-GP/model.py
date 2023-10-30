"""
-*- encoding: utf-8 -*-

@ File: model.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is a sample Python script.
"""
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x nz x 1 x 1
            self._block(nz, ngf * 16, 4, 1, 0),  # N x ngf*16 x 4 x 4
            self._block(ngf * 16, ngf * 8, 4, 2, 1),  # N x ngf*8 x 8 x 8
            self._block(ngf * 8, ngf * 4, 4, 2, 1),  # N x ngf*4 x 16 x 16
            self._block(ngf * 4, ngf * 2, 4, 2, 1),  # N x ngf*2 x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, kernel_size=4, stride=2, padding=1),  # N x 3 x 64 x 64
            nn.Tanh()  # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.gen(x)


class Critic(nn.Module):
    def __init__(self, nc, ndf):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # Initial layer
            # Input: N x 1 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1),  # N x ndf x 32 x 32
            nn.LeakyReLU(0.2),

            self._block(ndf, ndf * 2, 4, 2, 1),  # N x ndf*2 x 16 x 16
            self._block(ndf * 2, ndf * 4, 4, 2, 1),  # N x ndf*4 x 8 x 8
            self._block(ndf * 4, ndf * 8, 4, 2, 1),  # N x ndf*8 x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 2, 0)  # N x 1 x 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, mean=0.0, std=0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(model.bias.data, val=0)
