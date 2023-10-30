"""
-*- encoding: utf-8 -*-

@ File: model.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: Generator and Discriminator implementation of DCGAN.
"""
import torch
from torch import nn


# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of channels in the training images. For color images this is 3
# nc: 输入图像中的颜色通道数，对于彩色图像则是3
nc = 3

# Size of z latent vector (i.e. size of generator input)
# nz: 潜在向量长度
nz = 100

# Size of feature maps in generator
# ngf: 与生成器内部的特征图深度有关(number of generator feature map)
ngf = 64

# Size of feature maps in discriminator
# ndf: 通过判别器传播的特征图深度
ndf = 64


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


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Initial layer
            # Input: N x 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1),  # N x ndf x 32 x 32
            nn.LeakyReLU(0.2),

            self._block(ndf, ndf * 2, 4, 2, 1),  # N x ndf*2 x 16 x 16
            self._block(ndf * 2, ndf * 4, 4, 2, 1),  # N x ndf*4 x 8 x 8
            self._block(ndf * 4, ndf * 8, 4, 2, 1),  # N x ndf*8 x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 2, 0),  # N x 1 x 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
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


def test():
    N, H, W = 8, 64, 64
    x = torch.randn((N, nc, H, W))
    disc = Discriminator(nc, ndf)
    disc.apply(weights_init)
    assert disc(x).shape == (N, 1, 1, 1)

    z = torch.randn((N, nz, 1, 1))
    gen = Generator(nz, nc, ngf)
    gen.apply(weights_init)
    assert gen(z).shape == (N, nc, H, W)


test()
