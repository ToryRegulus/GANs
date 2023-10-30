"""
-*- coding: utf-8 -*-

@ File: model.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This file contains discriminator and generator.
"""
from torch import nn
from utils import device


def D(image_size, hidden_size):
    """
    Make the discriminator.
    :param image_size:
    :param hidden_size:
    :return: Discriminator
    """
    D = nn.Sequential(
        nn.Linear(image_size, hidden_size),
        nn.LeakyReLU(0.2),  # Use 0.2 degree as negative slope
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()  # out with 0 or 1
    )

    D = D.to(device())

    return D


def G(latent_size, image_size, hidden_size):
    """
    Make generator.
    :param latent_size: 噪声 z 的样本空间
    :param image_size:
    :param hidden_size:
    :return: Generator
    """
    G = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh()  # Use Tanh to let data distribute within [-1, 1]
    )

    G.to(device())

    return G
