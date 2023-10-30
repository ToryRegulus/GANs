"""
-*- encoding: utf-8 -*-

@ File: model.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: The model definition.
"""
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)  # num: 0~9
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)  # reshape as (batch_size, 784)
        c = self.label_embed(labels)
        x = torch.cat([x, c], 1)  # cat with cols, shape as (batch_size, 794)
        out = self.model(x)

        return out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_embed(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)

        return out.view(x.size(0), 28, 28)
