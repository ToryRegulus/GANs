"""
-*- encoding: utf-8 -*-

@ File: data.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is the DataLoader file.
"""
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def data_loader(batch_size=100):
    """
    Import MNIST data with shuffle.
    :param batch_size: Default: 100
    :return: MNIST DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    mnist_set = torchvision.datasets.MNIST('../dataset', train=True, transform=transform, download=True)

    mnist_loader = DataLoader(dataset=mnist_set, batch_size=batch_size, shuffle=True)

    return mnist_loader
