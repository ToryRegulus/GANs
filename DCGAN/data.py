"""
-*- encoding: utf-8 -*-

@ File: data.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is a sample Python script.
"""
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def load_data(img_root, img_size=64, batch_size=128):

    dataset = datasets.ImageFolder(root=img_root, transform=transforms.Compose([
        transforms.Resize(img_size),  # scale the shorter side ti img_size while maintaining the aspect ratio
        transforms.CenterCrop(img_size),  # crop the img from the center to (img_size x img_size)
        transforms.ToTensor(),  # normalize the data to the (0, 1)
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalize the data to the (-1 , 1)
    ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader
