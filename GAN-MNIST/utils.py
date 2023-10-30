"""
-*- coding: utf-8 -*-

@ File: utils.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This file contains some useful tools.
"""
import torch


def device():
    """
    Set device.
    :return: GPU/CPU
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev
