"""
-*- coding: utf-8 -*-

@ File: data.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: MNIST数据集读取文件
"""
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import mnist


def import_data():
    """
    读取MNIST数据
    :return: 一个tuple(train_data, test_data)，分别为训练数据与测试数据
    """
    train_set = mnist.MNIST('../dataset', download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet数据输入大小为32x32，而MNIST数据集大小为28x28
        transforms.ToTensor()
    ]))
    test_set = mnist.MNIST('./data', download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))

    # 训练集一个batch大小为256，且打乱顺序；测试集一个batch为1024，且不打乱顺序
    train_data = DataLoader(dataset=train_set, batch_size=256, shuffle=True)
    test_data = DataLoader(dataset=train_set, batch_size=1024, shuffle=False)

    return train_data, test_data


if __name__ == '__main__':
    # 下载数据集并调用TensorBoard查看数据集
    train, test = import_data()
    writer = SummaryWriter("logs")  # 创建一个SummaryWriter对象用于tensorboard显示数据
    step = 0  # 指定tensorboard的step

    for data in train:
        img, label = data
        writer.add_images("Training Data", img, step)
        step += 1
        # print(img.size(0))
        # break

    writer.close()
