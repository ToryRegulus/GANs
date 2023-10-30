"""
-*- coding: utf-8 -*-

@ File: model.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: LeNet模型架构文件
"""
import torch
from torch import nn


class LeNet(nn.Module):
    """
    LeNet模型
    """
    def __init__(self, in_channel=1):
        super(LeNet, self).__init__()
        # 输出特征是6通道，特征大小为30x30，卷积核为3x3
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=3)
        # 一个2x2的步长为2的池化层，输出特征大小为15x15
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 同样是一个3x3的卷积层，输出特征为16通道，特征大小为13x13
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # 同样是2x2的步长为2的池化层，输出的特征大小为6x6
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层做线性映射
        self.fc3 = nn.Linear(in_features=16*6*6, out_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        # 最后映射到10个分类
        self.fc5 = nn.Linear(in_features=84, out_features=10)

        # 使用线性整流函数作为激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # 将图像的4维张量转换为2维（将CHW三个维度合并到一个维度）
        # img = batch, channel, height, width
        x = x.view(x.size(0), -1)  # 等价于 x.view(x.shape[0], -1)

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        return x


if __name__ == '__main__':
    # 测试模型
    model = LeNet()
    img = torch.randn(256, 1, 32, 32)  # 按照输入图片的格式假定一个张量
    ret = model(img)
    print(ret.shape)
