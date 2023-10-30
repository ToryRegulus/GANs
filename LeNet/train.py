"""
-*- coding: utf-8 -*-

@ File: train.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: 模型训练文件
"""
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data import import_data
from model import LeNet


def training():
    # 将训练迁移至GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 定义LeNet模型
    lenet = LeNet().to(device)  # 在GPU上训练

    # 导入数据
    train_data, _ = import_data()

    # 定义损失函数与优化器
    # 使用交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss().to(device)  # 在GPU上训练
    lr = 0.01  # 定义学习率
    # 使用随机梯度下降作为优化器，设定动量和权值衰减
    optimizer = optim.SGD(params=lenet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # ---------- 开始训练 ----------
    num_epoch = 10  # 设置迭代次数

    for epoch in range(num_epoch):
        # 定义损失值与准确度
        train_loss = 0
        train_acc = 0

        # 设置为训练模式
        lenet = lenet.train()

        for batch, (img, label) in enumerate(train_data):
            # 将张量迁移至GPU并用Variable使其可反向传播
            img_v = Variable(img.to(device))
            label_v = Variable(label.to(device))

            # 前向传播
            out = lenet(img_v)
            loss = criterion(out, label_v)

            # 反向传播
            optimizer.zero_grad()  # 梯度值归零，否者会累加
            loss.backward()
            optimizer.step()  # 更新参数

            # 记录数据
            train_loss += loss.item()  # 记录误差
            _, pred = out.max(1)  # out返回两个值，分别为value和value所在的index
            correct = pred.eq(label_v).sum().item()

            train_acc += correct  # 正确的样本个数

        # 样本总数/batchsize是走完一个epoch所需的”步数“
        # 相对应的len(train_data.dataset)也就是样本总数，len(train_loader)就是这个步数
        # 误差采用每个epoch的误差的和除以步数的大小算得平均误差
        # 精准度采用预测正确的样本除以总样本
        print(
            f'[INFO] Epoch {epoch + 1}/{num_epoch} | '
            f'Loss: {train_loss / len(train_data):.4f} - Acc: {train_acc / len(train_data.dataset):.2%}'
        )

    torch.save(lenet, f"checkpoints/lenet.pth")  # 保存模型


if __name__ == '__main__':
    training()
