"""
-*- coding: utf-8 -*-

@ File: inference.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: 模型测试部分代码
"""
import torch
from torch import nn
from torch.autograd import Variable
from data import import_data


def inference():
    # 将测试迁移至GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载保存的模型
    model_path = "checkpoints/lenet.pth"
    model = torch.load(model_path).to(device)
    model.eval()  # 切换到测试状态

    # 定义损失函数，测试不需要优化器
    criterion = nn.CrossEntropyLoss().to(device)

    # 加载数据集
    _, test_data = import_data()

    # 定义测试集的误差与准确度和样本数量
    eval_loss = 0
    eval_acc = 0
    total = 0

    with torch.no_grad():  # 关闭计算图
        for batch, (img, label) in enumerate(test_data):
            img_v = Variable(img.to(device))
            label_v = Variable(label.to(device))

            out = model(img_v)
            loss = criterion(out, label_v)
            eval_loss += loss.item()

            _, pred = out.max(1)
            correct = pred.eq(label_v).sum().item()
            eval_acc += correct
            total += label.size(0)  # 样本数量逐增

            print(
                f'[INFO] Batch {batch + 1}/{len(test_data)} | '
                f'Loss: {eval_loss / (batch + 1):.4f} - Acc: {eval_acc / total:.2%}'
            )


if __name__ == '__main__':
    inference()
