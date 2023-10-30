"""
-*- encoding: utf-8 -*-

@ File: debug.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is a sample Python script.
"""
import numpy as np
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, in_features, out_features, hidden):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        x = x.cpu().detach().numpy()
        x = func(x)
        x = torch.from_numpy(x.astype(np.float32)).to(device).requires_grad_()
        return self.net(x)


def func(x):
    x = x**2 / 5 + 2
    x = np.array(x)
    return x


x = np.arange(0., 1., 0.001)
y = np.sin(2 * np.pi * x)
y = func(y)
y = torch.from_numpy(y.astype(np.float32)).to(device).requires_grad_()

y2 = torch.randn(len(y)).to(device)

model = Net(1000, 1000, 1000).to(device)
criterion = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    pred = model(y2)
    # if pred.is_cuda:
    #     pred = pred.cpu()
    # pred = pred.detach().numpy()
    # pred = pred.detach()
    # pred = func(pred)
    # pred = torch.from_numpy(pred.astype(np.float32))
    # if not pred.is_cuda:
    #     pred = pred.cuda()

    loss = criterion(pred, y)
    model.zero_grad()
    loss.backward()
    optim.step()

    print(f"[EPOCH] {epoch}/1000, Loss: {loss}")
