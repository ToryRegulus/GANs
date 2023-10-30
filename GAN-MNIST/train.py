"""
-*- coding: utf-8 -*-

@ File: train.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: Contains training code.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from data import import_data
from model import D, G
from utils import device


def reset_grad(optimizer):
    optimizer.zero_grad()


# 规范化处理
# 在Image processing中以mean,std=0.5进行规范化,out=2*input-1
# 故input=(out+1)/2
def denorm(x):
    out = (x + 1) / 2
    # 将out张量每个元素的范围限制到区间 [min,max]
    return out.clamp(0, 1)


def main():
    total_step = len(import_data())
    num_epochs = 200
    batch_size = 100
    latent_size = 64
    image_size = 784
    hidden_size = 256
    sample_dir = 'samples'

    # define model
    D_M = D(image_size, hidden_size)
    G_M = G(latent_size, image_size, hidden_size)

    criterion = nn.BCELoss()  # Use binary cross entropy
    d_optimizer = torch.optim.Adam(D_M.parameters(), lr=2e-4)
    g_optimizer = torch.optim.Adam(G_M.parameters(), lr=2e-4)

    # Start training
    for epoch in range(num_epochs):
        for i, (img, _) in enumerate(import_data()):
            real_labels = torch.ones(batch_size, 1).to(device())
            fake_labels = torch.zeros(batch_size, 1).to(device())

            # Training discriminator
            # define the loss func of real_img
            img = img.reshape(batch_size, -1).to(device())
            outputs = D_M(img)  # input the real image
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # define the loss func of fake_img
            z = torch.randn(batch_size, latent_size).to(device())
            fake_img = G_M(z)
            outputs = D_M(fake_img)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # get the total loss
            d_loss = d_loss_real + d_loss_fake

            # reset grad
            reset_grad(d_optimizer)
            d_loss.backward()
            d_optimizer.step()

            # Training generator
            # define the loss func of fake img
            z = torch.randn(batch_size, latent_size).to(device())
            fake_img = G_M(z)
            outputs = D_M(fake_img)
            g_loss = criterion(outputs, real_labels)

            # reset grad
            reset_grad(g_optimizer)
            g_loss.backward()
            g_optimizer.step()

            # print the value
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], step: [{i + 1}/{total_step}], "
                    f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                    f"D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}",
                )

        # save the img
        # real image
        if (epoch + 1) == 1:
            images = img.reshape(img.size(0), 1, 28, 28)
            save_image(denorm(images), Path(sample_dir) / "real_image.png")

        # fake image
        fake_images = fake_img.reshape(img.size(0), 1, 28, 28)
        save_image(denorm(fake_images), Path(sample_dir) / f"fake_image-{epoch + 1}.png")

    # save model
    torch.save(G_M.state_dict(), 'G.pth')
    torch.save(D_M.state_dict(), 'D.pth')


if __name__ == '__main__':
    main()
