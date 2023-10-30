"""
-*- encoding: utf-8 -*-

@ File: train.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is a sample Python script.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from data import data_loader
from model import Discriminator, Generator


def reset_grad(optimizer):
    optimizer.zero_grad()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = Discriminator().to(device)
    G = Generator().to(device)
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    total_step = len(data_loader())
    num_epochs = 200
    batch_size = 100
    sample_dir = 'sample'

    # Start Training
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader()):
            step = i + 1
            img = imgs.to(device)
            label = labels.to(device)

            # define the label wither is real
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.randint(0, 10, (batch_size, )).to(device)

            # ===== Training Discriminator =====
            # define the loss func of real img
            outputs = D(img, label)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs  # record the real score

            # define the loss func if fake img
            z = torch.randn(batch_size, 100).to(device)
            fake_img = G(z, fake_labels)
            outputs = D(fake_img, fake_labels)
            d_loss_fake = criterion(outputs, torch.zeros(batch_size).to(device))
            fake_score = outputs  # record the fake score
            d_loss = d_loss_fake + d_loss_real

            reset_grad(d_optimizer)
            d_loss.backward()
            d_optimizer.step()

            # ===== Training Generator =====
            z = torch.randn(batch_size, 100).to(device)
            gen_img = G(z, fake_labels)
            outputs = D(gen_img, fake_labels)
            g_loss = criterion(outputs, real_labels)

            reset_grad(g_optimizer)
            g_loss.backward()
            g_optimizer.step()

            if step % 200 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{step}/{total_step}], "
                    f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                    f"D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}"
                )

        # ===== save image =====
        # real image
        if epoch == 0:
            image = img.reshape(img.size(0), 1, 28, 28)
            save_image(image, Path(sample_dir) / "real_image.png")

        # fake image
        fake_image = gen_img.reshape(img.size(0), 1, 28, 28)
        save_image(fake_image, Path(sample_dir) / f"fake_image-{epoch + 1}.png")

    # save model
    torch.save(G.state_dict(), 'G.pth')
    torch.save(D.state_dict(), 'D.pth')
