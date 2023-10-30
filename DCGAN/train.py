"""
-*- encoding: utf-8 -*-

@ File: train.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: Training of DCGAN network.
"""
from pathlib import Path

import torch
import torchvision.utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from data import load_data
from model import Generator, Discriminator, weights_init

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
GEN_FEATURES = 64
DIS_FEATURES = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5 for _ in range(CHANNELS_IMG)],
            std=[0.5 for _ in range(CHANNELS_IMG)],
            inplace=True
        ),  # range to [-1, 1]
    ]
)

# Dataset Preparation
root = "D:/data/celeba"
# dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root=root, transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Instantiation
gen = Generator(Z_DIM, CHANNELS_IMG, GEN_FEATURES).to(device)
disc = Discriminator(CHANNELS_IMG, DIS_FEATURES).to(device)
gen.apply(weights_init)
disc.apply(weights_init)

# Loss Function
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Summary Writer
writer_real = SummaryWriter(f"log/real")
writer_fake = SummaryWriter(f"log/fake")
step = 0

# Train Mode
gen.train()
disc.train()

# Start Training
for epoch in range(NUM_EPOCHS):
    for batch, (real_img, _) in enumerate(loader):
        real_img = real_img.to(device)
        z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        noise = gen(z)

        # ----- Train Discriminator -----
        # Real Image
        disc_real = disc(real_img).view(-1)  # reshape (N x 1 x 1 x 1) to (N)
        real_label = torch.ones_like(disc_real)
        disc_loss_real = criterion(disc_real, real_label)
        real_score = disc_loss_real

        # Fake Image
        disc_fake = disc(noise.detach()).view(-1)
        fake_label = torch.zeros_like(disc_fake)
        disc_loss_fake = criterion(disc_fake, fake_label)

        disc_loss = disc_loss_real + disc_loss_fake

        disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()

        # ----- Train Generator -----
        fake = disc(noise).view(-1)
        gen_loss = criterion(fake, torch.ones_like(fake))
        gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        # ----- Print Output -----
        if batch % 100 == 0:
            print(
                f"EPOCH [{epoch} / {NUM_EPOCHS}]\tBATCH [{batch} / {len(loader)}]\t"
                f"Loss D: {disc_loss.item():.4f}\tLoss G: {gen_loss.item():.4f}"
            )

            with torch.no_grad():
                img_grid_real = torchvision.utils.make_grid(real_img[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(noise[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
