"""
-*- encoding: utf-8 -*-

@ File: train.py
@ Author: ToryRegulus(絵守辛玥)
@ Desc: This is a sample Python script.
"""
import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import Generator, Critic, weights_init

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
CRITIC_FEATURES = 64
GEN_FEATURES = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

# Dataset Preparation
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5 for _ in range(CHANNELS_IMG)],
        std=[0.5 for _ in range(CHANNELS_IMG)]
    )
])

dataset = datasets.MNIST(
    root='../dataset/', train=True, transform=transforms, download=True
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# Model Instantiation
gen = Generator(Z_DIM, CHANNELS_IMG, GEN_FEATURES).to(device)
c = Critic(CHANNELS_IMG, CRITIC_FEATURES).to(device)
weights_init(gen)
weights_init(c)

# Loss Function
optim_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
optim_c = optim.RMSprop(c.parameters(), lr=LEARNING_RATE)

# Summary Writer
writer_real = SummaryWriter(f"log/real")
writer_fake = SummaryWriter(f"log/fake")
step = 0

# Train Mode
gen.train()
c.train()

for epoch in range(NUM_EPOCHS):
    for batch, (real_img, _) in enumerate(dataloader):
        real_img = real_img.to(device)

        for _ in range(CRITIC_ITERATIONS):
            z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            noise = gen(z)
            critic_real = c(real_img).view(-1)
            critic_fake = c(noise).view(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            c.zero_grad()
            loss_critic.backward(retain_graph=True)  # because params will be clipped
            optim_c.step()

            for p in c.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        fake_c = c(noise).view(-1)
        loss_gen = -torch.mean(fake_c)
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # ----- Print Output -----
        if batch % 100 == 0:
            print(
                f"EPOCH [{epoch} / {NUM_EPOCHS}]\tBATCH [{batch} / {len(dataloader)}]\t"
                f"Loss C: {loss_critic.item():.4f}\tLoss G: {loss_gen.item():.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real_img[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
