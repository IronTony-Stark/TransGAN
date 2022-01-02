# Mostly taken from https://github.com/asarigun/TransGAN

import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from utils import inits_weight, gen_noise

from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=32, help="Size of image for discriminator input.")
parser.add_argument("--epoch", type=int, default=200, help="Number of epoch.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--lr_gen", type=float, default=0.0001, help="Learning rate for generator.")
parser.add_argument("--lr_dis", type=float, default=0.0001, help="Learning rate for discriminator.")
parser.add_argument("--beta1", type=int, default="0", help="beta1")
parser.add_argument("--beta2", type=float, default="0.99", help="beta2")
parser.add_argument("--patch_size", type=int, default=4, help="Patch size for discriminator.")
parser.add_argument("--initial_size", type=int, default=8, help="Initial size for generator.")
parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension of generator's input.")
parser.add_argument("--diff_aug", type=str, default="color,translation,cutout", help="Data Augmentation")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(
    depth1=5, depth2=4, depth3=2,
    initial_size=args.initial_size, dim=384, heads=4,
    mlp_ratio=4, drop_rate=0.5
)
discriminator = Discriminator(
    diff_aug=args.diff_aug, image_size=32, patch_size=args.patch_size, input_channel=3,
    num_classes=1, dim=384, depth=7, heads=4,
    mlp_ratio=4, drop_rate=0.
)

generator.apply(inits_weight)
discriminator.apply(inits_weight)

generator.to(device)
discriminator.to(device)

generator.train()
discriminator.train()

optimizer_gen = optim.Adam(
    filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen,
    betas=(args.beta1, args.beta2)
)
optimizer_dis = optim.Adam(
    filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_dis,
    betas=(args.beta1, args.beta2)
)

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
    transforms.Resize(size=(args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

writer = SummaryWriter("./logs/")

iteration = 0
for epoch in range(args.epoch):
    for index, (batch_imgs, _) in enumerate(train_loader):
        noise = gen_noise(batch_imgs.shape[0], args.latent_dim).to(device)
        real_imgs = batch_imgs.to(device)
        fake_imgs = generator(noise)

        # Update Discriminator
        optimizer_dis.zero_grad()

        real_valid = discriminator(real_imgs)
        fake_valid = discriminator(fake_imgs.detach())

        loss_dis = -torch.mean(real_valid) + torch.mean(fake_valid)
        loss_dis.backward()
        optimizer_dis.step()

        writer.add_scalar("Discriminator/Loss", loss_dis.item(), iteration)

        # Update Generator
        optimizer_gen.zero_grad()

        fake_valid = discriminator(fake_imgs)

        loss_gen = -torch.mean(fake_valid).to(device)
        loss_gen.backward()
        optimizer_gen.step()

        writer.add_scalar("Generator/Loss", loss_gen.item(), iteration)

        # Logging
        if iteration % 100 == 0:
            print(
                f"[Epoch {epoch}/{args.epoch}] [Batch {index % len(train_loader)}/{len(train_loader)}] "
                f"[D loss: {round(loss_dis.item(), 4)}] [G loss: {round(loss_gen.item(), 4)}]"
            )
        if iteration % 500 == 0:
            writer.add_image(
                "Generated Images",
                vutils.make_grid(fake_imgs, padding=2, normalize=True),
                iteration
            )

        iteration += 1
