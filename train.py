import argparse

import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=32, help="Size of image for discriminator input.")
parser.add_argument("--epoch", type=int, default=200, help="Number of epoch.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--lr_gen", type=float, default=0.0001, help="Learning rate for generator.")
parser.add_argument("--lr_dis", type=float, default=0.0001, help="Learning rate for discriminator.")
parser.add_argument("--lr_decay_start_epoch", type=int, default=50,
                    help="Epoch number to start linear decay of learning rate")
parser.add_argument("--beta1", type=float, default=0.0, help="beta1")
parser.add_argument("--beta2", type=float, default=0.99, help="beta2")
parser.add_argument("--phi", type=int, default=1, help="phi")
parser.add_argument("--patch_size", type=int, default=4, help="Patch size for discriminator.")
parser.add_argument("--initial_size", type=int, default=8, help="Initial size for generator.")
parser.add_argument("--mixing_prob", type=float, default=0.9, help="Probability of latent mixing.")
parser.add_argument("--latent_dim", type=int, default=1024, help="Latent dimension of generator's input.")
parser.add_argument("--diff_aug", type=str, default="color,translation,cutout", help="Data Augmentation")
parser.add_argument("--log_dir", type=str, default=None, help="Tensorboard log directory")
parser.add_argument("--weight_log_iter", type=int, default=100,
                    help="Log weights and gradients every <weight_log_iter> iterations (batches)")
parser.add_argument("--n_critic", type=int, default=1,
                    help="Allows to train discriminator more than generator. "
                         "Specifically discriminator will be updated every "
                         "iteration while generator - every <n_critic> iterations")

group_head_start = parser.add_argument_group(
    "Head start: allow generator/discriminator to train for several "
    "iterations before discriminator/generator will begin to train"
)
parser.add_argument("--dis_head_start", type=int, default=-1,
                    help="Allows discriminator to train for <dis_head_start> "
                         "iterations before generator will begin its training")
parser.add_argument("--gen_head_start", type=int, default=-1,
                    help="Allows generator to train for <gen_head_start> "
                         "iterations before discriminator will begin its training")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(style_dim=args.latent_dim)
discriminator = Discriminator(
    diff_aug=args.diff_aug, image_size=32, patch_size=args.patch_size, input_channel=3,
    num_classes=1, dim=384, depth=7, heads=4,
    mlp_ratio=4, drop_rate=0.7
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

lr_decay_gen = LinearLrDecay(optimizer_gen, args.lr_gen, 0.0, args.lr_decay_start_epoch, args.epoch)
lr_decay_dis = LinearLrDecay(optimizer_dis, args.lr_dis, 0.0, args.lr_decay_start_epoch, args.epoch)

normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
    transforms.Resize(size=(args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalization,
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

writer = SummaryWriter(log_dir=args.log_dir)
checkpoint = Checkpoint("./checkpoints/", generator, discriminator, optimizer_gen, optimizer_dis)

iteration = 0
for epoch in range(args.epoch):

    # Learning Rate Decay
    lr_gen = lr_decay_gen.step(epoch)
    lr_dis = lr_decay_dis.step(epoch)

    writer.add_scalar("Generator/LR", lr_gen, epoch)
    writer.add_scalar("Discriminator/LR", lr_dis, epoch)

    for index, (batch_imgs, _) in enumerate(train_loader):
        loss_dis = None
        loss_gen = None

        noise = gen_mixing_noise(batch_imgs.shape[0], args.latent_dim, args.mixing_prob, device)
        real_imgs = batch_imgs.to(device)
        fake_imgs = normalization(pixel_normalization(generator(noise)))

        # Update Discriminator
        if iteration > args.gen_head_start:
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            optimizer_dis.zero_grad()

            real_score = discriminator(real_imgs)
            fake_score = discriminator(fake_imgs.detach())

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), args.phi)
            loss_dis = -torch.mean(real_score) + torch.mean(fake_score) + gradient_penalty * 10 / (args.phi ** 2)

            loss_dis.backward()

            # optimizer_dis.step()

            writer.add_scalar("Discriminator/Loss", loss_dis.item(), iteration)
            writer.add_scalar("Discriminator/Real Score", -torch.mean(real_score).item(), iteration)
            writer.add_scalar("Discriminator/Fake Score", torch.mean(fake_score).item(), iteration)

        # Update Generator
        if iteration % args.n_critic == 0 and iteration > args.dis_head_start:
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            optimizer_gen.zero_grad()

            fake_score = discriminator(fake_imgs)

            loss_gen = -torch.mean(fake_score)

            loss_gen.backward()

            optimizer_gen.step()

            writer.add_scalar("Generator/Loss", loss_gen.item(), iteration)

        # Logging
        if iteration % 100 == 0:
            print(
                f"[Epoch {epoch}/{args.epoch}] "
                f"[Batch {index % len(train_loader)}/{len(train_loader)}] "
                f"[D loss: {round(loss_dis.item(), 4) if loss_dis else '?'}] "
                f"[G loss: {round(loss_gen.item(), 4) if loss_gen else '?'}] "
            )
        if iteration % 800 == 0:
            writer.add_image(
                "Generated Images",
                vutils.make_grid(fake_imgs, padding=2, normalize=True, scale_each=True),
                iteration
            )
        if iteration % args.weight_log_iter == 0 and iteration != 0:
            for name, weight in generator.named_parameters():
                if weight is not None:
                    writer.add_histogram(f"Generator/{name}", weight, iteration)
                if weight.grad is not None:
                    writer.add_histogram(f"Generator/{name}.grad", weight.grad, iteration)
            for name, weight in discriminator.named_parameters():
                if weight is not None:
                    writer.add_histogram(f"Discriminator/{name}", weight, iteration)
                if weight.grad is not None:
                    writer.add_histogram(f"Discriminator/{name}.grad", weight.grad, iteration)

        iteration += 1

    # Checkpoint
    # noinspection PyUnboundLocalVariable
    # checkpoint.save(f"{epoch}.pth", loss_gen.item(), epoch)

# Checkpoint
# checkpoint.save(f"final.pth", loss_gen.item(), args.epoch)
