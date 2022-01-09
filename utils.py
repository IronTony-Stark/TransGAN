import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def up_sampling(x, scale_factor=2, mode="pixel_shuffle"):
    if mode == "pixel_shuffle":
        x = nn.PixelShuffle(scale_factor)(x)
    else:
        x = nn.Upsample(scale_factor=scale_factor, mode=mode)(x)
    return x


def up_sampling_permute(x, H, W, scale_factor=2, mode="pixel_shuffle"):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    if mode == "pixel_shuffle":
        x = nn.PixelShuffle(scale_factor)(x)
    else:
        x = nn.Upsample(scale_factor, mode)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def inits_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Modules with Equalized Learning Rate do not have 'weight' attribute and
        # are already initialed using normal distribution
        if not hasattr(m, "weight"):
            return

        nn.init.xavier_uniform_(m.weight.data, 1.)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias.data, 0)
        nn.init.constant_(m.weight.data, 1.0)


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def gen_noise(batch_size: int, dim: int):
    return torch.FloatTensor(np.random.normal(0, 1, (batch_size, dim)))


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):
        assert start_lr > end_lr
        self.optimizer = optimizer
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


class NormalNoiseDecay:
    def __init__(
            self,
            mean_start: float = 0, mean_end: float = 0, mean_start_step: int = 0, mean_end_step: int = 0,
            std_start: float = 1, std_end: float = 0, std_start_step: int = 0, std_end_step: int = 0,
    ):
        super().__init__()
        assert std_end >= 0

        self.mean_start = mean_start
        self.mean_end = mean_end
        self.mean_start_step = mean_start_step
        self.mean_end_step = mean_end_step
        self.mean_delta = (mean_start - mean_end) / (mean_end_step - mean_start_step)

        self.std_start = std_start
        self.std_stop = std_end
        self.std_start_step = std_start_step
        self.std_end_step = std_end_step
        self.std_delta = (std_start - std_end) / (std_end_step - std_start_step)

        self.current_step = 0

    def __call__(self, size: Tuple[int, int]):
        return torch.randn(size).normal_(mean=self.get_current_mean(), std=self.get_current_std())

    def step(self):
        self.current_step += 1

    def get_current_mean(self) -> float:
        if self.current_step <= self.mean_start_step:
            mean = self.mean_start
        elif self.current_step >= self.mean_end_step:
            mean = self.mean_end
        else:
            mean = self.mean_start - self.mean_delta * (self.current_step - self.mean_start_step)
        return mean

    def get_current_std(self) -> float:
        if self.current_step <= self.std_start_step:
            std = self.std_start
        elif self.current_step >= self.std_end_step:
            std = self.std_stop
        else:
            std = self.std_start - self.std_delta * (self.current_step - self.std_start_step)
        return std


class Checkpoint:
    def __init__(self, checkpoint_folder: str, generator, discriminator, optimizer_gen, optimizer_dis):
        self.checkpoint_dir = checkpoint_folder
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis

    def save(self, filename: str, score: float, epoch: int):
        torch.save({
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_generator_state_dict": self.optimizer_gen.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_dis.state_dict(),
            "score": score,
            "epoch": epoch,
        }, os.path.join(self.checkpoint_dir, filename))

    def load(self, filename: str):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_gen.load_state_dict(checkpoint["optimizer_generator_state_dict"])
        self.optimizer_dis.load_state_dict(checkpoint["optimizer_discriminator_state_dict"])
        return checkpoint["score"], checkpoint["epoch"]


# NOTE: Remember to set num_workers=0 to avoid copying models across multiprocess
class GeneratorDataset(Dataset):
    def __init__(self, G, latent_dim: int, length: int, device: torch.device = torch.device("cuda")):
        self.G = G
        self.latent_dim = latent_dim
        self.length = length
        self.device = device

    def __len__(self):
        return self.length

    @torch.no_grad()
    def __getitem__(self, index):
        return self.G(gen_noise(1, self.latent_dim).to(self.device))[0]


# Pixelwise feature vector normalization introduced in Progressive GAN https://arxiv.org/abs/1710.10196
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Normalization(nn.Module):
    def __init__(self, type: str = "", dim: int = None):
        super().__init__()
        if type == "LN":
            self.norm = nn.LayerNorm(dim)
        elif type == "BN":
            self.norm = nn.BatchNorm1d(dim)
        elif type == "IN":
            self.norm = nn.InstanceNorm1d(dim)
        elif type == "PN":
            self.norm = PixelNorm()

    def forward(self, x):
        if hasattr(self, "norm"):
            return self.norm(x)
        return x


class NoiseInjection(nn.Module):
    def __init__(self, mean: int = 0, std: int = 1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_(mean=self.mean, std=self.std)

        return image + self.weight * noise
