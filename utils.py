import numpy as np
import torch
import torch.nn as nn
import os


def up_sampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def inits_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data, 1.)


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
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


class Checkpoint:
    def __init__(
            self,
            checkpoint_folder: str,
            discriminator, generator,
            optimizer_dis, optimizer_gen,
            max_checkpoints: int
    ):
        self.checkpoint_dir = checkpoint_folder
        self.discriminator = discriminator
        self.generator = generator
        self.optimizer_dis = optimizer_dis
        self.optimizer_gen = optimizer_gen
        self.best_score = 1.e4
        self.epoch = 0
        self.max_checkpoints = max_checkpoints

    def update(self, new_best_score, epoch):
        lowest_score = 1.e4
        num_checkpoints = os.listdir(self.checkpoint_dir)
        if num_checkpoints == self.max_checkpoints:
            lowest_score = self._find_lowest_score()
        if new_best_score < self.best_score:
            self.best_score = new_best_score
            self.epoch = epoch
            self.save(f"{self.best_score}.pth")

    def save(self, filename: str):
        torch.save({
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_dis.state_dict(),
            "optimizer_generator_state_dict": self.optimizer_gen.state_dict(),
            "score": self.best_score,
            "epoch": self.epoch,
        }, os.path.join(self.checkpoint_dir, filename))

    def load(self, filename: str):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizer_dis.load_state_dict(checkpoint["optimizer_discriminator_state_dict"])
        self.optimizer_gen.load_state_dict(checkpoint["optimizer_generator_state_dict"])
        self.best_score = checkpoint["score"]
        self.epoch = checkpoint["epoch"]

    def _find_lowest_score(self):
        scores = map(lambda filename: float(filename.split(".")[0]), os.listdir(self.checkpoint_dir))
        return min(scores)
