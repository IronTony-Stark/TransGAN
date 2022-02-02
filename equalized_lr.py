# Equalized Learning Rate
# Introduced in Progressive GAN https://arxiv.org/abs/1710.10196
# Implementation taken from https://personal-record.onrender.com/post/equalized-lr/

import math

import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                )
                * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class EqConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

# class EqualLR:
#     def __init__(self, name: str):
#         self.name = name
#
#     def compute_weight(self, module: Module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()
#
#         return weight * sqrt(2 / fan_in)
#
#     @staticmethod
#     def apply(module: Module, name: str):
#         fn = EqualLR(name)
#
#         weight = getattr(module, name)
#         # noinspection PyProtectedMember
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         module.register_forward_pre_hook(fn)
#
#         return fn
#
#     def __call__(self, module: Module, input: Tensor):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)
#
#
# class EqLinear(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#
#         linear = nn.Linear(*args, **kwargs)
#         nn.init.normal_(linear.weight.data)
#         if linear.bias is not None:
#             nn.init.constant_(linear.bias.data, 0)
#         EqualLR.apply(linear, "weight")
#
#         self.linear = linear
#
#     def forward(self, input: Tensor) -> Tensor:
#         return self.linear(input)
#
#
# class EqConv2d(Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#
#         conv = nn.Conv2d(*args, **kwargs)
#         nn.init.normal_(conv.weight.data)
#         if conv.bias is not None:
#             nn.init.constant_(conv.bias.data, 0)
#         EqualLR.apply(conv, "weight")
#
#         self.conv = conv
#
#     def forward(self, input: Tensor) -> Tensor:
#         return self.conv(input)
