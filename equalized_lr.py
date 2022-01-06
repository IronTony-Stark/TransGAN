# Equalized Learning Rate
# Introduced in Progressive GAN https://arxiv.org/abs/1710.10196
# Implementation taken from https://personal-record.onrender.com/post/equalized-lr/

from math import sqrt

from torch import Tensor
from torch import nn
from torch.nn import Module


class EqualLR:
    def __init__(self, name: str):
        self.name = name

    def compute_weight(self, module: Module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module: Module, name: str):
        fn = EqualLR(name)

        weight = getattr(module, name)
        # noinspection PyProtectedMember
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module: Module, input: Tensor):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class EqLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        linear = nn.Linear(*args, **kwargs)
        nn.init.xavier_uniform_(linear.weight.data)
        if linear.bias is not None:
            nn.init.constant_(linear.bias.data, 0)
        EqualLR.apply(linear, "weight")

        self.linear = linear

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


class EqConv2d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        nn.init.xavier_uniform_(conv.weight.data)
        if conv.bias is not None:
            nn.init.constant_(conv.bias.data, 0)
        EqualLR.apply(conv, "weight")

        self.conv = conv

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)
