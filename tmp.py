# todo remove
import torch
import torch.nn as nn
from equalized_lr import EqLinear


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, qdim, kdim, vdim):
        super(MultiHeadAttention, self).__init__()

        self.scale = embed_dim ** -0.5
        self.to_q = EqLinear(qdim, embed_dim, bias=False)
        self.to_k = EqLinear(kdim, embed_dim, bias=False)
        self.to_v = EqLinear(vdim, embed_dim, bias=False)

    def forward(self, q, k, v):
        b, n, dim = q.size()
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.bmm(attn, v)
        return out, (attn.detach(),)


def norm(input, norm_type='layernorm'):
    # [b, hw, c]
    if norm_type == 'layernorm' or norm_type == 'l2norm':
        normdim = -1
    elif norm_type == 'insnorm':
        normdim = 1
    else:
        raise NotImplementedError('have not implemented this type of normalization')

    if norm_type != 'l2norm':
        mean = torch.mean(input, dim=normdim, keepdim=True)
        input = input - mean

    demod = torch.rsqrt(torch.sum(input ** 2, dim=normdim, keepdim=True) + 1e-8)
    return input * demod


class StyleModulation(nn.Module):
    def __init__(
            self,
            size: int, content_dim: int, style_num: int, style_dim: int, patch_size: int,
            style_mod='prod',
            norm_type='layernorm'
    ):
        super().__init__()

        self.style_mod = style_mod
        self.norm_type = norm_type
        self.patch_size = patch_size
        self.keys = nn.Parameter(nn.init.orthogonal_(torch.empty(1, style_num, content_dim)))
        self.pos = nn.Parameter(torch.zeros(1, (size // patch_size) ** 2, content_dim))
        self.attention = MultiHeadAttention(content_dim, content_dim, content_dim, style_dim)

    def forward(self, input, style, is_new_style=False):
        b, t, c = input.size()

        # remove old style
        input = norm(input)
        input = input.view(b, t, -1, self.patch_size, self.patch_size)

        # calculate new style
        if not is_new_style:
            # multi-head attention
            query = torch.mean(input, dim=[3, 4])
            keys = self.keys.repeat(input.size(0), 1, 1)
            pos = self.pos.repeat(input.size(0), 1, 1)
            new_style, _ = self.attention(q=query + pos, k=keys, v=style)
        else:
            new_style = style

        # append new style
        if self.style_mod == 'prod':
            out = input * new_style.unsqueeze(-1).unsqueeze(-1)
        elif self.style_mod == 'plus':
            out = input + new_style.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('Have not implemented this type of style modulation')

        out = out.view(b, t, c)
        return out, (new_style.detach(),)
