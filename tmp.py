import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, qdim, kdim, vdim):
        super(MultiHeadAttention, self).__init__()

        self.scale = embed_dim ** -0.5
        self.to_q = nn.Linear(qdim, embed_dim, bias=False)
        self.to_k = nn.Linear(kdim, embed_dim, bias=False)
        self.to_v = nn.Linear(vdim, embed_dim, bias=False)

    def forward(self, q, k, v):
        b, n, dim = q.size()
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.bmm(attn, v)
        return out, (attn.detach(),)
