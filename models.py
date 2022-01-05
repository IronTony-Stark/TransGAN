import torch
import torch.nn as nn

from diff_aug import DiffAugment
from utils import up_sampling, Normalization
from equalized_lr import *


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = EqLinear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = EqLinear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.qkv = EqLinear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            EqLinear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)

        return self.out(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0., normalization_type="LN"):
        super().__init__()
        self.norm1 = Normalization(normalization_type, dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.norm2 = Normalization(normalization_type, dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1)
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0., norm_type="LN"):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(dim, heads, mlp_ratio, drop_rate, norm_type)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = EqConv2d(
            input_channel, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


class Generator(nn.Module):
    def __init__(self, depth1=5, depth2=4, depth3=2, latent_dim=1024, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                 drop_rate=0., norm_type="LN"):
        super().__init__()

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate

        self.mlp = EqLinear(latent_dim, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8 ** 2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8 * 2) ** 2, 384 // 4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8 * 4) ** 2, 384 // 16))

        self.transformer_encoder_1 = TransformerEncoder(
            depth=self.depth1, dim=self.dim, heads=self.heads,
            mlp_ratio=self.mlp_ratio, drop_rate=self.drop_rate,
            norm_type=norm_type
        )
        self.transformer_encoder_2 = TransformerEncoder(
            depth=self.depth2, dim=self.dim // 4, heads=self.heads,
            mlp_ratio=self.mlp_ratio, drop_rate=self.drop_rate,
            norm_type=norm_type
        )
        self.transformer_encoder_3 = TransformerEncoder(
            depth=self.depth3, dim=self.dim // 16, heads=self.heads,
            mlp_ratio=self.mlp_ratio, drop_rate=self.drop_rate,
            norm_type=norm_type
        )

        self.out = EqConv2d(self.dim // 16, 3, 1, 1, 0)

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        H, W = self.initial_size, self.initial_size
        x = x + self.positional_embedding_1
        x = self.transformer_encoder_1(x)

        x, H, W = up_sampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.transformer_encoder_2(x)

        x, H, W = up_sampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.transformer_encoder_3(x)

        x = self.out(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))

        return x


class Discriminator(nn.Module):
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channel=3, num_classes=1,
                 dim=384, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0., norm_type="LN"):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.dropout = nn.Dropout(p=drop_rate)
        self.transformer_encoder = TransformerEncoder(
            depth, dim, heads,
            mlp_ratio, drop_rate,
            norm_type=norm_type
        )
        self.norm = Normalization(norm_type, dim)
        self.out = EqLinear(dim, num_classes)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])

        return x
