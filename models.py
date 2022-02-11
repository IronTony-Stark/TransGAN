import typing
import random

import torch
import torch.nn as nn

from configs import GenConfig, TransConfig
from diff_aug import DiffAugment
from tmp import *
from utils import up_sampling_permute, Normalization, up_sampling, PixelNorm
from equalized_lr import EqLinear, EqConv2d


class ConstantInput(nn.Module):
    def __init__(self, shape: typing.Tuple):
        super().__init__()

        self.shape = shape
        self.input = nn.Parameter(torch.randn(1, *shape))

    def forward(self, batch_size: int):
        return self.input.repeat(batch_size, *(len(self.shape) * [1]))


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
        super().__init__()

        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = EqLinear(in_feat, hid_feat)
        self.act = nn.LeakyReLU(negative_slope=0.2)
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


class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, style_num=16, mlp_layers_num=8):
        super().__init__()

        layers = [PixelNorm()]
        for _ in range(mlp_layers_num):
            layers.append(EqLinear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        layers.append(EqLinear(style_dim, style_num * style_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.style_dim = style_dim
        self.style_num = style_num
        self.style = nn.Sequential(*layers)

    def forward(self, z):
        return self.style(z).view(-1, self.style_num, self.style_dim)


# class StyleModulation(nn.Module):
#     def __init__(self, size: int, content_dim: int, style_num: int, style_dim: int, patch_size: int):
#         super().__init__()
#
#         self.size, self.content_dim, self.style_num, self.style_dim, self.patch_size \
#             = size, content_dim, style_num, style_dim, patch_size
#
#         self.norm = Normalization("CLN")  # Custom Layer Norm
#         self.keys = nn.Parameter(nn.init.orthogonal_(torch.empty(1, style_num, content_dim)))
#         self.pos = nn.Parameter(torch.zeros(1, (size // patch_size) ** 2, content_dim))
#         self.attention = MultiHeadAttention(content_dim, content_dim, content_dim, style_dim)
#
#     def forward(self, input, style):
#         batch_size, content_num, _ = input.size()
#
#         # remove old style
#         input = self.norm(input)
#
#         # calculate new style
#         query = torch.mean(input.view(batch_size, content_num, -1, self.patch_size, self.patch_size), dim=[3, 4])
#         keys = self.keys.repeat(input.size(0), 1, 1)
#         new_style, _ = self.attention(q=query + self.pos, k=keys, v=style)
#
#         # add new style
#         return input * new_style


class ToRGB(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()

        self.in_channel = in_channel
        self.conv = EqConv2d(in_channel, 3, 1)
        self.act = nn.Tanh()

    def forward(self, input: torch.Tensor, H: int, W: int, skip: torch.Tensor = None):
        input = input.permute(0, 2, 1).view(-1, self.in_channel, H, W)

        out = self.conv(input)

        if skip is not None:
            out += up_sampling(skip, mode="bilinear")

        # return self.act(out)
        return out


class Generator(nn.Module):
    def __init__(self, img_size=32, style_num=32, style_dim=512, mlp_layers_num=8, trans_configs=None):
        super().__init__()

        self.img_size, self.style_num, self.style_dim, self.mlp_layers_num \
            = img_size, style_num, style_dim, mlp_layers_num

        if trans_configs is None:
            trans_configs = [
                TransConfig(depth=1, heads=1),
                TransConfig(depth=1, heads=1),
                TransConfig(depth=1, heads=1),
                TransConfig(depth=1, heads=1),
            ]

        self.configs = [
            GenConfig(8, 512, style_num, 1),
            GenConfig(16, 512, style_num, 1),
            GenConfig(32, 512, style_num, 1),
            GenConfig(64, 256, style_num, 1),
        ]

        self.num_blocks = next(i for i, config in enumerate(self.configs) if config.size == img_size) + 1

        self.mapping_network = MappingNetwork(style_dim=style_dim, style_num=style_num, mlp_layers_num=mlp_layers_num)

        self.constant_input = ConstantInput((self.configs[0].size ** 2, self.configs[0].content_dim))

        self.style_modulations = nn.ModuleList()
        # self.positional_embeddings = nn.ParameterList()
        # self.transformer_encoders = nn.ModuleList()
        self.to_RGBs = nn.ModuleList()
        for i in range(self.num_blocks):
            size, content_dim, style_num, patch_size = self.configs[i].get()
            depth, heads, mlp_ratio, drop_rate, norm_type = trans_configs[i].get()

            self.style_modulations.append(StyleModulation(size, content_dim, style_num, style_dim, patch_size))
            # self.positional_embeddings.append(nn.Parameter(torch.zeros(1, size ** 2, content_dim)))
            # self.transformer_encoders.append(TransformerEncoder(
            #     depth, content_dim, heads, mlp_ratio, drop_rate, norm_type
            # ))
            self.to_RGBs.append(ToRGB(content_dim))

    def forward(self, input):
        assert len(input) == 1 or len(input) == 2

        input = [self.mapping_network(i) for i in input]

        # todo truncation

        # todo noise mixing
        styles = input[0].unsqueeze(0).repeat(self.num_blocks, 1, 1, 1)

        x = self.constant_input(styles.size(1))

        i = 0
        skip = None
        # for style_modulation, positional_embedding, transformer_encoder, to_rgb in zip(
        #         self.style_modulations, self.positional_embeddings,
        #         self.transformer_encoders, self.to_RGBs
        # ):
        for style_modulation, to_rgb in zip(
                self.style_modulations, self.to_RGBs
        ):
            size = self.configs[i].size

            if i != 0:
                current_size = self.configs[i - 1].size
                x, _, _ = up_sampling_permute(x, current_size, current_size, mode="bilinear")

            x, _ = style_modulation(x, styles[i])
            # x = x + positional_embedding
            # x = transformer_encoder(x)
            # todo add norm here?
            skip = to_rgb(x, size, size, skip)

            i += 1

        return skip


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
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = DiffAugment(x, self.diff_aug)
        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])

        return x
