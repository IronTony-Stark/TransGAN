class GenConfig:
    def __init__(self, size: int, content_dim: int, style_num: int, patch_size: int):
        self.size = size
        self.content_dim = content_dim
        self.style_num = style_num  # style_num 16/32/64/128
        self.patch_size = patch_size

    def get(self):
        return self.size, self.content_dim, self.style_num, self.patch_size


class TransConfig:
    def __init__(self, depth: int, heads: int = 4, mlp_ratio: int = 4,
                 drop_rate: float = 0., norm_type: str = "LN"):
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.norm_type = norm_type

    def get(self):
        return self.depth, self.heads, self.mlp_ratio, self.drop_rate, self.norm_type
