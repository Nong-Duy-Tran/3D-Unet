import torch
import torch.nn as nn


def _window_partition_3d(x, window_size):
    # x: (B, D, H, W, C)
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d // window_size,
        window_size,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, window_size ** 3, c)


def _window_reverse_3d(windows, window_size, d, h, w):
    # windows: (num_windows*B, window_size^3, C)
    b = int(windows.shape[0] // (d * h * w / window_size / window_size / window_size))
    x = windows.view(
        b,
        d // window_size,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, d, h, w, -1)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        size = (2 * window_size - 1) ** 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )  # (3, ws, ws, ws)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 2] += window_size - 1
        relative_coords[:, :, 0] *= (2 * window_size - 1) ** 2
        relative_coords[:, :, 1] *= (2 * window_size - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        # x: (num_windows*B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_dim=256, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self._attn_mask_cache = {}

    def _get_attn_mask(self, d, h, w, device):
        key = (d, h, w, device.type)
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        cnt = 0
        ws = self.window_size
        ss = self.shift_size
        for d_slice in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for h_slice in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                for w_slice in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                    img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                    cnt += 1
        mask_windows = _window_partition_3d(img_mask, ws).view(-1, ws ** 3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        # x: (B, D, H, W, C)
        b, d, h, w, c = x.shape
        ws = self.window_size
        if d % ws != 0 or h % ws != 0 or w % ws != 0:
            raise ValueError("Input resolution must be divisible by window_size")

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = self._get_attn_mask(d, h, w, x.device)
        else:
            attn_mask = None

        x_windows = _window_partition_3d(x, ws)
        if attn_mask is not None and attn_mask.shape[0] != x_windows.shape[0]:
            repeat = x_windows.shape[0] // attn_mask.shape[0]
            attn_mask = attn_mask.repeat(repeat, 1, 1)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        x = _window_reverse_3d(attn_windows, ws, d, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Swin3DEncoder(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_channels=1,
        embed_dim=96,
        depth=4,
        num_heads=4,
        window_size=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        blocks = []
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock3D(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.patch_embed(x)
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2, 3))
        return x


class Swin3DClassifier(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_channels=1,
        num_classes=2,
        embed_dim=96,
        depth=4,
        num_heads=4,
        window_size=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Swin3DEncoder(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.classifier(feats)


def default_config():
    return {
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 2,
        "embed_dim": 96,
        "depth": 4,
        "num_heads": 4,
        "window_size": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
    }
