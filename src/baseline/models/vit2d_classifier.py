import math
import torch
import torch.nn as nn


class ViT2DEncoder(nn.Module):
    """Minimal ViT encoder for 2D inputs."""

    def __init__(
        self,
        image_size=64,
        patch_size=8,
        in_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]


class SliceAttentionPool(nn.Module):
    """Attention pooling over slice embeddings."""

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        weights = self.attn(x)  # (B, S, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled


class ViT2DClassifier(nn.Module):
    """2D ViT slice encoder + attention pooling for volume classification."""

    def __init__(
        self,
        image_size=64,
        patch_size=8,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = ViT2DEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.pool = SliceAttentionPool(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C, H, W) or (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feats = self.encoder(x)  # (B*S, D)
        feats = feats.view(b, s, -1)
        pooled = self.pool(feats)
        logits = self.classifier(pooled)
        return logits


def default_config():
    return {
        "image_size": 64,
        "patch_size": 8,
        "in_channels": 1,
        "num_classes": 2,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "mlp_dim": 512,
        "dropout": 0.1,
    }
