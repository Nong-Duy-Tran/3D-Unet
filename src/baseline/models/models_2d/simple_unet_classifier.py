import torch
import torch.nn as nn


class SliceAttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1)


class SimpleUNet2DClassifier(nn.Module):
    """Simplified 2D U-Net encoder + slice attention pooling for volume classification."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_features: int = 32,
        head_hidden: int = 256,
        dropout: float = 0.5,
        head_dropout: float = 0.3,
    ):
        super().__init__()

        self.enc1 = self._make_encoder(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._make_encoder(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._make_encoder(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self._make_encoder(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self._make_encoder(base_features * 8, base_features * 16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = SliceAttentionPool(base_features * 16)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_features * 16, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    @staticmethod
    def _make_encoder(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))
        pooled = self.global_pool(x5)
        return pooled.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C, H, W) or (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feats = self._encode_slices(x)
        feats = feats.view(b, s, -1)
        pooled = self.pool(feats)
        return self.classifier(pooled)


def default_config():
    return {
        "in_channels": 3,
        "num_classes": 2,
        "base_features": 32,
        "head_hidden": 256,
        "dropout": 0.5,
        "head_dropout": 0.3,
    }
