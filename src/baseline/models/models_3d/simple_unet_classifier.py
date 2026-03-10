import torch
import torch.nn as nn


class SimpleUNet3DClassifier(nn.Module):
    """Simplified 3D U-Net for classification."""

    def __init__(self, in_channels=1, num_classes=2, base_features=32):
        super().__init__()

        self.enc1 = self._make_encoder(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._make_encoder(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._make_encoder(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = self._make_encoder(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = self._make_encoder(base_features * 8, base_features * 16)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_features * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _make_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.bottleneck(self.pool4(x4))

        pooled = self.global_pool(x5)
        logits = self.classifier(pooled)
        return logits


def default_config():
    return {
        "in_channels": 1,
        "num_classes": 2,
        "base_features": 32,
    }
