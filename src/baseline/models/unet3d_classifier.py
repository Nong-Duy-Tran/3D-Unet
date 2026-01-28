import torch
import torch.nn as nn
from src.modules.pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D


class UNet3DClassifier(nn.Module):
    """
    3D U-Net adapted for classification
    Uses U-Net encoder + global pooling + classifier head

    Args:
        in_channels: Number of input channels (default: 1 for MRI)
        num_classes: Number of output classes (default: 2 for binary)
        f_maps: Base number of feature maps (default: 64)
        num_levels: Number of encoder/decoder levels (default: 4)
        use_residual: Use residual blocks instead of double conv (default: False)
    """

    def __init__(self, in_channels=1, num_classes=2, f_maps=64, num_levels=4, use_residual=False):
        super().__init__()

        if use_residual:
            self.unet = ResidualUNet3D(
                in_channels=in_channels,
                out_channels=num_classes,
                final_sigmoid=False,
                f_maps=f_maps,
                num_levels=num_levels,
                is_segmentation=True,
            )
        else:
            self.unet = UNet3D(
                in_channels=in_channels,
                out_channels=num_classes,
                final_sigmoid=False,
                f_maps=f_maps,
                num_levels=num_levels,
                is_segmentation=True,
            )

        bottleneck_features = f_maps * (2 ** (num_levels - 1))

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        encoders_features = []
        for encoder in self.unet.encoders:
            x = encoder(x)
            encoders_features.append(x)

        bottleneck_features = encoders_features[-1]
        pooled = self.global_pool(bottleneck_features)
        logits = self.classifier(pooled)
        return logits


def default_config():
    return {
        "in_channels": 1,
        "num_classes": 2,
        "f_maps": 64,
        "num_levels": 4,
        "use_residual": False,
    }
