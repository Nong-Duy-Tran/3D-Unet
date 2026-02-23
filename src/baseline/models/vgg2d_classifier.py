from __future__ import annotations

import torch
import torch.nn as nn

from .slice_attention import SliceAttentionPool


class VGG2DClassifier(nn.Module):
    """VGG encoder + slice attention pooling for volume classification."""

    def __init__(
        self,
        vgg_name: str = "vgg16_bn",
        pretrained: bool = True,
        image_size: int = 224,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.1,
        slice_attn_hidden_dim: int | None = None,
        slice_attn_dropout: float = 0.0,
        slice_attn_activation: str = "tanh",
        slice_attn_use_layernorm: bool = False,
    ):
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be > 0")

        backbone = self._build_backbone(vgg_name=vgg_name, pretrained=pretrained)
        if in_channels != 3:
            self._adapt_first_conv(backbone, in_channels=in_channels)

        self.encoder = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
            *list(backbone.classifier[:-1]),
        )
        embed_dim = 4096
        self.pool = SliceAttentionPool(
            embed_dim,
            hidden_dim=slice_attn_hidden_dim,
            dropout=slice_attn_dropout,
            activation=slice_attn_activation,
            use_layernorm=slice_attn_use_layernorm,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _build_backbone(vgg_name: str, pretrained: bool):
        try:
            import torchvision.models as tv_models
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for VGG2D. Install torchvision, then run again.") from exc

        spec = {
            "vgg11": ("vgg11", "VGG11_Weights"),
            "vgg11_bn": ("vgg11_bn", "VGG11_BN_Weights"),
            "vgg13": ("vgg13", "VGG13_Weights"),
            "vgg13_bn": ("vgg13_bn", "VGG13_BN_Weights"),
            "vgg16": ("vgg16", "VGG16_Weights"),
            "vgg16_bn": ("vgg16_bn", "VGG16_BN_Weights"),
            "vgg19": ("vgg19", "VGG19_Weights"),
            "vgg19_bn": ("vgg19_bn", "VGG19_BN_Weights"),
        }.get(vgg_name.lower())

        if spec is None:
            raise ValueError(f"Unsupported vgg_name: {vgg_name}")

        ctor_name, weights_enum_name = spec
        ctor = getattr(tv_models, ctor_name)

        weights = None
        if pretrained:
            weights_enum = getattr(tv_models, weights_enum_name, None)
            if weights_enum is None:
                raise ValueError(f"{weights_enum_name} is unavailable in this torchvision version")
            weights = weights_enum.IMAGENET1K_V1

        try:
            return ctor(weights=weights)
        except TypeError:
            return ctor(pretrained=pretrained)

    @staticmethod
    def _adapt_first_conv(backbone: nn.Module, in_channels: int) -> None:
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        first_conv = backbone.features[0]
        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError("VGG first layer is not Conv2d as expected")

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
            elif in_channels == 2:
                new_conv.weight[:, :2].copy_(first_conv.weight[:, :2])
            elif in_channels == 3:
                new_conv.weight.copy_(first_conv.weight)
            else:
                new_conv.weight[:, :3].copy_(first_conv.weight)
                mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c : c + 1].copy_(mean_weight)

            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        backbone.features[0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feats = self.encoder(x)  # (B*S, 4096)
        feats = feats.view(b, s, -1)
        pooled = self.pool(feats)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def default_config():
    return {
        "vgg_name": "vgg16_bn",
        "pretrained": True,
        "image_size": 224,
        "in_channels": 1,
        "num_classes": 2,
        "dropout": 0.1,
        "slice_attn_hidden_dim": None,
        "slice_attn_dropout": 0.0,
        "slice_attn_activation": "tanh",
        "slice_attn_use_layernorm": False,
    }

