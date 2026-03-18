from __future__ import annotations

import torch
import torch.nn as nn

from ..slice_attention import (
    SliceAttentionPool,
    SlicePositionalEncoding,
    SliceSelfAttentionEncoder,
)


class VGG25DClassifier(nn.Module):
    """VGG encoder + slice attention pooling for 2.5D volume classification."""

    def __init__(
        self,
        vgg_name: str = "vgg16_bn",
        pretrained: bool = False,
        image_size: int = 224,
        in_channels: int = 5,
        num_classes: int = 2,
        dropout: float = 0.1,
        slice_embed_dim: int | None = None,
        slice_pos_encoding: str = "none",
        slice_pos_max_len: int = 512,
        slice_pos_dropout: float = 0.0,
        slice_sequence_encoder: str = "none",
        slice_num_heads: int = 8,
        slice_transformer_depth: int = 1,
        slice_transformer_mlp_ratio: float = 4.0,
        slice_transformer_dropout: float = 0.0,
        slice_transformer_attn_dropout: float = 0.0,
        attn_init: bool = False,
        slice_attn_hidden_dim: int | None = None,
        slice_attn_dropout: float = 0.0,
        slice_attn_activation: str = "tanh",
        slice_attn_use_layernorm: bool = False,
        slice_attn_mode: str = "basic",
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
        sequence_dim = int(slice_embed_dim) if slice_embed_dim is not None else embed_dim
        if sequence_dim <= 0:
            raise ValueError("slice_embed_dim must be > 0")
        self.slice_proj = nn.Identity() if sequence_dim == embed_dim else nn.Linear(embed_dim, sequence_dim)
        self.slice_pos = SlicePositionalEncoding(
            embed_dim=sequence_dim,
            mode=slice_pos_encoding,
            max_len=slice_pos_max_len,
            dropout=slice_pos_dropout,
        )
        self.slice_sequence = SliceSelfAttentionEncoder(
            embed_dim=sequence_dim,
            mode=slice_sequence_encoder,
            num_heads=slice_num_heads,
            depth=slice_transformer_depth,
            mlp_ratio=slice_transformer_mlp_ratio,
            dropout=slice_transformer_dropout,
            attn_dropout=slice_transformer_attn_dropout,
            attn_init=attn_init,
        )
        self.pool = SliceAttentionPool(
            sequence_dim,
            hidden_dim=slice_attn_hidden_dim,
            dropout=slice_attn_dropout,
            activation=slice_attn_activation,
            use_layernorm=slice_attn_use_layernorm,
            mode=slice_attn_mode,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sequence_dim, num_classes)

        if not pretrained:
            self._init_weights(self.encoder)
        self._init_weights(self.slice_proj)
        if slice_pos_encoding == "learned":
            self._init_weights(self.slice_pos)
        self._init_weights(self.slice_sequence)
        self._init_weights(self.pool)
        self._init_weights(self.classifier)

    @staticmethod
    def _build_backbone(vgg_name: str, pretrained: bool):
        try:
            import torchvision.models as tv_models
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for VGG25D. Install torchvision, then run again.") from exc

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
        backbone.features[0] = new_conv

    @classmethod
    def _init_weights(cls, module: nn.Module) -> None:
        for submodule in module.modules():
            cls._init_module(submodule)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, N, C, H, W), got shape={tuple(x.shape)}")

        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.encoder(x)
        feats = feats.view(b, n, -1)
        feats = self.slice_proj(feats)
        feats = self.slice_pos(feats)
        feats = self.slice_sequence(feats)
        pooled = self.pool(feats)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def default_config():
    return {
        "vgg_name": "vgg16_bn",
        "pretrained": False,
        "image_size": 224,
        "in_channels": 5,
        "num_classes": 2,
        "dropout": 0.1,
        "slice_embed_dim": None,
        "slice_pos_encoding": "none",
        "slice_pos_max_len": 512,
        "slice_pos_dropout": 0.0,
        "slice_sequence_encoder": "none",
        "slice_num_heads": 8,
        "slice_transformer_depth": 1,
        "slice_transformer_mlp_ratio": 4.0,
        "slice_transformer_dropout": 0.0,
        "slice_transformer_attn_dropout": 0.0,
        "attn_init": False,
        "slice_attn_hidden_dim": None,
        "slice_attn_dropout": 0.0,
        "slice_attn_activation": "tanh",
        "slice_attn_use_layernorm": False,
        "slice_attn_mode": "basic",
    }
