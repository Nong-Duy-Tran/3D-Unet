from __future__ import annotations

import re

import torch
import torch.nn as nn

from ..slice_attention import (
    SliceAttentionPool,
    SlicePositionalEncoding,
    SliceSelfAttentionEncoder,
)


class Timm25DClassifier(nn.Module):
    """
    2.5D classifier for ordered slice windows.

    Expected input:
      - (B, N, C, H, W), where N is the number of sampled center slices
      - C is the local window size, e.g. 5 grayscale slices as channels
    """

    def __init__(
        self,
        timm_name: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        in_channels: int = 5,
        num_classes: int = 2,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
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
        slice_attn_hidden_dim: int | None = None,
        slice_attn_dropout: float = 0.0,
        slice_attn_activation: str = "tanh",
        slice_attn_use_layernorm: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise ImportError("timm is required for Timm25DClassifier.") from exc

        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "in_chans": in_channels,
            "img_size": image_size,
            "drop_path_rate": drop_path_rate,
            "attn_drop_rate": attn_drop_rate,
        }
        self.encoder = self._create_timm_model(timm, timm_name, create_kwargs)
        embed_dim = getattr(self.encoder, "num_features", None)
        if embed_dim is None:
            raise ValueError("Unable to infer embedding dim from timm model.")
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
        )
        self.pool = SliceAttentionPool(
            sequence_dim,
            hidden_dim=slice_attn_hidden_dim,
            dropout=slice_attn_dropout,
            activation=slice_attn_activation,
            use_layernorm=slice_attn_use_layernorm,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sequence_dim, num_classes)

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

    @staticmethod
    def _create_timm_model(timm_module, timm_name: str, create_kwargs: dict):
        kwargs = dict(create_kwargs)
        while True:
            try:
                return timm_module.create_model(timm_name, **kwargs)
            except TypeError as exc:
                message = str(exc)
                match = re.search(r"unexpected keyword argument '([^']+)'", message)
                if match is None:
                    raise
                bad_key = match.group(1)
                if bad_key not in kwargs:
                    raise
                kwargs.pop(bad_key)


def default_config():
    return {
        "timm_name": "resnet18",
        "pretrained": True,
        "image_size": 224,
        "in_channels": 5,
        "num_classes": 2,
        "drop_path_rate": 0.0,
        "attn_drop_rate": 0.0,
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
        "slice_attn_hidden_dim": None,
        "slice_attn_dropout": 0.0,
        "slice_attn_activation": "tanh",
        "slice_attn_use_layernorm": False,
    }
