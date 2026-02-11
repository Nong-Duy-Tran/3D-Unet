import torch
import torch.nn as nn


class SliceAttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "tanh",
        use_layernorm: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (embed_dim // 2)
        act = activation.lower()
        if act == "gelu":
            activation_layer = nn.GELU()
        elif act == "relu":
            activation_layer = nn.ReLU(inplace=True)
        else:
            activation_layer = nn.Tanh()

        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(embed_dim))
        layers.extend(
            [
                nn.Linear(embed_dim, hidden_dim),
                activation_layer,
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.attn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1)


class DeiT2DClassifier(nn.Module):
    """DeiT encoder from timm + slice attention pooling for volume classification."""

    def __init__(
        self,
        timm_name="deit_base_patch16_224.fb_in1k",
        pretrained=True,
        image_size=224,
        in_channels=1,
        num_classes=2,
        drop_path_rate=0.2,
        attn_drop_rate=0.1,
        dropout=0.1,
        slice_attn_hidden_dim: int | None = None,
        slice_attn_dropout: float = 0.0,
        slice_attn_activation: str = "tanh",
        slice_attn_use_layernorm: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise ImportError("timm is required for DeiT. Install timm, then run again.") from exc

        self.encoder = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            img_size=image_size,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
        )
        embed_dim = getattr(self.encoder, "num_features", None)
        if embed_dim is None:
            raise ValueError("Unable to infer embedding dim from timm model.")
        self.pool = SliceAttentionPool(
            embed_dim,
            hidden_dim=slice_attn_hidden_dim,
            dropout=slice_attn_dropout,
            activation=slice_attn_activation,
            use_layernorm=slice_attn_use_layernorm,
        )
        self.dropout = nn.Dropout(dropout)
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
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def default_config():
    return {
        "timm_name": "deit_base_patch16_224.fb_in1k",
        "pretrained": True,
        "image_size": 224,
        "in_channels": 1,
        "num_classes": 2,
        "drop_path_rate": 0.2,
        "attn_drop_rate": 0.1,
        "dropout": 0.1,
        "slice_attn_hidden_dim": None,
        "slice_attn_dropout": 0.0,
        "slice_attn_activation": "tanh",
        "slice_attn_use_layernorm": False,
    }
