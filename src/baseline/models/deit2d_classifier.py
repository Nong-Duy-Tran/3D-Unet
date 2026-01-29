import torch
import torch.nn as nn


class SliceAttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
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
        self.pool = SliceAttentionPool(embed_dim)
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
    }
