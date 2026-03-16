import math

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
        layers.extend([nn.Linear(embed_dim, hidden_dim), activation_layer])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.attn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1)


class SlicePositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mode: str = "none",
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        self.mode = mode.lower()
        self.max_len = int(max_len)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.mode == "none":
            return
        if self.mode == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            return
        if self.mode == "sinusoidal":
            position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
            )
            pe = torch.zeros(1, self.max_len, embed_dim, dtype=torch.float32)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[1]])
            self.register_buffer("pos_embed", pe, persistent=False)
            return

        raise ValueError(f"Unsupported positional encoding mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D), got shape={tuple(x.shape)}")

        seq_len = x.shape[1]
        if self.mode == "none":
            return self.dropout(x)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional encoding capacity {self.max_len}"
            )
        return self.dropout(x + self.pos_embed[:, :seq_len])


class SliceSelfAttentionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mode: str = "none",
        num_heads: int = 8,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        attn_init: bool = False,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.attn_init = bool(attn_init)
        if self.mode == "none":
            self.encoder = nn.Identity()
            return
        if self.mode != "self_attention":
            raise ValueError(f"Unsupported slice sequence encoder mode: {mode}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if depth <= 0:
            raise ValueError("depth must be > 0 for self_attention mode")

        ff_dim = int(round(embed_dim * mlp_ratio))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if attn_dropout > 0:
            layer.self_attn.dropout = float(attn_dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    @staticmethod
    def _build_center_prior_mask(
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        center = float(seq_len - 1) / 2.0
        sigma = max(float(seq_len) / 6.0, 1.0)
        prior = torch.exp(-0.5 * ((pos - center) / sigma) ** 2)
        prior = prior / prior.sum().clamp_min(1e-8)

        # Add a soft bias toward center keys for every query position.
        prior_logits = 0.2 * torch.log(prior.clamp_min(1e-8))
        return prior_logits.unsqueeze(0).expand(seq_len, seq_len).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D), got shape={tuple(x.shape)}")
        if not self.attn_init or self.mode != "self_attention":
            return self.encoder(x)

        seq_len = x.shape[1]
        attn_mask = self._build_center_prior_mask(
            seq_len=seq_len,
            device=x.device,
            dtype=x.dtype,
        )
        return self.encoder(x, mask=attn_mask)
