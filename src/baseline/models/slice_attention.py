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

