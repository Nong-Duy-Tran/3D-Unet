from __future__ import annotations

import torch


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: "torch.Tensor | None" = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("FocalLoss gamma must be >= 0.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("FocalLoss reduction must be one of: none, mean, sum.")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

