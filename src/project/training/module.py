from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch

from src.project.losses import FocalLoss
from src.project.models import create_model_from_config


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


def build_lightning_module(cfg: DictConfig, class_weights: "torch.Tensor | None" = None):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.utilities.types import STEP_OUTPUT
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    from torch import nn

    class LitClassifier(pl.LightningModule):
        def __init__(self, cfg_in: DictConfig, class_weights_in: "torch.Tensor | None" = None):
            super().__init__()
            self.cfg = cfg_in
            self.save_hyperparameters(OmegaConf.to_container(cfg_in, resolve=True))

            self.model = create_model_from_config(cfg_in)
            self._freeze_backbone = bool(getattr(cfg_in.training, "freeze_backbone", False))
            self._freeze_backbone_ratio = float(getattr(cfg_in.training, "freeze_backbone_ratio", 1.0))
            self._freeze_backbone_ratio = max(0.0, min(1.0, self._freeze_backbone_ratio))
            self._freeze_backbone_all = self._freeze_backbone_ratio >= 1.0
            if self._freeze_backbone:
                encoder = getattr(self.model, "encoder", None)
                if encoder is None:
                    raise ValueError("freeze_backbone=true but model has no encoder to freeze.")
                encoder_params = list(encoder.parameters())
                num_params_to_freeze = int(len(encoder_params) * self._freeze_backbone_ratio)
                for i, param in enumerate(encoder_params):
                    if i < num_params_to_freeze:
                        param.requires_grad = False
                if self._freeze_backbone_all:
                    encoder.eval()

            loss_cfg = getattr(cfg_in.training, "loss", None)
            loss_name = "cross_entropy"
            label_smoothing = 0.0
            focal_gamma = 2.0
            if loss_cfg:
                if getattr(loss_cfg, "name", None) is not None:
                    loss_name = str(loss_cfg.name).strip().lower()
                if getattr(loss_cfg, "label_smoothing", None) is not None:
                    label_smoothing = float(loss_cfg.label_smoothing)
                if getattr(loss_cfg, "gamma", None) is not None:
                    focal_gamma = float(loss_cfg.gamma)

            if loss_name in {"cross_entropy", "ce"}:
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_in, label_smoothing=label_smoothing)
            elif loss_name in {"focal", "focalloss", "focal_loss"}:
                self.loss_fn = FocalLoss(
                    gamma=focal_gamma,
                    weight=class_weights_in,
                    label_smoothing=label_smoothing,
                )
            else:
                raise ValueError(f"Unsupported training.loss.name: {loss_name}")

            self._val_preds: list[torch.Tensor] = []
            self._val_targets: list[torch.Tensor] = []
            self._val_probs: list[torch.Tensor] = []

        def train(self, mode: bool = True):
            super().train(mode)
            if self._freeze_backbone and self._freeze_backbone_all:
                encoder = getattr(self.model, "encoder", None)
                if encoder is not None:
                    encoder.eval()
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def _batch_f1(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            preds = preds.view(-1).long()
            targets = targets.view(-1).long()
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            denom = 2 * tp + fp + fn
            if denom == 0:
                return torch.tensor(0.0, device=preds.device)
            return (2 * tp) / denom

        def _batch_auc(self, probs: torch.Tensor, targets: torch.Tensor) -> float | None:
            if probs.shape[-1] != 2:
                return None
            try:
                from sklearn.metrics import roc_auc_score
            except Exception:
                return None
            y_true = targets.detach().cpu().numpy()
            y_score = probs[:, 1].detach().cpu().numpy()
            if len(set(y_true.tolist())) < 2:
                return None
            return float(roc_auc_score(y_true, y_score))

        def _shared_step(self, batch: Any, stage: str) -> STEP_OUTPUT:
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_fn(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()
            f1 = self._batch_f1(preds, y)
            auc = self._batch_auc(probs, y)
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/f1", f1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            if auc is not None:
                self.log(f"{stage}/auc", auc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            return self._shared_step(batch, stage="train")

        def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_fn(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self._val_preds.append(preds.detach().cpu())
            self._val_targets.append(y.detach().cpu())
            self._val_probs.append(probs.detach().cpu())
            return loss

        def on_validation_epoch_start(self) -> None:
            self._val_preds = []
            self._val_targets = []
            self._val_probs = []

        def on_validation_epoch_end(self) -> None:
            if not self._val_preds:
                return
            try:
                from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
            except Exception:
                return

            preds = torch.cat(self._val_preds).numpy()
            targets = torch.cat(self._val_targets).numpy()
            probs = torch.cat(self._val_probs).numpy() if self._val_probs else None

            precision = precision_score(targets, preds, zero_division=0, average="macro")
            recall = recall_score(targets, preds, zero_division=0, average="macro")
            f1_epoch = f1_score(targets, preds, zero_division=0, average="macro")
            acc_epoch = accuracy_score(targets, preds)
            if probs is not None and len(set(targets.tolist())) > 1:
                if probs.shape[1] > 2:
                    auc_epoch = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
                else:
                    auc_epoch = roc_auc_score(targets, probs[:, 1])
            else:
                auc_epoch = 0.0
            cm = confusion_matrix(targets, preds, labels=list(range(int(probs.shape[1])))) if probs is not None else None

            self.log("val/acc_epoch", acc_epoch, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/precision", precision, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/recall", recall, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/f1_epoch", f1_epoch, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/auc_epoch", auc_epoch, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            if cm is None:
                return
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.log(
                        f"val/cm_{i}{j}",
                        float(cm[i, j]),
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        def configure_optimizers(self):
            optimizer_cfg = getattr(self.cfg.training, "optimizer", None)
            if optimizer_cfg and getattr(optimizer_cfg, "name", None):
                optimizer_params = OmegaConf.to_container(optimizer_cfg, resolve=True)
                optimizer_name = optimizer_params.pop("name")
                optimizer_cls = getattr(torch.optim, optimizer_name, None)
                if optimizer_cls is None:
                    raise ValueError(f"Unknown optimizer: {optimizer_name}")
                optimizer_params.setdefault("lr", self.cfg.training.lr)
                optimizer_params.setdefault("weight_decay", self.cfg.training.weight_decay)
                optimizer = optimizer_cls(self.parameters(), **optimizer_params)
            else:
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.cfg.training.lr,
                    weight_decay=self.cfg.training.weight_decay,
                )

            lr_scheduler_cfg = getattr(self.cfg.training, "lr_scheduler", None)
            if not lr_scheduler_cfg or not lr_scheduler_cfg.enabled:
                return optimizer

            import torch.optim.lr_scheduler as lr_schedulers

            scheduler_cfg = OmegaConf.to_container(lr_scheduler_cfg, resolve=True)
            scheduler_cfg.pop("enabled", None)
            monitor = scheduler_cfg.pop("monitor", "val/loss")
            interval = scheduler_cfg.pop("interval", "epoch")
            frequency = scheduler_cfg.pop("frequency", 1)
            scheduler_name = scheduler_cfg.pop("name", None)
            if scheduler_name is None:
                return optimizer

            scheduler_cls = getattr(lr_schedulers, scheduler_name, None)

            if scheduler_name == "CosineWarmup":
                warmup_epochs = int(scheduler_cfg.pop("warmup_epochs", 0))
                warmup_start_factor = float(scheduler_cfg.pop("warmup_start_factor", 0.1))
                eta_min = float(scheduler_cfg.pop("eta_min", 0.0))
                t_max = scheduler_cfg.pop("T_max", None)
                if t_max is None:
                    t_max = max(1, int(self.cfg.training.max_epochs) - warmup_epochs)
                cosine = lr_schedulers.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
                if warmup_epochs > 0:
                    warmup = lr_schedulers.LinearLR(
                        optimizer,
                        start_factor=warmup_start_factor,
                        total_iters=warmup_epochs,
                    )
                    scheduler = lr_schedulers.SequentialLR(
                        optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[warmup_epochs],
                    )
                else:
                    scheduler = cosine
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": monitor,
                        "interval": interval,
                        "frequency": frequency,
                    },
                }

            if scheduler_cls is None:
                raise ValueError(f"Unknown lr scheduler: {scheduler_name}")

            if scheduler_name == "OneCycleLR":
                if "steps_per_epoch" not in scheduler_cfg:
                    if self.trainer is not None and getattr(self.trainer, "num_training_batches", 0):
                        scheduler_cfg["steps_per_epoch"] = self.trainer.num_training_batches
                if "epochs" not in scheduler_cfg:
                    scheduler_cfg["epochs"] = self.trainer.max_epochs
                interval = "step"

            scheduler = scheduler_cls(optimizer, **scheduler_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": interval,
                    "frequency": frequency,
                },
            }

    return LitClassifier(cfg, class_weights_in=class_weights)

