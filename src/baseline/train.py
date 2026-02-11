from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import torch

from src.baseline.data import get_dataloaders
from src.baseline.models import get_model
from src.baseline.utils import find_repo_root


def _resolve_paths(cfg: DictConfig) -> tuple[Path, Path, Path, Path]:
    orig_cwd = Path(get_original_cwd())
    repo_root = find_repo_root(orig_cwd)
    train_dir = repo_root / cfg.data.train_dir
    val_dir = repo_root / cfg.data.val_dir
    ckpt_dir = repo_root / cfg.checkpoint.dir
    return train_dir, val_dir, ckpt_dir, repo_root


def _load_subject_ids(cfg: DictConfig, repo_root: Path) -> tuple[list[str] | None, list[str] | None]:
    split_file = getattr(cfg.data, "split_file", None)
    if not split_file:
        return None, None

    split_path = repo_root / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        split_data = json.load(f)

    if "folds" in split_data:
        fold_index = int(getattr(cfg.data, "fold_index", 0))
        folds = split_data.get("folds", [])
        if fold_index < 0 or fold_index >= len(folds):
            raise ValueError(f"fold_index {fold_index} out of range for {len(folds)} folds")
        fold = folds[fold_index]
        train_items = fold.get("train", [])
        val_items = fold.get("val", [])
    else:
        train_items = split_data.get("train", [])
        val_items = split_data.get("val", [])

    def _to_ids(items):
        if not items:
            return []
        first = items[0]
        if isinstance(first, str):
            return [str(x) for x in items]
        return [str(x["subject_id"]) for x in items]

    train_ids = _to_ids(train_items)
    val_ids = _to_ids(val_items)
    if not train_ids or not val_ids:
        raise ValueError("Split file does not contain valid train/val subject IDs.")
    return train_ids, val_ids


def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


def _make_lightning_module(cfg: DictConfig, class_weights: "torch.Tensor | None" = None):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.utilities.types import STEP_OUTPUT
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    import torch
    from torch import nn

    class LitClassifier(pl.LightningModule):
        def __init__(self, cfg_in: DictConfig, class_weights: "torch.Tensor | None" = None):
            super().__init__()
            self.cfg = cfg_in
            self.save_hyperparameters(OmegaConf.to_container(cfg_in, resolve=True))

            self.model = get_model(
                model_name=cfg_in.model.name,
                **self._model_kwargs(cfg_in),
            )
            self._freeze_backbone = bool(getattr(cfg_in.training, "freeze_backbone", False))
            if self._freeze_backbone:
                encoder = getattr(self.model, "encoder", None)
                if encoder is None:
                    raise ValueError("freeze_backbone=true but model has no encoder to freeze.")
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()
            loss_cfg = getattr(cfg_in.training, "loss", None)
            label_smoothing = 0.0
            if loss_cfg and getattr(loss_cfg, "label_smoothing", None) is not None:
                label_smoothing = float(loss_cfg.label_smoothing)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            self._val_preds: list[torch.Tensor] = []
            self._val_targets: list[torch.Tensor] = []
            self._val_probs: list[torch.Tensor] = []

        def train(self, mode: bool = True):
            super().train(mode)
            if self._freeze_backbone:
                encoder = getattr(self.model, "encoder", None)
                if encoder is not None:
                    encoder.eval()
            return self

        @staticmethod
        def _model_kwargs(cfg_in: DictConfig) -> dict[str, Any]:
            model_kwargs: dict[str, Any] = {
                "in_channels": cfg_in.model.in_channels,
                "num_classes": cfg_in.model.num_classes,
            }
            if cfg_in.model.name == "simple":
                model_kwargs["base_features"] = cfg_in.model.base_features
                return model_kwargs
            if cfg_in.model.name == "deit2d":
                image_size = getattr(cfg_in.model, "image_size", None)
                if image_size is None:
                    image_size = int(cfg_in.data.target_shape[1])
                in_channels = cfg_in.model.in_channels
                if getattr(cfg_in.data, "rgb_mode", False):
                    in_channels = 3
                model_kwargs.update(
                    {
                        "timm_name": cfg_in.model.timm_name,
                        "pretrained": cfg_in.model.pretrained,
                        "image_size": image_size,
                        "in_channels": in_channels,
                        "drop_path_rate": cfg_in.model.drop_path_rate,
                        "attn_drop_rate": cfg_in.model.attn_drop_rate,
                        "dropout": cfg_in.model.dropout,
                        "slice_attn_hidden_dim": getattr(cfg_in.model, "slice_attn_hidden_dim", None),
                        "slice_attn_dropout": float(getattr(cfg_in.model, "slice_attn_dropout", 0.0)),
                        "slice_attn_activation": getattr(cfg_in.model, "slice_attn_activation", "tanh"),
                        "slice_attn_use_layernorm": bool(getattr(cfg_in.model, "slice_attn_use_layernorm", False)),
                    }
                )
                return model_kwargs
            if cfg_in.model.name == "simpleunet2d":
                in_channels = cfg_in.model.in_channels
                if getattr(cfg_in.data, "rgb_mode", False):
                    in_channels = 3
                model_kwargs.update(
                    {
                        "in_channels": in_channels,
                        "base_features": cfg_in.model.base_features,
                        "head_hidden": cfg_in.model.head_hidden,
                        "dropout": cfg_in.model.dropout,
                        "head_dropout": cfg_in.model.head_dropout,
                    }
                )
                return model_kwargs
            if cfg_in.model.name == "vit2d":
                image_size = getattr(cfg_in.model, "image_size", None)
                if image_size is None:
                    image_size = int(cfg_in.data.target_shape[1])
                model_kwargs.update(
                    {
                        "image_size": image_size,
                        "patch_size": cfg_in.model.patch_size,
                        "embed_dim": cfg_in.model.embed_dim,
                        "depth": cfg_in.model.depth,
                        "num_heads": cfg_in.model.num_heads,
                        "mlp_dim": cfg_in.model.mlp_dim,
                        "dropout": cfg_in.model.dropout,
                    }
                )
                return model_kwargs
            if cfg_in.model.name == "swin2d":
                image_size = getattr(cfg_in.model, "image_size", None)
                if image_size is None:
                    image_size = int(cfg_in.data.target_shape[1])
                model_kwargs.update(
                    {
                        "image_size": image_size,
                        "patch_size": cfg_in.model.patch_size,
                        "embed_dim": cfg_in.model.embed_dim,
                        "depth": cfg_in.model.depth,
                        "num_heads": cfg_in.model.num_heads,
                        "window_size": cfg_in.model.window_size,
                        "mlp_dim": cfg_in.model.mlp_dim,
                        "dropout": cfg_in.model.dropout,
                    }
                )
                return model_kwargs
            if cfg_in.model.name == "swin3d":
                model_kwargs.update(
                    {
                        "patch_size": cfg_in.model.patch_size,
                        "embed_dim": cfg_in.model.embed_dim,
                        "depth": cfg_in.model.depth,
                        "num_heads": cfg_in.model.num_heads,
                        "window_size": cfg_in.model.window_size,
                        "mlp_dim": cfg_in.model.mlp_dim,
                        "dropout": cfg_in.model.dropout,
                    }
                )
                return model_kwargs

            f_maps = getattr(cfg_in.model, "f_maps", None)
            if f_maps is None:
                f_maps = getattr(cfg_in.model, "base_features", 64)
            model_kwargs["f_maps"] = f_maps

            num_levels = getattr(cfg_in.model, "num_levels", None)
            if num_levels is not None:
                model_kwargs["num_levels"] = num_levels

            return model_kwargs

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def _batch_f1(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Binary F1 for batch; returns 0 if no positive predictions/targets.
            preds = preds.view(-1).long()
            targets = targets.view(-1).long()
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            denom = (2 * tp + fp + fn)
            if denom == 0:
                return torch.tensor(0.0, device=preds.device)
            return (2 * tp) / denom

        def _batch_auc(self, probs: torch.Tensor, targets: torch.Tensor) -> float | None:
            # Binary AUC for batch; returns None if undefined.
            if probs.shape[-1] != 2:
                return None
            try:
                from sklearn.metrics import roc_auc_score
            except Exception:
                return None
            y_true = targets.detach().cpu().numpy()
            y_score = probs[:, 1].detach().cpu().numpy()
            # AUC is undefined if only one class present.
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
            # Log epoch metrics only to keep output clean.
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
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc_score,
                    confusion_matrix,
                )
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
            if cm is not None:
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

            try:
                import torch.optim.lr_scheduler as lr_schedulers
            except Exception as exc:  # pragma: no cover
                raise ImportError("torch.optim.lr_scheduler is required for LR scheduling") from exc

            scheduler_cfg = OmegaConf.to_container(lr_scheduler_cfg, resolve=True)
            scheduler_cfg.pop("enabled", None)
            monitor = scheduler_cfg.pop("monitor", "val/loss")
            interval = scheduler_cfg.pop("interval", "epoch")
            frequency = scheduler_cfg.pop("frequency", 1)
            scheduler_name = scheduler_cfg.pop("name", None)
            if scheduler_name is None:
                return optimizer

            try:
                scheduler_cls = getattr(lr_schedulers, scheduler_name)
            except AttributeError as exc:
                scheduler_cls = None

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
                        optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs
                    )
                    scheduler = lr_schedulers.SequentialLR(
                        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
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
                raise ValueError(f"Unknown lr scheduler: {scheduler_name}") from exc

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

    return LitClassifier(cfg, class_weights=class_weights)


def _make_trainer(cfg: DictConfig, ckpt_dir: Path):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
        from lightning.pytorch.loggers import WandbLogger
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    callbacks = [
        TQDMProgressBar(refresh_rate=10, leave=False),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="baseline-{epoch:03d}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=1,
            save_last=True,
        )
    ]
    if getattr(cfg.training, "early_stopping", None) and cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                mode=cfg.training.early_stopping.mode,
                patience=cfg.training.early_stopping.patience,
                min_delta=cfg.training.early_stopping.min_delta,
                verbose=True,
            )
        )

    logger = None
    if cfg.logging.wandb.enabled:
        try:
            logger = WandbLogger(
                project=cfg.logging.wandb.project,
                name=cfg.logging.wandb.name,
                tags=list(cfg.logging.wandb.tags),
                save_dir=str(ckpt_dir),
                log_model=False,
            )
        except Exception as exc:  # pragma: no cover
            raise ImportError("wandb is required when logging.wandb.enabled=true") from exc

    trainer_kwargs = dict(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    accumulate = getattr(cfg.training, "accumulate_grad_batches", None)
    if accumulate is not None:
        trainer_kwargs["accumulate_grad_batches"] = int(accumulate)
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


@hydra.main(config_path="../../configs/baseline", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    pl.seed_everything(cfg.seed, workers=True)

    train_dir, val_dir, ckpt_dir, repo_root = _resolve_paths(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_subject_ids, val_subject_ids = _load_subject_ids(cfg, repo_root)

    target_shape = getattr(cfg.data, "target_shape", (64, 64, 64))
    train_loader, val_loader = get_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_shape=tuple(target_shape),
        weighted_sampler=getattr(cfg.data, "weighted_sampler", False),
        use_2d=getattr(cfg.data, "use_2d", False),
        num_slices=getattr(cfg.data, "num_slices", 8),
        slice_axis=getattr(cfg.data, "slice_axis", 0),
        resize_2d=getattr(cfg.data, "resize_2d", None),
        slice_strategy_train=getattr(cfg.data, "slice_strategy_train", "random"),
        slice_strategy_val=getattr(cfg.data, "slice_strategy_val", "uniform"),
        imagenet_norm=getattr(cfg.data, "imagenet_norm", False),
        randaugment=getattr(cfg.data, "randaugment", False),
        randaugment_ops=getattr(cfg.data, "randaugment_ops", 2),
        randaugment_mag=getattr(cfg.data, "randaugment_mag", 9),
        rgb_mode=getattr(cfg.data, "rgb_mode", False),
        normalize=getattr(cfg.data, "normalize", True),
        data_format=getattr(cfg.data, "format", "nifti"),
        classes=getattr(cfg.data, "classes", None),
        class_map=getattr(cfg.data, "class_map", None),
        jpg_view=getattr(cfg.data, "jpg_view", "ax"),
        image_size=getattr(cfg.data, "image_size", None),
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )

    lr_scheduler_cfg = getattr(cfg.training, "lr_scheduler", None)
    if lr_scheduler_cfg and lr_scheduler_cfg.enabled and lr_scheduler_cfg.get("name") == "OneCycleLR":
        steps = lr_scheduler_cfg.get("steps_per_epoch")
        if not steps or steps == float("inf"):
            from omegaconf import open_dict
            with open_dict(lr_scheduler_cfg):
                lr_scheduler_cfg["steps_per_epoch"] = len(train_loader)
        if not lr_scheduler_cfg.get("epochs"):
            from omegaconf import open_dict
            with open_dict(lr_scheduler_cfg):
                lr_scheduler_cfg["epochs"] = cfg.training.max_epochs

    class_weights = None
    class_weights_cfg = getattr(cfg.training, "class_weights", None)
    if class_weights_cfg and class_weights_cfg.enabled:
        values = class_weights_cfg.get("values") if hasattr(class_weights_cfg, "get") else None
        if values is not None:
            class_weights = torch.tensor(values, dtype=torch.float)
        else:
            labels = getattr(train_loader.dataset, "labels", None)
            if not labels:
                raise ValueError("class_weights enabled but dataset labels are unavailable.")
            class_weights = _compute_class_weights(labels, cfg.model.num_classes)
        print(f"Class weights: {class_weights.tolist()}")

    model = _make_lightning_module(cfg, class_weights=class_weights)
    trainer = _make_trainer(cfg, ckpt_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
