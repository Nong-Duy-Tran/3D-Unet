from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.baseline.data import get_dataloaders
from src.baseline.models import get_model
from src.baseline.utils import find_repo_root


def _resolve_paths(cfg: DictConfig) -> tuple[Path, Path, Path]:
    orig_cwd = Path(get_original_cwd())
    repo_root = find_repo_root(orig_cwd)
    train_dir = repo_root / cfg.data.train_dir
    val_dir = repo_root / cfg.data.val_dir
    ckpt_dir = repo_root / cfg.checkpoint.dir
    return train_dir, val_dir, ckpt_dir


def _make_lightning_module(cfg: DictConfig):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.utilities.types import STEP_OUTPUT
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    import torch
    from torch import nn

    class LitClassifier(pl.LightningModule):
        def __init__(self, cfg_in: DictConfig):
            super().__init__()
            self.cfg = cfg_in
            self.save_hyperparameters(OmegaConf.to_container(cfg_in, resolve=True))

            self.model = get_model(
                model_name=cfg_in.model.name,
                in_channels=cfg_in.model.in_channels,
                num_classes=cfg_in.model.num_classes,
                base_features=cfg_in.model.base_features,
            )
            self.loss_fn = nn.CrossEntropyLoss()

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
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log(f"{stage}/acc", acc, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log(f"{stage}/f1", f1, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
            if auc is not None:
                self.log(f"{stage}/auc", auc, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
            return loss

        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            return self._shared_step(batch, stage="train")

        def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            return self._shared_step(batch, stage="val")

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )

    return LitClassifier(cfg)


def _make_trainer(cfg: DictConfig, ckpt_dir: Path):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    callbacks = [
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

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    return trainer


@hydra.main(config_path="../../configs/baseline", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    pl.seed_everything(cfg.seed, workers=True)

    train_dir, val_dir, ckpt_dir = _resolve_paths(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_shape=tuple(cfg.data.target_shape),
        use_hdf5=cfg.data.use_hdf5,
    )

    model = _make_lightning_module(cfg)
    trainer = _make_trainer(cfg, ckpt_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
