from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def build_trainer(cfg: DictConfig, ckpt_dir: Path, use_validation: bool = True):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
        from lightning.pytorch.loggers import WandbLogger
        from lightning.pytorch.plugins.environments import LightningEnvironment
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    callbacks = [TQDMProgressBar(refresh_rate=10, leave=False)]
    if use_validation:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="baseline-{epoch:03d}",
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                save_top_k=1,
                save_last=True,
            )
        )
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="baseline-{epoch:03d}",
                save_top_k=0,
                save_last=True,
            )
        )

    if use_validation and getattr(cfg.training, "early_stopping", None) and cfg.training.early_stopping.enabled:
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
            wandb_save_dir = Path(getattr(cfg, "output_dir", ckpt_dir))
            logger_kwargs = dict(
                project=cfg.logging.wandb.project,
                name=cfg.logging.wandb.name,
                tags=list(cfg.logging.wandb.tags),
                save_dir=str(wandb_save_dir),
                log_model=False,
            )
            wandb_group = getattr(cfg.logging.wandb, "group", None)
            if wandb_group:
                logger_kwargs["group"] = str(wandb_group)
            logger = WandbLogger(
                **logger_kwargs,
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
        plugins=[LightningEnvironment()],
    )
    accumulate = getattr(cfg.training, "accumulate_grad_batches", None)
    if accumulate is not None:
        trainer_kwargs["accumulate_grad_batches"] = int(accumulate)
    gradient_clip_val = getattr(cfg.training, "gradient_clip_val", None)
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = float(gradient_clip_val)
    gradient_clip_algorithm = getattr(cfg.training, "gradient_clip_algorithm", None)
    if gradient_clip_algorithm is not None:
        trainer_kwargs["gradient_clip_algorithm"] = str(gradient_clip_algorithm)
    return pl.Trainer(**trainer_kwargs)
