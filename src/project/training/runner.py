from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from src.baseline.utils import find_repo_root
from src.project.data import build_dataloaders, load_subject_ids
from src.project.training.module import build_lightning_module, compute_class_weights
from src.project.training.trainer import build_trainer


def resolve_repo_root() -> Path:
    return find_repo_root(Path(get_original_cwd()))


def run_training(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    pl.seed_everything(cfg.seed, workers=True)

    repo_root = resolve_repo_root()
    ckpt_dir = repo_root / cfg.checkpoint.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_subject_ids, val_subject_ids = load_subject_ids(cfg, repo_root)
    train_loader, val_loader = build_dataloaders(
        cfg,
        repo_root,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )

    lr_scheduler_cfg = getattr(cfg.training, "lr_scheduler", None)
    if lr_scheduler_cfg and lr_scheduler_cfg.enabled and lr_scheduler_cfg.get("name") == "OneCycleLR":
        steps = lr_scheduler_cfg.get("steps_per_epoch")
        if not steps or steps == float("inf"):
            with open_dict(lr_scheduler_cfg):
                lr_scheduler_cfg["steps_per_epoch"] = len(train_loader)
        if not lr_scheduler_cfg.get("epochs"):
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
            class_weights = compute_class_weights(labels, cfg.model.num_classes)
        print(f"Class weights: {class_weights.tolist()}")

    model = build_lightning_module(cfg, class_weights=class_weights)
    trainer = build_trainer(cfg, ckpt_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


@hydra.main(config_path="../../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()

