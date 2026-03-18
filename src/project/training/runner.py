from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from src.baseline.utils import find_repo_root
from src.project.data import (
    build_dataloaders,
    build_test_dataloader,
    infer_classes_from_data,
    load_subject_ids,
)
from src.project.training.artifacts import save_model_artifacts
from src.project.training.module import build_lightning_module, compute_class_weights
from src.project.training.trainer import build_trainer


def resolve_repo_root() -> Path:
    return find_repo_root(Path(get_original_cwd()))


def resolve_artifact_dir(ckpt_dir: Path) -> Path:
    if ckpt_dir.name.startswith("fold_"):
        return ckpt_dir.parent / "artifact"
    return ckpt_dir / "artifact"


def run_training(cfg: DictConfig) -> None:
    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover
        raise ImportError("lightning is required. Install it, then run again.") from exc

    pl.seed_everything(cfg.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(str(getattr(cfg.training, "matmul_precision", "high")))

    repo_root = resolve_repo_root()
    inferred_classes = infer_classes_from_data(cfg, repo_root)
    with open_dict(cfg):
        cfg.data.classes = inferred_classes
        cfg.model.num_classes = len(inferred_classes)
    print(OmegaConf.to_yaml(cfg))
    print(f"Inferred classes from data: {inferred_classes}")

    ckpt_dir = repo_root / cfg.checkpoint.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_validation = bool(getattr(cfg.data, "use_validation", True))

    train_subject_ids, val_subject_ids = load_subject_ids(cfg, repo_root)
    if not use_validation:
        val_subject_ids = None
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
    sample_batch = next(iter(train_loader))
    save_model_artifacts(model, sample_batch, cfg, resolve_artifact_dir(ckpt_dir))
    trainer = build_trainer(cfg, ckpt_dir, use_validation=use_validation and val_loader is not None)
    if use_validation and val_loader is not None:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        test_ckpt = "best"
    else:
        print("Running training without validation.")
        trainer.fit(model, train_dataloaders=train_loader)
        test_ckpt = "last"

    test_loader = build_test_dataloader(cfg, repo_root)
    print(f"\nRunning test with checkpoint: {test_ckpt}")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=test_ckpt)


@hydra.main(config_path="../../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
