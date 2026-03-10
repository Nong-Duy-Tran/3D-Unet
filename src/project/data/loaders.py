from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from src.baseline.data import get_dataloaders


def build_dataloaders(
    cfg: DictConfig,
    repo_root: Path,
    train_subject_ids: list[str] | None = None,
    val_subject_ids: list[str] | None = None,
):
    train_dir = repo_root / cfg.data.train_dir
    val_dir = repo_root / cfg.data.val_dir
    target_shape = getattr(cfg.data, "target_shape", (64, 64, 64))

    return get_dataloaders(
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
        jpg_view=getattr(cfg.data, "jpg_view", "axial"),
        jpg_mode=getattr(cfg.data, "jpg_mode", "stack"),
        window_size=getattr(cfg.data, "window_size", 5),
        image_size=getattr(cfg.data, "image_size", None),
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
    )
