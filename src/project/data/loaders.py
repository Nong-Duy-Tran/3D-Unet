from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from src.baseline.data import get_dataloaders
from .splits import build_internal_val_subject_ids


def infer_classes_from_data(cfg: DictConfig, repo_root: Path) -> list[str]:
    train_dir = repo_root / cfg.data.train_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    for parent in [train_dir, *train_dir.parents]:
        split_file = parent / "split_info" / "folds.json"
        if split_file.exists():
            with split_file.open("r", encoding="utf-8") as handle:
                split_data = json.load(handle)
            class_names = split_data.get("class_names")
            if class_names:
                return [str(name) for name in class_names]

    class_dirs = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found under train_dir={train_dir}")
    return class_dirs


def build_dataloaders(
    cfg: DictConfig,
    repo_root: Path,
    train_subject_ids: list[str] | None = None,
    val_subject_ids: list[str] | None = None,
):
    train_dir = repo_root / cfg.data.train_dir
    use_validation = getattr(cfg.data, "use_validation", True)
    val_dir_cfg = getattr(cfg.data, "val_dir", None)
    val_dir = None
    if use_validation and val_dir_cfg:
        candidate = repo_root / val_dir_cfg
        if candidate.exists():
            val_dir = candidate
    if use_validation and val_dir is None:
        train_subject_ids, val_subject_ids = build_internal_val_subject_ids(cfg, repo_root)
    target_shape = getattr(cfg.data, "target_shape", (64, 64, 64))

    return get_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir) if val_dir is not None else str(train_dir) if use_validation else None,
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


def build_test_dataloader(cfg: DictConfig, repo_root: Path):
    if getattr(cfg.data, "test_dir", None):
        test_dir = repo_root / cfg.data.test_dir
    elif getattr(cfg.data, "train_dir", None):
        train_dir = repo_root / cfg.data.train_dir
        if train_dir.name == "train":
            test_dir = train_dir.parent / "test"
        else:
            raise ValueError(
                f"Unable to infer test_dir from train_dir={train_dir}. "
                "Either make train_dir end with 'train' or set cfg.data.test_dir explicitly."
            )
    else:
        val_dir = repo_root / cfg.data.val_dir
        if val_dir.name != "val":
            raise ValueError(
                f"Unable to infer test_dir from val_dir={val_dir}. "
                "Either make val_dir end with 'val' or set cfg.data.test_dir explicitly."
            )
        test_dir = val_dir.parent / "test"

    target_shape = getattr(cfg.data, "target_shape", (64, 64, 64))
    _, test_loader = get_dataloaders(
        train_dir=str(test_dir),
        val_dir=str(test_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_shape=tuple(target_shape),
        weighted_sampler=False,
        use_2d=getattr(cfg.data, "use_2d", False),
        num_slices=getattr(cfg.data, "num_slices", 8),
        slice_axis=getattr(cfg.data, "slice_axis", 0),
        resize_2d=getattr(cfg.data, "resize_2d", None),
        slice_strategy_train=getattr(cfg.data, "slice_strategy_val", "uniform"),
        slice_strategy_val=getattr(cfg.data, "slice_strategy_val", "uniform"),
        imagenet_norm=getattr(cfg.data, "imagenet_norm", False),
        randaugment=False,
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
        train_subject_ids=None,
        val_subject_ids=None,
    )
    return test_loader
