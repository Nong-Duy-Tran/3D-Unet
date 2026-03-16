from __future__ import annotations

import json
import math
import re
from pathlib import Path

from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

_JPG_SLICE_RE = re.compile(r"^(?P<stem>.+)_(?P<view>[a-z]+)_(?P<idx>-?\d+)$", re.IGNORECASE)


def load_subject_ids(cfg: DictConfig, repo_root: Path) -> tuple[list[str] | None, list[str] | None]:
    split_file = getattr(cfg.data, "split_file", None)
    if not split_file:
        return None, None

    split_path = repo_root / split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)

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

    def to_ids(items: list[object]) -> list[str]:
        if not items:
            return []
        first = items[0]
        if isinstance(first, str):
            return [str(item) for item in items]
        return [str(item["subject_id"]) for item in items]

    train_ids = to_ids(train_items)
    val_ids = to_ids(val_items)
    if not train_ids or not val_ids:
        raise ValueError("Split file does not contain valid train/val subject IDs.")
    return train_ids, val_ids


def _patient_id_from_session_id(session_id: str) -> str:
    parts = session_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return session_id


def _session_id_from_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    match = _JPG_SLICE_RE.match(Path(name).stem)
    if match:
        return match.group("stem")
    return Path(name).stem


def build_internal_val_subject_ids(cfg: DictConfig, repo_root: Path) -> tuple[list[str], list[str]]:
    train_dir = repo_root / cfg.data.train_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    classes = list(getattr(cfg.data, "classes", None) or [])
    if not classes:
        classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    data_format = str(getattr(cfg.data, "format", "nifti")).lower()
    subject_to_label: dict[str, int] = {}
    subject_to_group: dict[str, str] = {}

    for label_idx, class_name in enumerate(classes):
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue

        if data_format == "jpg":
            paths = sorted(class_dir.glob("*.jpg"))
        else:
            paths = sorted([*class_dir.glob("*.nii"), *class_dir.glob("*.nii.gz")])

        for path in paths:
            session_id = _session_id_from_name(path.name)
            patient_id = _patient_id_from_session_id(session_id)
            subject_to_label.setdefault(session_id, label_idx)
            subject_to_group.setdefault(session_id, patient_id)

    subject_ids = sorted(subject_to_label.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects in train_dir for internal validation split.")

    labels = [subject_to_label[sid] for sid in subject_ids]
    groups = [subject_to_group[sid] for sid in subject_ids]

    val_ratio = float(getattr(cfg.data, "internal_val_ratio", 0.2))
    val_ratio = min(max(val_ratio, 0.05), 0.5)
    seed = int(getattr(cfg.data, "internal_val_seed", getattr(cfg, "seed", 42)))
    n_splits = max(2, int(round(1.0 / val_ratio)))

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(subject_ids, labels, groups))
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(subject_ids, labels, groups))

    train_ids = [subject_ids[i] for i in train_idx]
    val_ids = [subject_ids[i] for i in val_idx]
    return train_ids, val_ids
