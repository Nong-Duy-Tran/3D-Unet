from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig


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

