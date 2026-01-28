"""Helpers for converting ADNI NIfTI to HDF5."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


def normalize_key(value: str) -> str:
    return str(value).strip().lower()


def parse_int_map(raw: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if not raw:
        return mapping
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid mapping entry: {part!r}")
        key, val = part.split("=", 1)
        mapping[normalize_key(key)] = int(val)
    return mapping


def parse_str_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw:
        return mapping
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid mapping entry: {part!r}")
        key, val = part.split("=", 1)
        mapping[normalize_key(key)] = val.strip()
    return mapping


def parse_ratio_list(raw: str) -> list[tuple[str, float]]:
    ratios: list[tuple[str, float]] = []
    if not raw:
        return ratios
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid ratio entry: {part!r}")
        key, val = part.split("=", 1)
        ratios.append((key.strip(), float(val)))
    return ratios


def load_label_map(label_map: str | None, label_map_json: str | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if label_map_json:
        with open(label_map_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("label-map-json must be a JSON object of {label: int}")
        for key, val in data.items():
            mapping[normalize_key(key)] = int(val)
    if label_map:
        mapping.update(parse_int_map(label_map))
    if mapping:
        return mapping
    # Default ADNI-friendly binary mapping; other labels are skipped.
    return {
        "cn": 0,
        "nl": 0,
        "normal": 0,
        "control": 0,
        "healthy": 0,
        "mci": 0,
        "emci": 0,
        "lmci": 0,
        "smc": 0,
        "ad": 1,
        "alzheimer": 1,
        "alzheimer's disease": 1,
        "ad dementia": 1,
    }


def sanitize_token(value: str) -> str:
    value = re.sub(r"\s+", "_", str(value).strip())
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def source_key_from_path(path: Path) -> str:
    stem = path.stem
    token = stem.split("_")[0]
    return normalize_key(token)


def resolve_path(path_str: str, images_dir: Path | None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if images_dir is None:
        return path
    return images_dir / path


def extract_image_id(path: Path) -> str | None:
    match = re.search(r"I\d{4,}", path.name)
    if match:
        return match.group(0)
    for part in path.parts:
        match = re.search(r"I\d{4,}", part)
        if match:
            return match.group(0)
    return None


def build_id_index(adni_root: Path) -> tuple[dict[str, Path], set[str]]:
    mapping: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in adni_root.rglob("*.nii*"):
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            continue
        image_id = extract_image_id(path)
        if not image_id:
            continue
        if image_id in mapping:
            duplicates.add(image_id)
            continue
        mapping[image_id] = path
    return mapping, duplicates


def load_nifti(path: Path) -> np.ndarray:
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    return img_data


def write_hdf5(output_path: Path, img_data: np.ndarray, label: int, meta: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=img_data, compression="gzip")
        f.create_dataset("label", data=int(label))
        for key, value in meta.items():
            f.attrs[key] = value


def resolve_nifti_path(
    *,
    path_value: str,
    image_id_value: str,
    images_dir: Path | None,
    id_index: dict[str, Path],
    duplicate_ids: set[str],
) -> tuple[Path | None, str | None]:
    if path_value:
        return resolve_path(path_value, images_dir), None
    if not image_id_value:
        return None, "missing_image_id"
    image_id = normalize_key(image_id_value).upper()
    if image_id in duplicate_ids:
        return None, "duplicate_image_id"
    path = id_index.get(image_id)
    if path is None:
        return None, "image_id_not_found"
    return path, None


def split_subjects(
    subject_ids: list[str],
    split_ratios: list[tuple[str, float]],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    rng.shuffle(subject_ids)
    total = len(subject_ids)
    counts: list[tuple[str, int]] = []
    remaining = total
    for idx, (name, ratio) in enumerate(split_ratios):
        if idx == len(split_ratios) - 1:
            count = remaining
        else:
            count = int(total * ratio)
        counts.append((name, count))
        remaining -= count
    assignment: dict[str, str] = {}
    cursor = 0
    for split_name, count in counts:
        for subject_id in subject_ids[cursor : cursor + count]:
            assignment[subject_id] = split_name
        cursor += count
    return assignment
