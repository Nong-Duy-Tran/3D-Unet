"""
Create oriented 2D JPG slices from preprocessed OASIS NIfTI volumes.

Input layout (from preprocess_oasis_3d.py):
  <input_dir>/_cache/*.nii.gz
  <input_dir>/split_info/folds.json

Output layout:
    <output_dir>/{coronal,axial,sagittal}/fold_<k>/{train,val,test}/{class_name}/*.jpg

Slicing strategy:
  - Axial   (source axis 1): N_AXIAL evenly-spaced centre slices  → rot180
  - Coronal (source axis 0): N_CORONAL evenly-spaced centre slices → rot180
  - Sagittal(source axis 2): N_SAGITTAL evenly-spaced centre slices → rot90 

Normalisation: per-slice uint8 min-max normalization → [0, 255].
Resizing: cubic interpolation to target_size × target_size.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

EXPORT_VIEWS: tuple[dict[str, int | str], ...] = (
    {"name": "coronal", "source_axis": 0, "rotation_k": 2},
    {"name": "axial", "source_axis": 1, "rotation_k": 2},
    {"name": "sagittal", "source_axis": 2, "rotation_k": 1},
)


def _subject_id_from_filename(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def _evenly_spaced_indices(dim_size: int, n_slices: int) -> list[int]:
    """
    Return n_slices indices evenly spaced within the central 90% of the volume,
    excluding the first and last 5%.
    """
    start = int(dim_size * 0.05)
    end = int(dim_size * 0.95) - 1

    if n_slices == 1:
        return [dim_size // 2]
    
    step = (end - start) / (n_slices - 1)
    indices = [round(start + i * step) for i in range(n_slices)]
    return indices


def _extract_slice(volume: np.ndarray, source_axis: int, idx: int) -> np.ndarray:
    if source_axis == 0:
        return volume[idx, :, :]
    if source_axis == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def _orient_slice_for_export(slice_2d: np.ndarray, rotation_k: int) -> np.ndarray:
    rotation_k = int(rotation_k) % 4
    if rotation_k == 0:
        return slice_2d
    return np.rot90(slice_2d, rotation_k)


def _to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.where(finite, arr, 0.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _center_pad_or_crop(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int,
) -> np.ndarray:
    h, w = image.shape

    if h > target_h:
        start_h = (h - target_h) // 2
        image = image[start_h:start_h + target_h, :]
        h = target_h
    if w > target_w:
        start_w = (w - target_w) // 2
        image = image[:, start_w:start_w + target_w]
        w = target_w

    out = np.full((target_h, target_w), fill_value=pad_value, dtype=image.dtype)
    top = (target_h - h) // 2
    left = (target_w - w) // 2
    out[top:top + h, left:left + w] = image
    return out


def _label_from_item(item: dict) -> int:
    if "label" in item and item["label"] is not None:
        return int(item["label"])
    return 0 if str(item.get("class_name", "normal")) == "normal" else 1


def _to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _coerce_csv_item_types(row: dict) -> dict:
    item = dict(row)
    if "label" in item and item["label"] not in (None, ""):
        item["label"] = int(item["label"])
    if "fold" in item and item["fold"] not in (None, ""):
        item["fold"] = int(item["fold"])
    if "cdr" in item and item["cdr"] not in (None, ""):
        item["cdr"] = float(item["cdr"])
    if "label_raw_cdr" in item and item["label_raw_cdr"] not in (None, ""):
        item["label_raw_cdr"] = float(item["label_raw_cdr"])
    if "is_mr1" in item:
        item["is_mr1"] = _to_bool(item.get("is_mr1"))
    if "is_averaged_image" in item:
        item["is_averaged_image"] = _to_bool(item.get("is_averaged_image"))
    return item


def _dedupe_by_session(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            continue
        if session_id in seen:
            continue
        seen.add(session_id)
        deduped.append(item)
    return deduped


def _select_stratified_indices(
    labels: list[int],
    preferred_n_splits: int,
    fold_selector: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels)
    classes, class_counts = np.unique(labels_arr, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for stratified split.")

    min_class_count = int(class_counts.min())
    n_splits = min(int(preferred_n_splits), min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough samples per class for StratifiedKFold.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    selected_fold = int(fold_selector) % n_splits
    all_splits = list(skf.split(np.zeros(len(labels_arr)), labels_arr))
    return all_splits[selected_fold]


def _stratified_train_val_test(
    items: list[dict],
    fold_index: int,
    random_state: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [_label_from_item(item) for item in items]

    train_val_idx, test_idx = _select_stratified_indices(
        labels=labels,
        preferred_n_splits=5,
        fold_selector=fold_index,
        random_state=random_state + fold_index,
    )

    train_val_items = [items[int(i)] for i in train_val_idx]
    train_val_labels = [labels[int(i)] for i in train_val_idx]

    train_idx_rel, val_idx_rel = _select_stratified_indices(
        labels=train_val_labels,
        preferred_n_splits=5,
        fold_selector=fold_index,
        random_state=random_state + 1000 + fold_index,
    )

    train_items = [train_val_items[int(i)] for i in train_idx_rel]
    val_items = [train_val_items[int(i)] for i in val_idx_rel]
    test_items = [items[int(i)] for i in test_idx]
    return train_items, val_items, test_items


def _build_split_json_64_16_20(
    input_dir: Path,
    split_json_name: str,
    random_state: int,
) -> Path:
    split_info_dir = input_dir / "split_info"
    source_split_file = split_info_dir / "folds.json"
    source_csv_file = split_info_dir / "folds.csv"

    split_data: dict = {}
    if source_split_file.exists():
        with source_split_file.open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)

    fold_items_map: dict[int, list[dict]] = {}
    if source_csv_file.exists():
        with source_csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                session_id = str(row.get("session_id", "")).strip()
                if not session_id:
                    continue
                fold_index = int(row.get("fold", 0))
                item = _coerce_csv_item_types(row)
                item["fold"] = fold_index
                fold_items_map.setdefault(fold_index, []).append(item)
    else:
        for fold in split_data.get("folds", []):
            fold_index = int(fold["fold_index"])
            fold_items_map[fold_index] = _dedupe_by_session(
                list(fold.get("train", []))
                + list(fold.get("val", []))
                + list(fold.get("test", []))
            )

    if not fold_items_map:
        raise RuntimeError(
            f"No fold items found in {source_csv_file} or {source_split_file}."
        )

    out_data = dict(split_data) if split_data else {}
    out_data["n_folds"] = len(fold_items_map)
    out_data["split_ratio"] = {"train": 0.64, "val": 0.16, "test": 0.20}
    out_data["split_strategy"] = "StratifiedKFold(5)->test(20%), StratifiedKFold(5)->val(16%)"
    out_data["split_source"] = "folds.csv" if source_csv_file.exists() else "folds.json"

    all_class_names: set[str] = set()
    for items in fold_items_map.values():
        for item in items:
            all_class_names.add(str(item.get("class_name", "")))
    if all_class_names:
        out_data["class_names"] = sorted(all_class_names)

    new_folds: list[dict] = []
    for fold_index in sorted(fold_items_map):
        all_items = _dedupe_by_session(fold_items_map[fold_index])
        if not all_items:
            raise RuntimeError(f"Fold {fold_index} has no samples.")

        train_items, val_items, test_items = _stratified_train_val_test(
            items=all_items,
            fold_index=fold_index,
            random_state=random_state,
        )

        def _with_split(items_for_split: list[dict], split_name: str) -> list[dict]:
            out_items: list[dict] = []
            for item in items_for_split:
                updated = dict(item)
                updated["split"] = split_name
                updated["fold"] = fold_index
                out_items.append(updated)
            return out_items

        new_fold = {"fold_index": fold_index}
        new_fold["train"] = _with_split(train_items, "train")
        new_fold["val"] = _with_split(val_items, "val")
        new_fold["test"] = _with_split(test_items, "test")
        new_folds.append(new_fold)

    out_data["folds"] = new_folds
    output_split_file = split_info_dir / split_json_name
    with output_split_file.open("w", encoding="utf-8") as handle:
        json.dump(out_data, handle, indent=2)

    for fold in out_data.get("folds", []):
        fold_index = int(fold["fold_index"])
        ratios: list[str] = []
        for split_name in ("train", "val", "test"):
            items = fold.get(split_name, [])
            counts = Counter(str(item.get("class_name", "")) for item in items)
            n_total = len(items)
            ratios.append(f"{split_name}:{n_total} {dict(counts)}")
        print(f"Fold {fold_index} split summary -> " + " | ".join(ratios))

    return output_split_file


def _iter_nifti_samples(input_dir: Path, split_json_path: Path | None = None):
    split_file = split_json_path or (input_dir / "split_info" / "folds.json")
    cache_dir = input_dir / "_cache"

    if split_file.exists() and cache_dir.exists():
        with split_file.open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        for fold in split_data.get("folds", []):
            fold_name = f"fold_{int(fold['fold_index'])}"
            for split_name in ("train", "val", "test"):
                for item in fold.get(split_name, []):
                    session_id = str(item["session_id"])
                    class_name = str(item["class_name"])
                    nii_path = cache_dir / f"{session_id}.nii.gz"
                    if nii_path.exists():
                        yield fold_name, split_name, class_name, nii_path
        return

    # Backward-compatible fallback for old materialized fold directories.
    fold_dirs = sorted([p for p in input_dir.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        fold_dirs = [input_dir]

    for fold_dir in fold_dirs:
        for split_name in ("train", "val", "test"):
            split_dir = fold_dir / split_name
            if not split_dir.exists():
                continue
            class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
            for class_dir in class_dirs:
                for nii_path in sorted(class_dir.glob("*.nii*")):
                    if nii_path.name.endswith(".nii") or nii_path.name.endswith(".nii.gz"):
                        yield fold_dir.name, split_name, class_dir.name, nii_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create oriented, center-padded JPG slices from preprocessed OASIS NIfTI volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed_oasis_3d",
        help="Directory containing preprocessed folds with NIfTI files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_oasis_2d_jpeg",
        help="Directory to write oriented, padded JPG slices.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Output square slice resolution (target_size x target_size).",
    )
    parser.add_argument(
        "--pad_value",
        type=int,
        default=0,
        help="Blank value used for symmetric padding in uint8 space [0..255].",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality when saving slices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing slice files.",
    )
    parser.add_argument(
        "--limit_volumes",
        type=int,
        default=None,
        help="Optional max number of volumes to process (for quick checks).",
    )
    parser.add_argument(
        "--n_axial",
        type=int,
        default=80,
        help="Number of axial slices per subject (extracted from source axis 1).",
    )
    parser.add_argument(
        "--n_coronal",
        type=int,
        default=80,
        help="Number of coronal slices per subject (extracted from source axis 0).",
    )
    parser.add_argument(
        "--n_sagittal",
        type=int,
        default=80,
        help="Number of sagittal slices per subject (extracted from source axis 2).",
    )
    parser.add_argument(
        "--split_json_name",
        type=str,
        default="folds_64_16_20.json",
        help="Output split json name written under <input_dir>/split_info/.",
    )
    parser.add_argument(
        "--split_random_state",
        type=int,
        default=42,
        help="Random seed used by StratifiedKFold to create train/val/test.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not (0 <= args.pad_value <= 255):
        raise ValueError("--pad_value must be in [0, 255]")
    if args.n_axial <= 0 or args.n_coronal <= 0 or args.n_sagittal <= 0:
        raise ValueError("--n_axial, --n_coronal, --n_sagittal must be > 0")

    split_json_path: Path | None = None
    split_info_dir = input_dir / "split_info"
    if split_info_dir.exists():
        split_json_path = _build_split_json_64_16_20(
            input_dir=input_dir,
            split_json_name=args.split_json_name,
            random_state=args.split_random_state,
        )

    samples = list(_iter_nifti_samples(input_dir, split_json_path=split_json_path))
    if not samples:
        raise RuntimeError(f"No NIfTI files found under: {input_dir}")
    if args.limit_volumes is not None:
        samples = samples[: args.limit_volumes]

    saved = 0
    skipped = 0

    for fold_name, split_name, class_name, nii_path in tqdm(samples, desc="Volumes"):
        nii = nib.load(str(nii_path))
        volume = nii.get_fdata()
        if volume.ndim == 4:
            volume = volume[..., 0]

        subject_id = _subject_id_from_filename(nii_path.name)

        for view in EXPORT_VIEWS:
            axis_name = str(view["name"])
            source_axis = int(view["source_axis"])
            rotation_k = int(view["rotation_k"])
            
            # Determine number of slices based on axis
            if source_axis == 0:  # coronal
                n_slices = args.n_coronal
            elif source_axis == 1:  # axial
                n_slices = args.n_axial
            else:  # sagittal (source_axis == 2)
                n_slices = args.n_sagittal
            
            axis_len = int(volume.shape[source_axis])
            slice_indices = _evenly_spaced_indices(axis_len, n_slices)
            
            out_class_dir = output_dir / axis_name / fold_name / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for slice_idx, idx in enumerate(slice_indices):
                out_path = out_class_dir / f"{subject_id}_{axis_name}_{slice_idx:03d}.jpg"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                slice_2d = _extract_slice(volume, source_axis, idx)
                slice_2d = _orient_slice_for_export(slice_2d, rotation_k)
                slice_u8 = _to_uint8(slice_2d)
                padded = _center_pad_or_crop(
                    slice_u8,
                    target_h=args.target_size,
                    target_w=args.target_size,
                    pad_value=args.pad_value,
                )

                Image.fromarray(padded, mode="L").save(
                    out_path, format="JPEG", quality=args.jpeg_quality
                )
                saved += 1

    print("=" * 70)
    print("Slice export complete")
    print("=" * 70)
    print(f"Input dir      : {input_dir}")
    if split_json_path is not None:
        print(f"Split json     : {split_json_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Slices config  : {args.n_axial} axial + {args.n_coronal} coronal + {args.n_sagittal} sagittal = {args.n_axial + args.n_coronal + args.n_sagittal} per subject")
    print(f"Volumes        : {len(samples)}")
    print(f"Saved          : {saved}")
    print(f"Skipped        : {skipped}")


if __name__ == "__main__":
    main()

