"""
3D → 2D Slice Extraction Pipeline for OASIS
============================================

Input layout (from preprocess_oasis_3d.py):
  <input_dir>/_cache/*.nii.gz
  <input_dir>/split_info/folds.json

Output layout:
  <output_dir>/fold_<k>/train.npz       ← images (N, n_slices, H, W), labels (N,), subject_ids
  <output_dir>/fold_<k>/val.npz
  <output_dir>/fold_<k>/test.npz
  <output_dir>/fold_<k>/metadata.json

Split strategy:
  folds.json provides 80% train / 20% test per fold (stratified by class).
  The 80% train is further stratified-split 80/20 → train / val.

Slicing strategy:
  - Axial   (source axis X): n_axial   evenly-spaced centre slices
  - Coronal (source axis Z): n_coronal  evenly-spaced centre slices
  - Sagittal(source axis Y): n_sagittal evenly-spaced centre slices

Normalisation: per-volume 1st–99th percentile clipping → [0, 255] uint8.
Resizing: cubic interpolation to target_size × target_size.
"""
from __future__ import annotations

from utils import to_uint8

import cv2
import argparse
import os
import json
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm


def _evenly_spaced_indices(dim_size: int, n_slices: int) -> list[int]:
    """Return n_slices indices evenly spaced within the central 90% of the volume,
    excluding the first and last 5%."""
    start = int(dim_size * 0.05)
    end = int(dim_size * 0.95) - 1

    if n_slices == 1:
        return [dim_size // 2]

    step = (end - start) / (n_slices - 1)
    return [round(start + i * step) for i in range(n_slices)]


def extract_slices(
    nii_path: Path,
    subject_id: str,
    n_axial: int,
    n_coronal: int,
    n_sagittal: int,
    target_size: int = 224,
) -> Tuple[np.ndarray, bool]:
    """
    Load nii_path, extract evenly-spaced centre slices along all three axes.

    Returns (slices_array, True) with shape (n_axial+n_coronal+n_sagittal, H, W),
    or (None, False) on failure.
    """
    try:
        img = nib.load(str(nii_path))
        # Load and squeeze to remove singleton dims, then cast to float32
        # This minimizes intermediate float64 allocations
        data_np = img.get_fdata()
        data = np.squeeze(data_np)
        del data_np  # Free intermediate memory immediately
        data = data.astype(np.float32)

        if data.ndim != 3:
            print(f"  Warning: unexpected shape {data.shape} for {subject_id} – skipping.")
            return None, False

        data_u8 = to_uint8(data)
        x_dim, y_dim, z_dim = data_u8.shape

        n_total = n_axial + n_coronal + n_sagittal
        slices = np.zeros((n_total, target_size, target_size), dtype=np.uint8)

        idx = 0
        # Axial: X → slice (Y, Z)
        for x in _evenly_spaced_indices(x_dim, n_axial):
            slices[idx] = cv2.resize(data_u8[x, :, :], (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            idx += 1

        # Coronal: Z → slice (X, Y)
        for z in _evenly_spaced_indices(z_dim, n_coronal):
            slices[idx] = cv2.resize(data_u8[:, :, z], (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            idx += 1

        # Sagittal: Y → slice (X, Z)
        for y in _evenly_spaced_indices(y_dim, n_sagittal):
            slices[idx] = cv2.resize(data_u8[:, y, :], (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            idx += 1

        return slices, True

    except Exception as exc:
        print(f"  Error – {subject_id}: {exc}")
        return None, False


def _worker(task: tuple) -> tuple[str, str, str, np.ndarray, int, bool]:
    """Unpack task and call extract_slices.
    Returns (fold_name, split_name, subject_id, slices, label, ok)."""
    nii_path, subject_id, label, n_axial, n_coronal, n_sagittal, target_size, fold_name, split_name = task
    slices, ok = extract_slices(
        nii_path=Path(nii_path),
        subject_id=subject_id,
        n_axial=n_axial,
        n_coronal=n_coronal,
        n_sagittal=n_sagittal,
        target_size=target_size,
    )
    gc.collect()  # Force cleanup of temporary arrays
    return fold_name, split_name, subject_id, slices, label, ok


def save_npz_and_metadata(
    fold_output: Path,
    fold_data: Dict[str, List[Tuple]],
    n_axial: int,
    n_coronal: int,
    n_sagittal: int,
    target_size: int,
) -> None:
    """For each split (train/val/test), save NPZ archive and a shared metadata.json."""
    fold_output.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ("train", "val", "test"):
        if split not in fold_data or not fold_data[split]:
            continue

        images_list, labels_list, subject_ids_list = [], [], []
        for subject_id, slices, label in fold_data[split]:
            if slices is not None:
                images_list.append(slices)
                labels_list.append(label)
                subject_ids_list.append(subject_id)

        if not images_list:
            print(f"  Warning: {split} has no valid subjects")
            stats[split] = {"label_0": 0, "label_1": 0}
            continue

        images_array = np.array(images_list, dtype=np.uint8)    # (N, n_slices, H, W)
        labels_array = np.array(labels_list, dtype=np.uint8)     # (N,)

        npz_path = fold_output / f"{split}.npz"
        np.savez(
            npz_path,
            images=images_array,
            labels=labels_array,
            subject_ids=np.array(subject_ids_list, dtype=object),
        )
        print(f"  Saved {split}: {npz_path} – {len(images_list)} subjects")

        unique, counts = np.unique(labels_array, return_counts=True)
        stats[split] = {f"label_{int(k)}": int(v) for k, v in zip(unique, counts)}

    metadata = {
        "fold": int(fold_output.name.split("_")[1]),
        "description": "2D slices extracted from 3D MRI",
        "slice_composition": {
            "axial": f"indices 0-{n_axial - 1}",
            "coronal": f"indices {n_axial}-{n_axial + n_coronal - 1}",
            "sagittal": f"indices {n_axial + n_coronal}-{n_axial + n_coronal + n_sagittal - 1}",
            "total_slices_per_subject": n_axial + n_coronal + n_sagittal,
            "slice_size": [target_size, target_size],
        },
        "stats": stats,
    }

    metadata_path = fold_output / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")


def _stratified_train_val_split(
    items: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split of items (each with a 'label' key) into train / val.
    Maintains class proportions in both partitions.
    """
    rng = np.random.default_rng(seed)

    by_label: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        by_label[int(item["label"])].append(item)

    train_items: list[dict] = []
    val_items: list[dict] = []
    for cls_items in by_label.values():
        indices = np.arange(len(cls_items))
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_ratio))
        val_items.extend(cls_items[i] for i in indices[:n_val])
        train_items.extend(cls_items[i] for i in indices[n_val:])

    return train_items, val_items


def convert_dataset(args: argparse.Namespace) -> None:
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    cache_dir = input_root / "_cache"
    split_file = input_root / "split_info" / "folds.json"

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"folds.json not found: {split_file}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"_cache directory not found: {cache_dir}")

    with split_file.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    folds = split_data.get("folds", [])
    if not folds:
        raise RuntimeError("No folds found in folds.json")

    n_expected = args.n_axial + args.n_coronal + args.n_sagittal
    workers = args.workers or os.cpu_count() or 1

    print(f"\nFound {len(folds)} fold(s)")
    print(f"Slices per subject : {args.n_axial} axial + {args.n_coronal} coronal "
          f"+ {args.n_sagittal} sagittal = {n_expected} total")
    print(f"Target size        : {args.target_size}×{args.target_size}")
    print(f"Workers            : {workers}\n")

    all_tasks: list[tuple] = []

    for fold in folds:
        fold_idx = int(fold["fold_index"])
        fold_name = f"fold_{fold_idx}"

        raw_train = fold.get("train", [])
        test_items = fold.get("test", [])

        train_items, val_items = _stratified_train_val_split(
            raw_train, val_ratio=0.2, seed=fold_idx
        )

        print(f"  {fold_name}:")
        print(f"    train : {len(train_items)} subjects")
        print(f"    val   : {len(val_items)} subjects")
        print(f"    test  : {len(test_items)} subjects")

        for split_name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
            for item in items:
                sid = str(item["session_id"])
                label = int(item["label"])
                nii_path = cache_dir / f"{sid}.nii.gz"
                if not nii_path.exists():
                    print(f"  Warning: {nii_path} not found – skipping.")
                    continue
                all_tasks.append((
                    str(nii_path), sid, label,
                    args.n_axial, args.n_coronal, args.n_sagittal, args.target_size,
                    fold_name, split_name,
                ))

    if not all_tasks:
        print("\nNothing to do.")
        return

    print(f"\nTotal tasks: {len(all_tasks)}\n")

    # Parallel dispatch
    success = failed = 0
    fold_results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t for t in all_tasks}
        with tqdm(total=len(futures), desc="Converting", unit="subj") as pbar:
            for future in as_completed(futures):
                fold_name, split_name, subject_id, slices, label, ok = future.result()
                if ok and slices is not None:
                    success += 1
                    fold_results[fold_name][split_name].append((subject_id, slices, label))
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nWriting NPZ archives...\n")
    for fold_name in sorted(fold_results.keys()):
        fold_output = output_root / fold_name
        save_npz_and_metadata(
            fold_output,
            fold_results[fold_name],
            args.n_axial,
            args.n_coronal,
            args.n_sagittal,
            args.target_size,
        )

    total = len(all_tasks)
    print(f"\n{'=' * 60}")
    print("Conversion complete")
    print(f"{'=' * 60}")
    print(f"  Total subjects : {total}")
    print(f"  Success        : {success}")
    print(f"  Failed         : {failed}")
    print(f"  Output         : {output_root}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OASIS 3D NIfTI volumes to 2D NPZ archives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=str,
        default="./data/processed_oasis_3d",
        help="Root of 3D processed data (must contain _cache/ and split_info/folds.json).",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./data/processed_oasis_2d_npz",
        help="Destination root for 2D NPZ archives (fold_X/{train,val,test}.npz).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Output square slice resolution (target_size x target_size).",
    )
    parser.add_argument(
        "--n_axial", type=int, default=80,
        help="Number of axial slices per subject.",
    )
    parser.add_argument(
        "--n_coronal", type=int, default=80,
        help="Number of coronal slices per subject.",
    )
    parser.add_argument(
        "--n_sagittal", type=int, default=80,
        help="Number of sagittal slices per subject.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel worker processes (0 = use all CPU cores).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OASIS 3D → 2D NPZ Conversion")
    print("=" * 60)
    print(f"  Input           : {args.input_dir}")
    print(f"  Output          : {args.output_dir}")
    print(f"  Target size     : {args.target_size}×{args.target_size}")
    print(f"  Axial slices    : {args.n_axial}")
    print(f"  Coronal slices  : {args.n_coronal}")
    print(f"  Sagittal slices : {args.n_sagittal}")

    convert_dataset(args)


if __name__ == "__main__":
    main()
