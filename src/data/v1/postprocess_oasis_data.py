"""
Create oriented 2D JPG slices from preprocessed OASIS NIfTI volumes.

Input layout (from preprocess_oasis_data.py):
  <input_dir>/_cache/*.nii.gz
  <input_dir>/split_info/folds.json

Output layout:
  <output_dir>/{coronal,axial,sagittal}/fold_<k>/{train,test}/{class_name}/*.jpg

Export convention for downstream 2D data, aligned to the current on-disk
preprocessed volumes:
  - export coronal  from source axis 0 -> rot180
  - export axial    from source axis 1 -> rot180
  - export sagittal from source axis 2 -> rot90 counter-clockwise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _iter_nifti_samples(input_dir: Path):
    split_file = input_dir / "split_info" / "folds.json"
    cache_dir = input_dir / "_cache"

    if split_file.exists() and cache_dir.exists():
        with split_file.open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        for fold in split_data.get("folds", []):
            fold_name = f"fold_{int(fold['fold_index'])}"
            for split_name in ("train", "test"):
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
        default="data/processed_oasis_3d_cv5_v1",
        help="Directory containing preprocessed folds with NIfTI files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_oasis_2d_cv5_v1",
        help="Directory to write oriented, padded JPG slices.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Output square slice resolution (target_size x target_size).",
    )
    parser.add_argument(
        "--slice_ratio_start",
        type=float,
        default=0.0,
        help="Start ratio along each axis.",
    )
    parser.add_argument(
        "--slice_ratio_end",
        type=float,
        default=1.0,
        help="End ratio along each axis.",
    )
    parser.add_argument(
        "--slice_step",
        type=int,
        default=1,
        help="Step size when iterating slices along each axis.",
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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.slice_step <= 0:
        raise ValueError("--slice_step must be > 0")
    if not (0.0 <= args.slice_ratio_start < args.slice_ratio_end <= 1.0):
        raise ValueError("Require 0.0 <= --slice_ratio_start < --slice_ratio_end <= 1.0")
    if not (0 <= args.pad_value <= 255):
        raise ValueError("--pad_value must be in [0, 255]")

    samples = list(_iter_nifti_samples(input_dir))
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
            axis_len = int(volume.shape[source_axis])
            start = int(axis_len * args.slice_ratio_start)
            end = int(axis_len * args.slice_ratio_end)
            start = max(0, min(start, axis_len))
            end = max(start + 1, min(end, axis_len))
            out_class_dir = output_dir / axis_name / fold_name / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(start, end, args.slice_step):
                out_path = out_class_dir / f"{subject_id}_{axis_name}_{idx:03d}.jpg"
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
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Volumes   : {len(samples)}")
    print(f"Saved     : {saved}")
    print(f"Skipped   : {skipped}")


if __name__ == "__main__":
    main()
