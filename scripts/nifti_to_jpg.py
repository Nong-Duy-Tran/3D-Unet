from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.helpers.adni_defaults import DEFAULT_SOURCE_ROOT_MAP  # noqa: E402
from src.helpers.adni_nifti import extract_image_id, load_nifti  # noqa: E402


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _is_slice_valid(slice_2d: np.ndarray, brightness_thresh: int, ratio_thresh: float = 0.05) -> bool:
    ratio = float(np.sum(slice_2d > brightness_thresh)) / float(slice_2d.size)
    return ratio > ratio_thresh


def _normalize_and_save(
    slice_2d: np.ndarray,
    save_path: Path,
    var_threshold: float,
    clip_limit: float,
    tile_grid: int,
    image_size: int,
) -> bool:
    slice_u8 = _to_uint8(slice_2d)
    if np.var(slice_u8) < var_threshold:
        return False

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for CLAHE. Install it and retry.") from exc

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    slice_u8 = clahe.apply(slice_u8)
    slice_u8 = cv2.resize(slice_u8, (image_size, image_size))
    cv2.imwrite(str(save_path), slice_u8)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADNI NIfTI volumes to 2D JPG slices")
    parser.add_argument("--output-dir", default="data/ADNI/jpg", help="Output root for JPG slices")
    parser.add_argument(
        "--source-root-map",
        default="",
        help="Map source->root, e.g. abb=data/ADNI/abb/ADNI,bbc=data/ADNI/bbc/ADNI,ecc=data/ADNI/ecc/ADNI",
    )
    parser.add_argument("--slice-ratio-start", type=float, default=0.3, help="Axial start ratio")
    parser.add_argument("--slice-ratio-end", type=float, default=0.7, help="Axial end ratio")
    parser.add_argument("--axial-step", type=int, default=4, help="Axial step")
    parser.add_argument("--sagittal-step", type=int, default=7, help="Sagittal step")
    parser.add_argument("--coronal-step", type=int, default=7, help="Coronal step")
    parser.add_argument("--central-window", type=int, default=20, help="Central window for sag/cor")
    parser.add_argument("--brightness-threshold", type=int, default=10, help="Brightness threshold")
    parser.add_argument("--var-threshold", type=float, default=50, help="Variance threshold")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clipLimit")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tileGridSize")
    parser.add_argument("--image-size", type=int, default=224, help="Output image size")
    parser.add_argument("--limit", type=int, default=None, help="Optional NIfTI limit")
    parser.add_argument("--reorient-ras", action="store_true", help="Reorient to RAS+ before slicing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPGs")
    args = parser.parse_args()

    source_root_map = {}
    if args.source_root_map:
        for pair in args.source_root_map.split(","):
            if not pair:
                continue
            key, value = pair.split("=", 1)
            source_root_map[key.strip()] = Path(value.strip())
    if not source_root_map:
        source_root_map = DEFAULT_SOURCE_ROOT_MAP.copy()

    label_map = {"abb": "AD", "bbc": "MCI", "ecc": "CN"}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_paths: list[tuple[str, Path]] = []
    for source_key, root in source_root_map.items():
        if not root.exists():
            print(f"Skipping missing root: {root}")
            continue
        for path in root.rglob("*.nii*"):
            if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
                continue
            all_paths.append((source_key, path))

    if args.limit is not None:
        all_paths = all_paths[: args.limit]

    for source_key, path in tqdm(all_paths, desc="Converting"):
        label_name = label_map.get(source_key)
        if label_name is None:
            continue
        img_data = load_nifti(path, reorient_ras=args.reorient_ras)
        if img_data.ndim == 4:
            img_data = img_data[..., 0]

        image_id = extract_image_id(path) or path.stem.replace(".nii", "")
        out_dir = output_root / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        z_start = int(img_data.shape[2] * args.slice_ratio_start)
        z_end = int(img_data.shape[2] * args.slice_ratio_end)
        for i in range(z_start, z_end, args.axial_step):
            slice_2d = np.rot90(img_data[:, :, i])
            if not _is_slice_valid(slice_2d, args.brightness_threshold):
                continue
            out_path = out_dir / f"{image_id}_ax_{i}.jpg"
            if out_path.exists() and not args.overwrite:
                continue
            _normalize_and_save(
                slice_2d,
                out_path,
                var_threshold=args.var_threshold,
                clip_limit=args.clahe_clip,
                tile_grid=args.clahe_grid,
                image_size=args.image_size,
            )

        center_x = img_data.shape[0] // 2
        for i in range(center_x - args.central_window, center_x + args.central_window, args.sagittal_step):
            slice_2d = np.rot90(img_data[i, :, :])
            if not _is_slice_valid(slice_2d, args.brightness_threshold):
                continue
            out_path = out_dir / f"{image_id}_sag_{i}.jpg"
            if out_path.exists() and not args.overwrite:
                continue
            _normalize_and_save(
                slice_2d,
                out_path,
                var_threshold=args.var_threshold,
                clip_limit=args.clahe_clip,
                tile_grid=args.clahe_grid,
                image_size=args.image_size,
            )

        center_y = img_data.shape[1] // 2
        for i in range(center_y - args.central_window, center_y + args.central_window, args.coronal_step):
            slice_2d = np.rot90(img_data[:, i, :])
            if not _is_slice_valid(slice_2d, args.brightness_threshold):
                continue
            out_path = out_dir / f"{image_id}_cor_{i}.jpg"
            if out_path.exists() and not args.overwrite:
                continue
            _normalize_and_save(
                slice_2d,
                out_path,
                var_threshold=args.var_threshold,
                clip_limit=args.clahe_clip,
                tile_grid=args.clahe_grid,
                image_size=args.image_size,
            )


if __name__ == "__main__":
    main()
