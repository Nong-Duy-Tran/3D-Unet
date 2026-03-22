from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

AXIS_NAME_TO_INDEX = {
    "coronal": 0,
    "axial": 1,
    "sagittal": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of cropped volume size along a chosen axis for OASIS crop versions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version_dirs",
        nargs="+",
        default=[
            "data/oasis_c1/processed_oasis_3d_cv5_v1",
            "data/oasis_c1/processed_oasis_3d_cv5_v2",
            "data/oasis_c1/processed_oasis_3d_cv5_v3",
        ],
        help="List of processed 3D cache directories to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="Display labels for the provided version_dirs.",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(AXIS_NAME_TO_INDEX.keys()),
        default="axial",
        help="Axis to compare. For the OASIS export pipeline, axial maps to source axis 1.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/eda/crop_compare_axial",
        help="Directory to save figures and summary files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def subject_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_axis_lengths(cache_root: Path, axis_idx: int) -> dict[str, int]:
    cache_dir = cache_root / "_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"_cache directory not found under: {cache_root}")

    lengths: dict[str, int] = {}
    for nii_path in sorted(cache_dir.glob("*.nii.gz")):
        subject_id = subject_id_from_path(nii_path)
        shape = nib.load(str(nii_path)).shape
        if axis_idx >= len(shape):
            raise ValueError(f"Axis index {axis_idx} out of range for {nii_path} with shape {shape}")
        lengths[subject_id] = int(shape[axis_idx])
    if not lengths:
        raise RuntimeError(f"No NIfTI files found in {cache_dir}")
    return lengths


def summarize(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    if len(args.version_dirs) != len(args.labels):
        raise ValueError("--version_dirs and --labels must have the same length")

    axis_idx = AXIS_NAME_TO_INDEX[args.axis]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lengths: dict[str, dict[str, int]] = {}
    for label, version_dir in zip(args.labels, args.version_dirs, strict=True):
        all_lengths[label] = load_axis_lengths(Path(version_dir), axis_idx)

    common_ids = sorted(set.intersection(*(set(v.keys()) for v in all_lengths.values())))
    if not common_ids:
        raise RuntimeError("No common session IDs found across the provided versions")

    aligned: dict[str, list[int]] = {
        label: [all_lengths[label][sid] for sid in common_ids] for label in args.labels
    }

    summary = {
        "axis": args.axis,
        "axis_index": axis_idx,
        "num_common_sessions": len(common_ids),
        "versions": {label: summarize(values) for label, values in aligned.items()},
    }

    csv_path = output_dir / f"{args.axis}_lengths_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_id", *args.labels])
        for sid in common_ids:
            writer.writerow([sid, *[all_lengths[label][sid] for label in args.labels]])

    json_path = output_dir / f"{args.axis}_lengths_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    values = [np.asarray(aligned[label], dtype=np.float32) for label in args.labels]
    global_min = min(float(v.min()) for v in values)
    global_max = max(float(v.max()) for v in values)
    bins = np.linspace(global_min - 0.5, global_max + 0.5, args.bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for label, arr in zip(args.labels, values, strict=True):
        axes[0].hist(arr, bins=bins, alpha=0.45, label=label, edgecolor="black")
    axes[0].set_title(f"{args.axis.capitalize()} length histogram")
    axes[0].set_xlabel(f"Number of slices along {args.axis} axis")
    axes[0].set_ylabel("Number of sessions")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].boxplot(values, tick_labels=args.labels, showmeans=True)
    axes[1].set_title(f"{args.axis.capitalize()} length boxplot")
    axes[1].set_ylabel(f"Number of slices along {args.axis} axis")
    axes[1].grid(alpha=0.2)

    fig_path = output_dir / f"{args.axis}_lengths_histogram.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure : {fig_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved summary: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
