from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

AXIS_NAME_TO_INDEX = {
    "coronal": 0,
    "axial": 1,
    "sagittal": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot foreground voxel counts per normalized slice position for multiple crop versions.",
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
        help="List of processed 3D version directories.",
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
        help="Axis to analyze. For the OASIS export pipeline, axial maps to source axis 1.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-5,
        help="Foreground threshold on absolute intensity, matching crop logic.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=60,
        help="Number of normalized position bins in [0, 1].",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/eda/axial_voxel_profile",
        help="Directory to save figures and summaries.",
    )
    return parser.parse_args()


def subject_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_foreground_profile(
    cache_root: Path,
    axis_idx: int,
    threshold: float,
    desc: str | None = None,
) -> dict[str, np.ndarray]:
    cache_dir = cache_root / "_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"_cache directory not found under: {cache_root}")

    nii_paths = sorted(cache_dir.glob("*.nii.gz"))
    profiles: dict[str, np.ndarray] = {}
    for nii_path in tqdm(nii_paths, desc=desc or cache_root.name, unit="vol"):
        subject_id = subject_id_from_path(nii_path)
        data = np.asarray(nib.load(str(nii_path)).get_fdata(dtype=np.float32))
        mask = np.isfinite(data) & (np.abs(data) > threshold)
        if axis_idx == 0:
            counts = mask.sum(axis=(1, 2))
        elif axis_idx == 1:
            counts = mask.sum(axis=(0, 2))
        else:
            counts = mask.sum(axis=(0, 1))
        profiles[subject_id] = counts.astype(np.float32)
    if not profiles:
        raise RuntimeError(f"No NIfTI files found in {cache_dir}")
    return profiles


def aggregate_into_bins(profiles: dict[str, np.ndarray], num_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    all_points: list[tuple[int, float]] = []

    for values in profiles.values():
        n = int(values.shape[0])
        if n <= 1:
            positions = np.array([0.0], dtype=np.float32)
        else:
            positions = np.linspace(0.0, 1.0, n, dtype=np.float32)
        bin_indices = np.clip((positions * num_bins).astype(np.int64), 0, num_bins - 1)
        for bin_idx, voxel_count in zip(bin_indices, values, strict=True):
            sums[bin_idx] += float(voxel_count)
            counts[bin_idx] += 1
            all_points.append((int(bin_idx), float(voxel_count)))

    means = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)

    stds = np.zeros(num_bins, dtype=np.float64)
    if all_points:
        grouped: list[list[float]] = [[] for _ in range(num_bins)]
        for bin_idx, voxel_count in all_points:
            grouped[bin_idx].append(voxel_count)
        for idx, bucket in enumerate(grouped):
            if bucket:
                stds[idx] = float(np.std(np.asarray(bucket, dtype=np.float64)))
    return means, stds, counts.astype(np.float64)


def main() -> None:
    args = parse_args()
    if len(args.version_dirs) != len(args.labels):
        raise ValueError("--version_dirs and --labels must have the same length")

    axis_idx = AXIS_NAME_TO_INDEX[args.axis]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version_profiles: dict[str, dict[str, np.ndarray]] = {}
    for label, version_dir in zip(args.labels, args.version_dirs, strict=True):
        version_profiles[label] = load_foreground_profile(
            Path(version_dir),
            axis_idx,
            float(args.threshold),
            desc=f"Loading {label}",
        )

    common_ids = sorted(set.intersection(*(set(v.keys()) for v in version_profiles.values())))
    if not common_ids:
        raise RuntimeError("No common session IDs found across the provided versions")

    aligned_profiles = {
        label: {sid: version_profiles[label][sid] for sid in common_ids} for label in args.labels
    }

    bin_centers = (np.arange(args.num_bins, dtype=np.float64) + 0.5) / float(args.num_bins)
    summary: dict[str, object] = {
        "axis": args.axis,
        "axis_index": axis_idx,
        "threshold": float(args.threshold),
        "num_bins": int(args.num_bins),
        "num_common_sessions": len(common_ids),
        "versions": {},
    }

    csv_path = output_dir / f"{args.axis}_voxel_profile_bins.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["bin_center"]
        for label in args.labels:
            header.extend([f"{label}_mean", f"{label}_std", f"{label}_num_slices"])
        writer.writerow(header)

        aggregates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label in args.labels:
            aggregates[label] = aggregate_into_bins(aligned_profiles[label], args.num_bins)
            means, stds, counts = aggregates[label]
            summary["versions"][label] = {
                "peak_bin_center": float(bin_centers[int(np.argmax(means))]),
                "peak_mean_foreground_voxels": float(np.max(means)),
                "global_mean_foreground_voxels": float(np.mean(np.concatenate(list(aligned_profiles[label].values())))),
            }

        for i, center in enumerate(bin_centers):
            row = [float(center)]
            for label in args.labels:
                means, stds, counts = aggregates[label]
                row.extend([float(means[i]), float(stds[i]), int(counts[i])])
            writer.writerow(row)

    json_path = output_dir / f"{args.axis}_voxel_profile_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for label in args.labels:
        means, stds, counts = aggregate_into_bins(aligned_profiles[label], args.num_bins)
        axes[0].plot(bin_centers, means, label=label, linewidth=2)
        axes[0].fill_between(bin_centers, np.maximum(means - stds, 0), means + stds, alpha=0.15)
        axes[1].plot(bin_centers, counts, label=label, linewidth=2)

    axes[0].set_title(f"Foreground voxels per normalized {args.axis} slice")
    axes[0].set_xlabel(f"Normalized {args.axis} slice position")
    axes[0].set_ylabel("Mean foreground voxels per slice")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].set_title(f"Number of slices contributing to each {args.axis} bin")
    axes[1].set_xlabel(f"Normalized {args.axis} slice position")
    axes[1].set_ylabel("Number of slices")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig_path = output_dir / f"{args.axis}_voxel_profile.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure : {fig_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved summary: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
