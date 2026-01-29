"""
Convert ADNI NIfTI files to per-scan HDF5 files for classification.

Default CSV columns:
- path: path to .nii/.nii.gz (absolute or relative to --images-dir)
- Group: diagnosis label (e.g., CN, AD, MCI)
- split: optional train/val/test column

If the CSV has no path column, provide --adni-root or --source-root-map and
--image-id-col (e.g., Image Data ID like I48617) to resolve files under ADNI.

Supports multiple CSVs, source-level label overrides, and global train/val/test
splitting after pooling records. By default, splitting is subject-level.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from tqdm import tqdm

from src.helpers.adni_defaults import (
    DEFAULT_CSVS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_LABEL_MAP,
    DEFAULT_SOURCE_ROOT_MAP,
    DEFAULT_SPLIT_RATIOS,
)
from src.helpers.adni_hdf5 import (
    build_id_index,
    load_label_map,
    load_nifti,
    normalize_key,
    normalize_volume,
    parse_int_map,
    parse_ratio_list,
    parse_str_map,
    resolve_nifti_path,
    sanitize_token,
    source_key_from_path,
    split_subjects,
    write_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADNI NIfTI files to HDF5")
    parser.add_argument("--csv", action="append", default=None, help="CSV with file paths and labels")
    parser.add_argument("--images-dir", default=None, help="Base dir for relative paths")
    parser.add_argument("--adni-root", default=None, help="Root dir to search for ADNI NIfTI files")
    parser.add_argument(
        "--source-root-map",
        default="",
        help="Map source->root, e.g. abb=data/ADNI/abb/ADNI",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output root for HDF5 files",
    )
    parser.add_argument("--path-col", default="path", help="CSV column for NIfTI path")
    parser.add_argument("--image-id-col", default="Image Data ID", help="CSV column for Image Data ID")
    parser.add_argument("--label-col", default="Group", help="CSV column for label")
    parser.add_argument("--subject-col", default="Subject", help="CSV column for subject ID")
    parser.add_argument("--id-col", default=None, help="CSV column for unique ID (optional)")
    parser.add_argument("--split-col", default=None, help="CSV column for split (optional)")
    parser.add_argument("--split", default="train", help="Default split name when split-col missing")
    parser.add_argument("--split-map", default="", help="Map split values, e.g. Train=train,Validation=val")
    parser.add_argument(
        "--split-ratios",
        default=DEFAULT_SPLIT_RATIOS,
        help="Global ratios, e.g. train=0.8,val=0.1,test=0.1",
    )
    parser.add_argument(
        "--no-split-by-subject",
        dest="split_by_subject",
        action="store_false",
        help="Disable subject-level split; split by scan instead.",
    )
    parser.set_defaults(split_by_subject=True)
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for split")
    parser.add_argument("--label-map", default="", help="Map labels, e.g. CN=0,AD=1,MCI=2")
    parser.add_argument("--label-map-json", default=None, help="Path to JSON label map")
    parser.add_argument(
        "--normalize-mode",
        default="zscore",
        help="Normalization: minmax | zscore | none",
    )
    parser.add_argument(
        "--normalize-eps",
        type=float,
        default=1e-6,
        help="Epsilon for normalization stability",
    )
    parser.add_argument(
        "--no-reorient-ras",
        dest="reorient_ras",
        action="store_false",
        help="Disable reorientation to RAS+ before saving.",
    )
    parser.set_defaults(reorient_ras=True)
    parser.add_argument(
        "--source-label-map",
        default=DEFAULT_SOURCE_LABEL_MAP,
        help="Map source->label, e.g. abb=1,bbc=0,ecc=0",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .h5 files")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs only")

    args = parser.parse_args()

    if args.csv is None:
        args.csv = [str(path) for path in DEFAULT_CSVS]
    csv_paths = [Path(path) for path in args.csv]

    assert len(csv_paths) == 3, "Expected exactly 3 CSV files (abb, bbc, ecc)."
    expected_csvs = [path.resolve() for path in DEFAULT_CSVS]
    provided_csvs = [path.resolve() for path in csv_paths]
    assert set(provided_csvs) == set(expected_csvs), (
        "CSV list must match the default ABB/BBC/ECC files. "
        "Use those exact paths or omit --csv."
    )
    for path in expected_csvs:
        assert path.exists(), f"Missing CSV file: {path}"

    images_dir = Path(args.images_dir) if args.images_dir else None
    adni_root = Path(args.adni_root) if args.adni_root else None
    output_dir = Path(args.output_dir)
    split_map = parse_str_map(args.split_map) if args.split_map else {}
    split_ratios = parse_ratio_list(args.split_ratios) if args.split_ratios else []
    label_map = load_label_map(args.label_map, args.label_map_json)
    source_label_map = parse_int_map(args.source_label_map) if args.source_label_map else {}
    source_root_map = {k: Path(v) for k, v in parse_str_map(args.source_root_map).items()}
    if not source_root_map:
        source_root_map = DEFAULT_SOURCE_ROOT_MAP.copy()

    if split_ratios:
        ratio_sum = sum(ratio for _, ratio in split_ratios)
        if ratio_sum <= 0 or ratio_sum > 1.0 + 1e-6:
            raise ValueError("split-ratios must sum to (0, 1]")
        if ratio_sum < 1.0 - 1e-6:
            names = {name for name, _ in split_ratios}
            if "train" in names:
                split_ratios = [
                    (name, ratio + (1.0 - ratio_sum) if name == "train" else ratio)
                    for name, ratio in split_ratios
                ]
            else:
                split_ratios.append(("train", 1.0 - ratio_sum))

    id_index_cache: dict[Path, dict[str, Path]] = {}
    duplicate_cache: dict[Path, set[str]] = {}

    records: list[dict[str, str | int | Path]] = []
    total = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    info_counts: dict[str, int] = {}
    subject_labels: dict[str, int] = {}
    conflict_subjects: set[str] = set()

    for csv_path in expected_csvs:
        source_key = source_key_from_path(csv_path)
        source_label_override = source_label_map.get(source_key)

        source_root = source_root_map.get(source_key, adni_root)
        id_index: dict[str, Path] = {}
        duplicate_ids: set[str] = set()
        if source_root:
            if source_root not in id_index_cache:
                print(f"Indexing ADNI files under: {source_root}")
                id_index_cache[source_root], duplicate_cache[source_root] = build_id_index(source_root)
                print(
                    f"Indexed {len(id_index_cache[source_root])} files "
                    f"(duplicates: {len(duplicate_cache[source_root])})"
                )
            id_index = id_index_cache[source_root]
            duplicate_ids = duplicate_cache[source_root]

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if args.label_col not in reader.fieldnames:
                raise ValueError(
                    f"CSV {csv_path} must contain label column {args.label_col!r}; "
                    f"found {reader.fieldnames}"
                )
            has_path_col = args.path_col in reader.fieldnames
            has_id_col = args.image_id_col in reader.fieldnames
            if not has_path_col and not has_id_col:
                raise ValueError(
                    f"CSV {csv_path} must contain either {args.path_col!r} or {args.image_id_col!r}; "
                    f"found {reader.fieldnames}"
                )
            if not has_path_col and not source_root:
                raise ValueError(
                    f"CSV {csv_path} has no path column; provide --adni-root or --source-root-map."
                )

            for row in tqdm(reader, desc=f"Reading {csv_path.name}"):
                if args.limit is not None and total >= args.limit:
                    break
                total += 1

                path_value = row.get(args.path_col, "").strip() if has_path_col else ""
                label_raw = row.get(args.label_col, "").strip()
                subject_value = (
                    row.get(args.subject_col, "").strip()
                    if args.subject_col in reader.fieldnames
                    else ""
                )
                subject_id = normalize_key(subject_value) if subject_value else ""

                if not label_raw and source_label_override is None:
                    skipped += 1
                    skip_reasons["missing_label"] = skip_reasons.get("missing_label", 0) + 1
                    continue

                if source_label_override is not None:
                    label = int(source_label_override)
                    label_token = "ad" if label == 1 else "not_ad"
                else:
                    label_key = normalize_key(label_raw)
                    if label_key in label_map:
                        label = label_map[label_key]
                    elif label_raw.isdigit():
                        label = int(label_raw)
                    else:
                        skipped += 1
                        skip_reasons["label_unmapped"] = skip_reasons.get("label_unmapped", 0) + 1
                        continue
                    label_token = sanitize_token(label_raw)

                if subject_id:
                    existing = subject_labels.get(subject_id)
                    if existing is None:
                        subject_labels[subject_id] = label
                    elif existing != label:
                        conflict_subjects.add(subject_id)

                split_value = args.split
                if args.split_col:
                    split_value = row.get(args.split_col, args.split).strip() or args.split
                if split_map:
                    split_value = split_map.get(normalize_key(split_value), split_value)

                image_id_value = row.get(args.image_id_col, "").strip() if has_id_col else ""
                nifti_path, resolve_reason = resolve_nifti_path(
                    path_value=path_value,
                    image_id_value=image_id_value,
                    images_dir=images_dir,
                    id_index=id_index,
                    duplicate_ids=duplicate_ids,
                )

                if nifti_path is None:
                    skipped += 1
                    reason = resolve_reason or "nifti_resolve_failed"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue
                if not nifti_path.exists():
                    skipped += 1
                    skip_reasons["nifti_missing"] = skip_reasons.get("nifti_missing", 0) + 1
                    continue

                if args.id_col and row.get(args.id_col):
                    base_id = sanitize_token(row[args.id_col])
                elif image_id_value:
                    base_id = sanitize_token(image_id_value)
                else:
                    base_id = nifti_path.name
                    if base_id.endswith(".nii.gz"):
                        base_id = base_id[:-7]
                    elif base_id.endswith(".nii"):
                        base_id = base_id[:-4]

                records.append(
                    {
                        "nifti_path": nifti_path,
                        "label": label,
                        "label_raw": label_raw or label_token,
                        "label_token": label_token,
                        "split": split_value,
                        "base_id": base_id,
                        "subject_id": subject_id,
                        "source_csv": str(csv_path),
                        "source_key": source_key,
                    }
                )

    if conflict_subjects:
        before = len(records)
        records = [rec for rec in records if rec["subject_id"] not in conflict_subjects]
        removed = before - len(records)
        if removed:
            skipped += removed
            skip_reasons["subject_label_conflict"] = (
                skip_reasons.get("subject_label_conflict", 0) + removed
            )

    if split_ratios:
        if args.split_by_subject and subject_labels:
            subject_to_split: dict[str, str] = {}
            subjects_by_label: dict[int, set[str]] = {}
            for rec in records:
                subject_id = rec.get("subject_id", "")
                if not subject_id:
                    continue
                label = int(rec["label"])
                subjects_by_label.setdefault(label, set()).add(subject_id)

            for label, subject_ids in subjects_by_label.items():
                if not subject_ids:
                    continue
                subject_to_split.update(
                    split_subjects(sorted(subject_ids), split_ratios, args.seed + label)
                )

            missing_subject = 0
            for rec in records:
                subject_id = rec.get("subject_id", "")
                if subject_id and subject_id in subject_to_split:
                    rec["split"] = subject_to_split[subject_id]
                else:
                    missing_subject += 1
            if missing_subject:
                info_counts["missing_subject_id"] = (
                    info_counts.get("missing_subject_id", 0) + missing_subject
                )
        else:
            rng = random.Random(args.seed)
            rng.shuffle(records)
            total_records = len(records)
            counts: list[tuple[str, int]] = []
            remaining = total_records
            for idx, (name, ratio) in enumerate(split_ratios):
                if idx == len(split_ratios) - 1:
                    count = remaining
                else:
                    count = int(total_records * ratio)
                counts.append((name, count))
                remaining -= count
            cursor = 0
            for split_name, count in counts:
                for rec in records[cursor : cursor + count]:
                    rec["split"] = split_name
                cursor += count

    name_counts: dict[str, int] = {}
    written = 0

    for rec in tqdm(records, desc="Writing"):
        nifti_path = rec["nifti_path"]
        label = int(rec["label"])
        label_token = rec["label_token"]
        split_value = str(rec["split"])
        base_id = str(rec["base_id"])

        base_name = f"{split_value}_{label_token}_{base_id}"
        suffix = name_counts.get(base_name, 0)
        name_counts[base_name] = suffix + 1
        if suffix:
            base_name = f"{base_name}_{suffix}"

        output_path = output_dir / split_value / f"{base_name}.h5"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            skip_reasons["output_exists"] = skip_reasons.get("output_exists", 0) + 1
            continue

        if args.dry_run:
            print(f"{nifti_path} -> {output_path} (label={label})")
            written += 1
            continue

        img_data = load_nifti(nifti_path, reorient_ras=args.reorient_ras)
        img_data = normalize_volume(img_data, args.normalize_mode, args.normalize_eps)
        meta = {
            "source_path": str(nifti_path),
            "label_raw": str(rec["label_raw"]),
            "split": split_value,
            "source_csv": str(rec["source_csv"]),
            "source_key": str(rec["source_key"]),
        }
        meta["normalize_mode"] = str(args.normalize_mode)
        meta["reorient_ras"] = str(bool(args.reorient_ras))
        if rec.get("subject_id"):
            meta["subject_id"] = str(rec["subject_id"])
        write_hdf5(output_path, img_data, label, meta)
        written += 1

    print(f"Done. Total rows: {total}, written: {written}, skipped: {skipped}")
    if skip_reasons:
        print("Skip breakdown:")
        for reason, count in sorted(skip_reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason}: {count}")
    if info_counts:
        print("Info:")
        for reason, count in sorted(info_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {reason}: {count}")
    for root, duplicates in duplicate_cache.items():
        if duplicates:
            print(f"Warning: skipped {len(duplicates)} Image Data IDs due to duplicates under {root}.")


if __name__ == "__main__":
    main()
