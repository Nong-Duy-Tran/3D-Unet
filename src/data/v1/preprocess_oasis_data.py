"""
3D MRI Preprocessing Pipeline for OASIS (Alzheimer's Disease Classification)
==============================================================================

Two-stage design:
  Stage 1 - Cache: preprocess every subject exactly once.
    load .img → reorient RAS+ → N4 bias correction → skull-strip (HD-BET) →
    resample 1 mm iso → crop non-blank → .nii.gz
  Stage 2 - Assemble: populate fold dirs from cache via symlinks (no recomputation).

Split strategy: true nested CV via outer StratifiedKFold (each subject in test exactly once).
Duplicate handling: keep only *_MR1 scans.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_to_output
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None

# ---------------------------------------------------------------------------
# Default per-fold validation seeds (k = 5)
# ---------------------------------------------------------------------------
DEFAULT_FOLD_VAL_SEEDS: list[int] = [7, 13, 36, 11, 42]

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def parse_txt_metadata(txt_path: Path) -> float | None:
    """
    Extract CDR (Clinical Dementia Rating) from an OASIS subject .txt file.

    Returns:
        float  – CDR value (0.0 means healthy/young control)
        None   – CDR field absent (subject is skipped)
    """
    try:
        with open(txt_path, "r") as fh:
            for line in fh:
                if line.startswith("CDR:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            return 0.0  # empty field → young healthy control
                    return 0.0  # field present but no value
    except Exception as exc:
        print(f"  Warning: could not read {txt_path}: {exc}")
    return None


def find_all_oasis_subjects(oasis_root: str | Path, verbose: bool = False) -> list[dict]:
    """
    Walk all disc* sub-folders under *oasis_root* and collect valid subjects.

    Keeps only *_MR1 scans (MR2 and beyond are discarded).

    Returns:
        List of subject dicts: subject_id, subject_path, img_file,
                               cdr, label, class_name, disc
    """
    oasis_path = Path(oasis_root)
    disc_folders = sorted(
        d for d in oasis_path.iterdir() if d.is_dir() and d.name.startswith("disc")
    )
    print(f"\nFound {len(disc_folders)} disc folder(s)")

    stats = dict(total=0, no_txt=0, no_cdr=0, no_subj_path=0,
                 no_img=0, not_mr1=0, success=0)
    subjects: list[dict] = []

    for disc in disc_folders:
        print(f"\nProcessing {disc.name}…")
        subj_dirs = [d for d in disc.iterdir()
                     if d.is_dir() and d.name.startswith("OAS1_")]
        stats["total"] += len(subj_dirs)

        for subj_dir in tqdm(subj_dirs, desc=f"  {disc.name}"):
            sid = subj_dir.name

            # Keep MR1 only
            if "_MR1" not in sid:
                stats["not_mr1"] += 1
                if verbose:
                    print(f"    Skipping non-MR1: {sid}")
                continue

            # Metadata
            txt_file = subj_dir / f"{sid}.txt"
            if not txt_file.exists():
                stats["no_txt"] += 1
                if verbose:
                    print(f"    No .txt: {sid}")
                continue

            cdr = parse_txt_metadata(txt_file)
            if cdr is None:
                stats["no_cdr"] += 1
                if verbose:
                    print(f"    No CDR: {sid}")
                continue

            label = 0 if cdr == 0.0 else 1
            class_name = "normal" if label == 0 else "alzheimer"

            # Image file
            subj_111 = subj_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111"
            if not subj_111.exists():
                stats["no_subj_path"] += 1
                if verbose:
                    print(f"    No SUBJ_111: {sid}")
                continue

            img_file: Path | None = None
            for f in subj_111.iterdir():
                if f.name.endswith("sbj_111.img") or f.name.endswith("sbj_111.4dfp.img"):
                    img_file = f
                    break

            if img_file is None:
                stats["no_img"] += 1
                if verbose:
                    print(f"    No .img: {sid}")
                continue

            stats["success"] += 1
            subjects.append(
                dict(
                    subject_id=sid,
                    subject_path=str(subj_dir),
                    img_file=str(img_file),
                    cdr=cdr,
                    label=label,
                    class_name=class_name,
                    disc=disc.name,
                )
            )

    _width = 70
    print(f"\n{'=' * _width}")
    print("Scan summary")
    print(f"  Total subject directories : {stats['total']}")
    print(f"  Non-MR1 (skipped)         : {stats['not_mr1']}")
    print(f"  Missing .txt              : {stats['no_txt']}")
    print(f"  Missing CDR               : {stats['no_cdr']}")
    print(f"  Missing SUBJ_111          : {stats['no_subj_path']}")
    print(f"  Missing .img              : {stats['no_img']}")
    print(f"  Successfully loaded       : {stats['success']}")
    print(f"{'=' * _width}")

    return subjects


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def create_cv_splits(
    data: list[dict],
    n_folds: int = 5,
    val_ratio: float = 0.15,
    outer_seed: int = 42,
    fold_val_seeds: list[int] | None = None,
) -> list[tuple[list[dict], list[dict], list[dict]]]:
    """
    Build true-nested stratified k-fold splits.

    Outer loop  : StratifiedKFold → each subject in test exactly once.
    Inner split : stratified train_test_split per fold using a unique seed.

    Returns:
        List of (train, val, test) tuples, one per fold.
    """
    if fold_val_seeds is None:
        # Generate deterministic per-fold seeds when not supplied
        rng = np.random.default_rng(outer_seed)
        fold_val_seeds = rng.integers(1, 10_000, size=n_folds).tolist()
    elif len(fold_val_seeds) < n_folds:
        raise ValueError(
            f"fold_val_seeds has {len(fold_val_seeds)} entries but n_folds={n_folds}"
        )

    subjects = np.array(data)
    labels = np.array([s["label"] for s in data])
    n_total = len(data)

    _w = 70
    print(f"\n{'=' * _w}")
    print(f"Creating {n_folds}-Fold Stratified Cross-Validation Splits")
    print(f"  outer_seed={outer_seed}  val_ratio={val_ratio}")
    print(f"  fold_val_seeds={fold_val_seeds[:n_folds]}")
    print(f"{'=' * _w}")
    print(f"\nTotal : {n_total}")
    print(f"  Normal (CDR=0)   : {int((labels == 0).sum())}")
    print(f"  Alzheimer (CDR>0): {int((labels == 1).sum())}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
    cv_splits: list[tuple[list[dict], list[dict], list[dict]]] = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(subjects, labels)):
        trainval = subjects[trainval_idx]
        trainval_labels = labels[trainval_idx]
        test_data: list[dict] = subjects[test_idx].tolist()

        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval)),
            test_size=val_ratio,
            stratify=trainval_labels,
            random_state=fold_val_seeds[fold_idx],
        )

        train_data: list[dict] = trainval[tr_idx].tolist()
        val_data: list[dict] = trainval[val_idx].tolist()

        tl = labels[trainval_idx][tr_idx]
        vl = labels[trainval_idx][val_idx]
        el = labels[test_idx]

        print(f"\nFold {fold_idx}  (val_seed={fold_val_seeds[fold_idx]})")
        print(f"  Train {len(train_data):>4}  N={int((tl==0).sum())} A={int((tl==1).sum())}")
        print(f"  Val   {len(val_data):>4}  N={int((vl==0).sum())} A={int((vl==1).sum())}")
        print(f"  Test  {len(test_data):>4}  N={int((el==0).sum())} A={int((el==1).sum())}")

        cv_splits.append((train_data, val_data, test_data))

    print(f"\n{'=' * _w}")
    print(f"All {n_folds} folds created – each subject in test exactly once.")
    print(f"{'=' * _w}")
    return cv_splits


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def _find_hdbet() -> str:
    """Locate the hd-bet executable; raise RuntimeError if not found."""
    cmd = shutil.which("hd-bet")
    if cmd:
        return cmd
    # Common conda environment paths as fallback
    candidates = [
        "/home/ntq/miniconda3/envs/3dunet/bin/hd-bet",
        os.path.expanduser("~/miniconda3/envs/3dunet/bin/hd-bet"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError(
        "hd-bet not found. Install HD-BET or disable --skull_strip."
    )


def skull_strip(img: nib.Nifti1Image, subject_id: str, device: str = "cpu") -> nib.Nifti1Image:
    """
    Apply HD-BET skull stripping.

    Configuration: no fast mode, no test-time augmentation (TTA).

    Args:
        img         : nibabel image (already loaded)
        subject_id  : used for logging
        device      : 'cpu' or 'cuda'

    Returns:
        Brain-masked nibabel image, or original if HD-BET fails.
    """
    try:
        hd_bet_bin = _find_hdbet()
    except RuntimeError as exc:
        print(f"  Warning: {exc}")
        return img

    try:
        with tempfile.TemporaryDirectory(prefix="hdbet_") as tmpdir:
            in_path = os.path.join(tmpdir, f"{subject_id}_input.nii.gz")
            out_path = os.path.join(tmpdir, f"{subject_id}_bet.nii.gz")

            nib.save(img, in_path)

            cmd = [
                hd_bet_bin,
                "-i", in_path,
                "-o", out_path,
                "-device", device,
                "--disable_tta",   # no TTA per spec
                # no -mode fast    # no fast mode per spec
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"  Warning: HD-BET failed for {subject_id}: {result.stderr[:300]}")
                return img

            if not os.path.exists(out_path):
                print(f"  Warning: HD-BET produced no output for {subject_id}")
                return img

            # Materialise data before tmpdir is cleaned up (nibabel lazy-loads)
            bet = nib.load(out_path)
            data = bet.get_fdata()
            return nib.Nifti1Image(data, bet.affine, bet.header)

    except subprocess.TimeoutExpired:
        print(f"  Warning: HD-BET timed out for {subject_id}")
        return img
    except Exception as exc:
        print(f"  Warning: HD-BET error for {subject_id}: {exc}")
        return img


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return the image reoriented to RAS+ canonical orientation."""
    return nib.as_closest_canonical(img)


def n4_bias_correct(
    img: nib.Nifti1Image,
    subject_id: str,
    shrink_factor: int = 4,
    num_iterations: tuple[int, ...] = (50, 50, 30, 20),
) -> nib.Nifti1Image:
    """Apply N4 bias correction with an Otsu mask using SimpleITK."""
    if sitk is None:
        raise RuntimeError("SimpleITK is required for N4 bias correction.")

    shrink_factor = max(1, int(shrink_factor))

    with tempfile.TemporaryDirectory(prefix="n4_") as tmpdir:
        in_path = os.path.join(tmpdir, f"{subject_id}_input.nii.gz")
        out_path = os.path.join(tmpdir, f"{subject_id}_n4.nii.gz")
        nib.save(img, in_path)

        sitk_img = sitk.ReadImage(in_path, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(list(num_iterations))

        if shrink_factor > 1:
            shrink = [shrink_factor] * sitk_img.GetDimension()
            sitk_img_small = sitk.Shrink(sitk_img, shrink)
            mask_small = sitk.Shrink(mask, shrink)
            corrector.Execute(sitk_img_small, mask_small)
            log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
            corrected = sitk_img / sitk.Exp(log_bias)
        else:
            corrected = corrector.Execute(sitk_img, mask)

        sitk.WriteImage(corrected, out_path)
        corrected_nib = nib.load(out_path)
        data = corrected_nib.get_fdata(dtype=np.float32)
        return nib.Nifti1Image(data, corrected_nib.affine, corrected_nib.header)


def resample_isotropic(img: nib.Nifti1Image, voxel_mm: float = 1.0) -> nib.Nifti1Image:
    """
    Resample image to isotropic *voxel_mm* mm resolution using trilinear interpolation.
    Skips resampling if already at target resolution (within 1 µm tolerance).
    """
    zooms = img.header.get_zooms()[:3]
    if all(abs(z - voxel_mm) <= 1e-3 for z in zooms):
        return img
    return resample_to_output(
        img,
        voxel_sizes=(voxel_mm, voxel_mm, voxel_mm),
        order=1,
    )


def crop_nonblank_3d(
    img: nib.Nifti1Image,
    threshold: float = 1e-5,
) -> nib.Nifti1Image:
    """
    Crop the volume to the tight 3D bounding box of non-blank voxels.

    A voxel is considered non-blank if its absolute value exceeds *threshold*.
    The affine is updated so world coordinates remain consistent.

    Args:
        img       : nibabel image
        threshold : intensity threshold for "non-blank"

    Returns:
        Cropped nibabel image (same affine origin shifted to crop corner).
    """
    data = img.get_fdata()
    mask = np.abs(data) > threshold

    if not mask.any():
        return img  # all-zero volume; return unchanged

    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1  # +1 → exclusive upper bound

    cropped = data[x_min:x_max, y_min:y_max, z_min:z_max]

    # Shift affine origin to the crop corner
    new_origin = img.affine[:3, :3] @ np.array([x_min, y_min, z_min]) + img.affine[:3, 3]
    new_affine = img.affine.copy()
    new_affine[:3, 3] = new_origin

    return nib.Nifti1Image(cropped, new_affine, img.header)


def preprocess_volume(
    img_path: Path,
    subject_id: str,
    apply_skull_strip: bool = True,
    hdbet_device: str = "cpu",
) -> nib.Nifti1Image:
    """
    Full 3D preprocessing pipeline for a single volume:
      1. Load Analyze (.img) file
      2. Reorient to RAS+ canonical (first normalization step)
      3. N4 bias correction
      4. Skull stripping via HD-BET (if enabled)
      5. Resample to isotropic 1 mm
      6. Crop to non-blank 3D bounding box

    Returns a preprocessed nibabel image ready to save as .nii.gz.
    """
    # 1. Load
    img: nib.Nifti1Image = nib.load(str(img_path))

    # Handle 4-D volumes (keep first volume)
    if img.ndim == 4:
        img = nib.Nifti1Image(img.get_fdata()[..., 0], img.affine, img.header)

    # 2. Reorient to RAS+ first (unifies axis order across datasets)
    img = reorient_to_ras(img)

    # 3. N4 bias correction before skull stripping
    img = n4_bias_correct(img, subject_id)

    # 4. Skull stripping
    if apply_skull_strip:
        img = skull_strip(img, subject_id, device=hdbet_device)

    # 5. Resample to 1 mm isotropic
    img = resample_isotropic(img, voxel_mm=1.0)

    # 6. Crop non-blank 3D
    img = crop_nonblank_3d(img)

    return img


# ---------------------------------------------------------------------------
# Stage 1 & 2
# ---------------------------------------------------------------------------

def preprocess_all_to_cache(
    all_subjects: list[dict],
    cache_dir: Path,
    apply_skull_strip: bool,
    hdbet_device: str,
) -> dict[str, Path]:
    """
    Preprocess every subject exactly once and save to *cache_dir*.
    Already-cached subjects are skipped (resumable).
    Returns {subject_id: cached_nii_path}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Path] = {}
    skipped = 0

    print(f"\n{'=' * 70}")
    print(f"Stage 1: preprocessing {len(all_subjects)} subjects → {cache_dir}")
    print(f"{'=' * 70}")

    for item in tqdm(all_subjects, desc="  cache"):
        sid = item["subject_id"]
        out_path = cache_dir / f"{sid}.nii.gz"

        if out_path.exists():
            cached[sid] = out_path
            skipped += 1
            continue

        try:
            processed = preprocess_volume(
                Path(item["img_file"]),
                sid,
                apply_skull_strip=apply_skull_strip,
                hdbet_device=hdbet_device,
            )
            nib.save(processed, str(out_path))
            cached[sid] = out_path
        except Exception as exc:
            print(f"\n  Error – {sid}: {exc}")

    print(f"\n  Processed: {len(cached) - skipped}  Already cached: {skipped}  "
          f"Failed: {len(all_subjects) - len(cached)}")
    return cached


def assemble_splits_from_cache(
    cv_splits: list[tuple[list[dict], list[dict], list[dict]]],
    cache_map: dict[str, Path],
    output_dir: Path,
    use_symlinks: bool = True,
) -> dict:
    """
    Stage 2: build fold directories from cached files.
    Uses symlinks by default; falls back to copies when *use_symlinks=False*.
    Returns a counts summary per fold.
    """
    mode = "symlinks" if use_symlinks else "copies"
    print(f"\n{'=' * 70}")
    print(f"Stage 2: assembling fold directories ({mode})")
    print(f"{'=' * 70}")

    fold_counts: dict = {}

    for fold_idx, (train_data, val_data, test_data) in enumerate(cv_splits):
        fold_dir = output_dir / f"fold_{fold_idx}"
        print(f"\n  Fold {fold_idx}")
        fold_counts[f"fold_{fold_idx}"] = {}

        for split_name, split_data in [("train", train_data),
                                        ("val",   val_data),
                                        ("test",  test_data)]:
            counts: dict[str, int] = {"normal": 0, "alzheimer": 0, "missing": 0}

            for item in split_data:
                sid = item["subject_id"]
                class_name = item["class_name"]
                dest_dir = fold_dir / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{sid}.nii.gz"

                if dest.exists():
                    counts[class_name] += 1
                    continue

                src = cache_map.get(sid)
                if src is None or not src.exists():
                    counts["missing"] += 1
                    print(f"    Warning: cache missing for {sid}")
                    continue

                if use_symlinks:
                    dest.symlink_to(src.resolve())
                else:
                    shutil.copy2(str(src), str(dest))

                counts[class_name] += 1

            print(f"    {split_name:5s}: N={counts['normal']} "
                  f"A={counts['alzheimer']} missing={counts['missing']}")
            fold_counts[f"fold_{fold_idx}"][split_name] = counts

    return fold_counts


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def save_fold_metadata(
    cv_splits: list[tuple[list[dict], list[dict], list[dict]]],
    output_dir: Path,
) -> dict:
    """
    For each fold save:
      fold_<k>_train.json / .csv
      fold_<k>_val.json   / .csv
      fold_<k>_test.json  / .csv
      summary.json
      cdr_distribution.json
    """
    info_dir = output_dir / "split_info"
    info_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_folds": len(cv_splits), "stratified": True,
                     "inner_validation": True, "folds": {}}
    cdr_stats: dict = {}

    for fold_idx, (train_data, val_data, test_data) in enumerate(cv_splits):
        fold_key = f"fold_{fold_idx}"

        for split_name, split_data in [("train", train_data),
                                        ("val",   val_data),
                                        ("test",  test_data)]:
            base = info_dir / f"{fold_key}_{split_name}"
            base.with_suffix(".json").write_text(
                json.dumps(split_data, indent=2), encoding="utf-8"
            )
            pd.DataFrame(split_data).to_csv(str(base) + ".csv", index=False)

        def _stats(d: list[dict]) -> dict:
            return {
                "total":     len(d),
                "normal":    sum(1 for x in d if x["label"] == 0),
                "alzheimer": sum(1 for x in d if x["label"] == 1),
            }

        summary["folds"][fold_key] = {
            "train": _stats(train_data),
            "val":   _stats(val_data),
            "test":  _stats(test_data),
        }

        cdr_stats[fold_key] = {"train": {}, "val": {}, "test": {}}
        for split_name, split_data in [("train", train_data),
                                        ("val",   val_data),
                                        ("test",  test_data)]:
            for item in split_data:
                cdr = str(item["cdr"])
                cdr_stats[fold_key][split_name][cdr] = (
                    cdr_stats[fold_key][split_name].get(cdr, 0) + 1
                )

    (info_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (info_dir / "cdr_distribution.json").write_text(
        json.dumps(cdr_stats, indent=2), encoding="utf-8"
    )

    print(f"\nSplit metadata saved → {info_dir}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("OASIS 3D MRI Preprocessing Pipeline")
    print("=" * 70)

    oasis_dir = Path(args.oasis_dir)
    output_dir = Path(args.output_dir)

    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")

    print(f"\nOASIS directory  : {oasis_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Number of folds  : {args.n_folds}")
    print(f"Val ratio        : {args.val_ratio}")
    print(f"Outer seed       : {args.seed}")
    print("N4 bias correction: ENABLED")
    print(f"Skull stripping  : {'ENABLED (device=' + args.hdbet_device + ')' if args.skull_strip else 'DISABLED'}")
    print("Output orientation: RAS canonical")

    # Resolve per-fold val seeds
    if args.fold_val_seeds:
        fold_val_seeds: list[int] = [int(s) for s in args.fold_val_seeds.split(",")]
    else:
        if args.n_folds == 5:
            fold_val_seeds = DEFAULT_FOLD_VAL_SEEDS
        else:
            rng = np.random.default_rng(args.seed)
            fold_val_seeds = rng.integers(1, 10_000, size=args.n_folds).tolist()

    print(f"Fold val seeds   : {fold_val_seeds[:args.n_folds]}")

    # ------------------------------------------------------------------ scan
    print("\n" + "-" * 70)
    print("Scanning OASIS subjects (MR1 only)…")
    print("-" * 70)

    all_subjects = find_all_oasis_subjects(oasis_dir, verbose=args.verbose)

    if not all_subjects:
        print("\nNo subjects found – check OASIS directory structure.")
        return

    n_total = len(all_subjects)
    n_normal = sum(1 for s in all_subjects if s["label"] == 0)
    n_alz = sum(1 for s in all_subjects if s["label"] == 1)
    print(f"\nTotal valid subjects : {n_total}")
    print(f"  Normal (CDR=0)     : {n_normal}")
    print(f"  Alzheimer (CDR>0)  : {n_alz}")

    cdr_counts: dict[float, int] = {}
    for s in all_subjects:
        cdr_counts[s["cdr"]] = cdr_counts.get(s["cdr"], 0) + 1
    print("\nCDR distribution:")
    for cdr in sorted(cdr_counts):
        print(f"  CDR {cdr}: {cdr_counts[cdr]} subject(s)")

    # --------------------------------------------------------------- splits
    print("\n" + "-" * 70)
    print("Building stratified CV splits…")
    print("-" * 70)

    cv_splits = create_cv_splits(
        all_subjects,
        n_folds=args.n_folds,
        val_ratio=args.val_ratio,
        outer_seed=args.seed,
        fold_val_seeds=fold_val_seeds,
    )

    if args.dry_run:
        print("\n" + "=" * 70)
        print("Dry run – no files written. Remove --dry_run to process.")
        print("=" * 70)
        save_fold_metadata(cv_splits, output_dir)
        return

    # ------------------------------------------------------------ process
    cache_dir = output_dir / "_cache"

    cache_map = preprocess_all_to_cache(
        all_subjects,
        cache_dir=cache_dir,
        apply_skull_strip=args.skull_strip,
        hdbet_device=args.hdbet_device,
    )

    assemble_splits_from_cache(
        cv_splits,
        cache_map=cache_map,
        output_dir=output_dir,
        use_symlinks=not args.no_symlinks,
    )

    # ---------------------------------------------------------- metadata
    summary = save_fold_metadata(cv_splits, output_dir)

    # ---------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nDirectory structure:")
    for fold_idx in range(args.n_folds):
        fk = f"fold_{fold_idx}"
        fs = summary["folds"][fk]
        print(f"\n  fold_{fold_idx}/")
        for split_name in ("train", "val", "test"):
            st = fs[split_name]
            print(f"    {split_name}/")
            print(f"      normal/     ({st['normal']} files)")
            print(f"      alzheimer/  ({st['alzheimer']} files)")
    print(f"\n  split_info/")
    print(f"    fold_{{k}}_train.json, fold_{{k}}_val.json, fold_{{k}}_test.json")
    print(f"    fold_{{k}}_train.csv,  fold_{{k}}_val.csv,  fold_{{k}}_test.csv")
    print(f"    summary.json")
    print(f"    cdr_distribution.json")

    print("\n" + "=" * 70)
    print("Next steps")
    print("=" * 70)
    print("\n1. Train each fold:")
    print("   bash script/train_cv5_fold.sh 0  # fold 0")
    print("   bash script/train_cv5_fold.sh 1  # fold 1")
    print("   ...")
    print("\n2. Evaluate all folds:")
    print("   bash script/evaluate_cv5_all.sh")
    print("\n3. Aggregate results:")
    print("   python 3d/aggregate_cv5_results.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OASIS 3D MRI data with stratified k-fold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--oasis_dir", type=str, default="./data/OASIS",
        help="Root OASIS directory containing disc1, disc2, … sub-folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/processed_oasis_3d_cv5",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of outer cross-validation folds",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Fraction of trainval pool used as validation per fold",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the outer StratifiedKFold",
    )
    parser.add_argument(
        "--fold_val_seeds", type=str, default=None,
        help=(
            "Comma-separated per-fold val-split seeds "
            "(e.g. '7,13,36,11,42'). "
            "Defaults to [7,13,36,11,42] for n_folds=5."
        ),
    )
    parser.add_argument(
        "--skull_strip", action="store_true", default=False,
        help="Apply HD-BET skull stripping (no fast mode, no TTA)",
    )
    parser.add_argument(
        "--hdbet_device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for HD-BET",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Create split metadata only; skip file conversion",
    )
    parser.add_argument(
        "--no_symlinks", action="store_true", default=False,
        help="Copy cached files into fold dirs instead of symlinking (needed across filesystems)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print reasons for each skipped subject",
    )

    main(parser.parse_args())
