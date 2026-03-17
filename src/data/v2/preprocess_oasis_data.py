"""
3D MRI preprocessing pipeline for OASIS v2.

Version policy:
  - v1: voxel-wise nonblank crop
  - v2: slice-energy crop

Stage 1 - Cache:
  load .img -> reorient RAS+ -> N4 bias correction -> optional skull-strip ->
  resample 1 mm iso -> crop by slice energy -> .nii.gz

Stage 2 - Split metadata:
  create standard 5-fold CV metadata only (train/test) using StratifiedGroupKFold.
  No per-fold 3D directories are materialized; folds are managed by JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_to_output
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.oasis_labels import VALID_LABEL_MODES, build_label, get_class_names

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None

# ---------------------------------------------------------------------------
# Default per-fold validation seeds (k = 5)
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


def _patient_id_from_session_id(session_id: str) -> str:
    parts = session_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return session_id


def _visit_id_from_session_id(session_id: str) -> str:
    parts = session_id.split("_")
    if parts and parts[-1].startswith("MR"):
        return parts[-1]
    return "UNKNOWN"


def find_all_oasis_sessions(
    oasis_root: str | Path,
    label_mode: str = "normal_vs_abnormal",
    verbose: bool = False,
) -> list[dict]:
    """
    Walk all disc* sub-folders under *oasis_root* and collect valid subjects.

    Returns:
        List of session dicts with patient/session metadata and averaged image path.
    """
    oasis_path = Path(oasis_root)
    disc_folders = sorted(
        d for d in oasis_path.iterdir() if d.is_dir() and d.name.startswith("disc")
    )
    print(f"\nFound {len(disc_folders)} disc folder(s)")

    stats = dict(total=0, no_txt=0, no_cdr=0, no_subj_path=0, no_img=0, dropped_label=0, success=0)
    sessions: list[dict] = []

    for disc in disc_folders:
        print(f"\nProcessing {disc.name}…")
        subj_dirs = [d for d in disc.iterdir()
                     if d.is_dir() and d.name.startswith("OAS1_")]
        stats["total"] += len(subj_dirs)

        for subj_dir in tqdm(subj_dirs, desc=f"  {disc.name}"):
            session_id = subj_dir.name
            patient_id = _patient_id_from_session_id(session_id)
            visit_id = _visit_id_from_session_id(session_id)

            # Metadata
            txt_file = subj_dir / f"{session_id}.txt"
            if not txt_file.exists():
                stats["no_txt"] += 1
                if verbose:
                    print(f"    No .txt: {session_id}")
                continue

            cdr = parse_txt_metadata(txt_file)
            if cdr is None:
                stats["no_cdr"] += 1
                if verbose:
                    print(f"    No CDR: {session_id}")
                continue

            label_info = build_label(cdr, label_mode)
            if label_info is None:
                stats["dropped_label"] += 1
                if verbose:
                    print(f"    Dropped by label_mode={label_mode}: {session_id} (CDR={cdr})")
                continue
            label, class_name = label_info

            # Image file
            subj_111 = subj_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111"
            if not subj_111.exists():
                stats["no_subj_path"] += 1
                if verbose:
                    print(f"    No SUBJ_111: {session_id}")
                continue

            img_file: Path | None = None
            for f in subj_111.iterdir():
                if f.name.endswith("sbj_111.img") or f.name.endswith("sbj_111.4dfp.img"):
                    img_file = f
                    break

            if img_file is None:
                stats["no_img"] += 1
                if verbose:
                    print(f"    No .img: {session_id}")
                continue

            stats["success"] += 1
            sessions.append(
                dict(
                    patient_id=patient_id,
                    session_id=session_id,
                    visit_id=visit_id,
                    subject_path=str(subj_dir),
                    image_path=str(img_file),
                    cdr=cdr,
                    label_raw_cdr=cdr,
                    label_mode=label_mode,
                    label=label,
                    class_name=class_name,
                    is_mr1=(visit_id == "MR1"),
                    is_averaged_image=True,
                    group_id=patient_id,
                    disc=disc.name,
                )
            )

    _width = 70
    print(f"\n{'=' * _width}")
    print("Scan summary")
    print(f"  Total subject directories : {stats['total']}")
    print(f"  Missing .txt              : {stats['no_txt']}")
    print(f"  Missing CDR               : {stats['no_cdr']}")
    print(f"  Missing SUBJ_111          : {stats['no_subj_path']}")
    print(f"  Missing .img              : {stats['no_img']}")
    print(f"  Dropped by label mode     : {stats['dropped_label']}")
    print(f"  Successfully loaded       : {stats['success']}")
    print(f"{'=' * _width}")

    return sessions


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def create_cv_splits(
    data: list[dict],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """
    Build standard stratified grouped k-fold splits (train/test only).

    Grouping rule:
      group_id = patient_id

    This keeps all sessions/visits from the same patient in the same fold.

    Returns a list of (train, test) tuples.
    """
    samples = np.array(data, dtype=object)
    labels = np.array([s["label"] for s in data], dtype=np.int64)
    groups = np.array([s["group_id"] for s in data], dtype=object)
    patients = np.array([s["patient_id"] for s in data], dtype=object)
    n_total = len(samples)

    _w = 70
    print(f"\n{'=' * _w}")
    print(f"Creating {n_folds}-Fold Stratified Group Cross-Validation Splits")
    print(f"  seed={seed}  test_ratio={1.0 / n_folds:.2f}")
    print(f"{'=' * _w}")
    class_names = sorted({s["class_name"] for s in data}, key=lambda name: min(s["label"] for s in data if s["class_name"] == name))
    print(f"\nTotal sessions : {n_total}")
    print("  Label counts : " + " ".join(
        f"{name}={sum(1 for s in data if s['class_name'] == name)}" for name in class_names
    ))
    print(f"  Patients     : {len(set(patients.tolist()))}")
    print(f"  Groups        : {len(set(groups.tolist()))}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits: list[tuple[list[dict], list[dict]]] = []

    seen_test_sessions: set[str] = set()
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(samples, labels, groups)):
        train_data: list[dict] = samples[train_idx].tolist()
        test_data: list[dict] = samples[test_idx].tolist()

        train_session_ids = {x["session_id"] for x in train_data}
        test_session_ids = {x["session_id"] for x in test_data}
        overlap = train_session_ids & test_session_ids
        if overlap:
            raise RuntimeError(f"Fold {fold_idx}: session overlap detected: {sorted(overlap)[:5]}")

        repeated_test = seen_test_sessions & test_session_ids
        if repeated_test:
            raise RuntimeError(f"Fold {fold_idx}: repeated test sessions detected: {sorted(repeated_test)[:5]}")
        seen_test_sessions |= test_session_ids

        train_groups = {x["group_id"] for x in train_data}
        test_groups = {x["group_id"] for x in test_data}
        group_overlap = train_groups & test_groups
        if group_overlap:
            raise RuntimeError(f"Fold {fold_idx}: group overlap detected: {sorted(group_overlap)[:5]}")

        print(f"\nFold {fold_idx}")
        print(
            f"  Train {len(train_data):>4}  "
            + " ".join(f"{name}={sum(1 for x in train_data if x['class_name'] == name)}" for name in class_names)
        )
        print(
            f"  Test  {len(test_data):>4}  "
            + " ".join(f"{name}={sum(1 for x in test_data if x['class_name'] == name)}" for name in class_names)
        )
        print(f"  Train groups={len(train_groups)}  Test groups={len(test_groups)}")

        cv_splits.append((train_data, test_data))

    print(f"\n{'=' * _w}")
    print(f"All {n_folds} folds created – each session in test exactly once.")
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


def crop_by_slice_energy_3d(
    img: nib.Nifti1Image,
    energy_ratio: float = 1e-3,
    min_energy: float = 1e-6,
) -> nib.Nifti1Image:
    """
    Crop volume by selecting the tightest 3D box whose slices carry enough
    aggregate energy along each axis.

    This is more robust to isolated noisy voxels than voxel-wise nonblank crop,
    because a few outlier voxels outside the brain contribute very little slice
    energy and are therefore ignored.

    Args:
        img: nibabel image
        energy_ratio: keep slices whose energy >= energy_ratio * max_axis_energy
        min_energy: absolute floor to avoid zero thresholds on degenerate inputs
    """
    data = np.asarray(img.get_fdata(dtype=np.float32))
    energy_ratio = float(max(0.0, energy_ratio))
    min_energy = float(max(0.0, min_energy))

    abs_data = np.abs(data)
    if not np.isfinite(abs_data).any():
        return img
    abs_data = np.where(np.isfinite(abs_data), abs_data, 0.0)

    def _bounds(energies: np.ndarray) -> tuple[int, int]:
        if energies.size == 0:
            return 0, 0
        max_energy = float(np.max(energies))
        threshold = max(min_energy, energy_ratio * max_energy)
        keep = np.flatnonzero(energies >= threshold)
        if keep.size == 0:
            return 0, energies.shape[0]
        return int(keep[0]), int(keep[-1]) + 1

    x_min, x_max = _bounds(abs_data.sum(axis=(1, 2)))
    y_min, y_max = _bounds(abs_data.sum(axis=(0, 2)))
    z_min, z_max = _bounds(abs_data.sum(axis=(0, 1)))

    cropped = data[x_min:x_max, y_min:y_max, z_min:z_max]

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
      6. Crop by slice-energy 3D bounding box

    Returns a preprocessed nibabel image ready to save as .nii.gz.
    """
    # 1. Load
    img: nib.Nifti1Image = nib.load(str(img_path))

    # OASIS processed inputs are typically stored as (H, W, D, 1).
    if img.ndim == 4 and img.shape[-1] == 1:
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

    # 6. Crop by slice energy to avoid noisy outlier voxels expanding the box
    img = crop_by_slice_energy_3d(img)

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
    Preprocess every session exactly once and save to *cache_dir*.
    Already-cached sessions are skipped (resumable).
    Returns {session_id: cached_nii_path}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, Path] = {}
    skipped = 0

    print(f"\n{'=' * 70}")
    print(f"Stage 1: preprocessing {len(all_subjects)} sessions → {cache_dir}")
    print(f"{'=' * 70}")

    for item in tqdm(all_subjects, desc="  cache"):
        session_id = item["session_id"]
        out_path = cache_dir / f"{session_id}.nii.gz"

        if out_path.exists():
            cached[session_id] = out_path
            skipped += 1
            continue

        try:
            processed = preprocess_volume(
                Path(item["image_path"]),
                session_id,
                apply_skull_strip=apply_skull_strip,
                hdbet_device=hdbet_device,
            )
            nib.save(processed, str(out_path))
            cached[session_id] = out_path
        except Exception as exc:
            print(f"\n  Error – {session_id}: {exc}")

    print(f"\n  Processed: {len(cached) - skipped}  Already cached: {skipped}  "
          f"Failed: {len(all_subjects) - len(cached)}")
    return cached


def save_fold_metadata(
    cv_splits: list[tuple[list[dict], list[dict]]],
    output_dir: Path,
) -> dict:
    """
    Save one JSON-driven split manifest plus summaries.
    """
    info_dir = output_dir / "split_info"
    info_dir.mkdir(parents=True, exist_ok=True)

    label_mode = cv_splits[0][0][0]["label_mode"] if cv_splits and cv_splits[0][0] else None
    class_names = sorted(
        {item["class_name"] for pair in cv_splits for split in pair for item in split},
        key=lambda name: min(item["label"] for pair in cv_splits for split in pair for item in split if item["class_name"] == name),
    )
    folds_json: dict = {
        "n_folds": len(cv_splits),
        "split_strategy": "StratifiedGroupKFold",
        "label_mode": label_mode,
        "class_names": class_names,
        "folds": [],
    }
    summary: dict = {"n_folds": len(cv_splits), "stratified": True, "grouped": True, "folds": {}}
    cdr_stats: dict = {}

    for fold_idx, (train_data, test_data) in enumerate(cv_splits):
        fold_key = f"fold_{fold_idx}"
        fold_entry = {"fold_index": fold_idx, "train": [], "test": []}
        for split_name, split_data in [("train", train_data), ("test", test_data)]:
            split_rows = []
            for item in split_data:
                row = dict(item)
                row["fold"] = fold_idx
                row["split"] = split_name
                split_rows.append(row)
            fold_entry[split_name] = split_rows
        folds_json["folds"].append(fold_entry)

        def _stats(d: list[dict]) -> dict:
            counts = {name: 0 for name in class_names}
            for item in d:
                counts[item["class_name"]] += 1
            return {"total": len(d), "counts": counts}

        summary["folds"][fold_key] = {
            "train": _stats(train_data),
            "test":  _stats(test_data),
        }

        cdr_stats[fold_key] = {"train": {}, "test": {}}
        for split_name, split_data in [("train", train_data), ("test", test_data)]:
            for item in split_data:
                cdr = str(item["cdr"])
                cdr_stats[fold_key][split_name][cdr] = (
                    cdr_stats[fold_key][split_name].get(cdr, 0) + 1
                )

    (info_dir / "folds.json").write_text(
        json.dumps(folds_json, indent=2), encoding="utf-8"
    )
    (info_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (info_dir / "cdr_distribution.json").write_text(
        json.dumps(cdr_stats, indent=2), encoding="utf-8"
    )

    all_rows = []
    for fold in folds_json["folds"]:
        for split_name in ("train", "test"):
            all_rows.extend(fold[split_name])
    pd.DataFrame(all_rows).to_csv(info_dir / "folds.csv", index=False)

    print(f"\nSplit metadata saved → {info_dir}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("OASIS 3D MRI Preprocessing Pipeline (v2: energy crop)")
    print("=" * 70)

    oasis_dir = Path(args.oasis_dir)
    output_dir = Path(args.output_dir)

    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")

    print(f"\nOASIS directory  : {oasis_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Number of folds  : {args.n_folds}")
    print(f"Seed             : {args.seed}")
    print(f"Label mode       : {args.label_mode}")
    print("N4 bias correction: ENABLED")
    print(f"Skull stripping  : {'ENABLED (device=' + args.hdbet_device + ')' if args.skull_strip else 'DISABLED'}")
    print("Output orientation: RAS canonical")

    # ------------------------------------------------------------------ scan
    print("\n" + "-" * 70)
    print("Scanning OASIS sessions (MR1 + MR2 when available)…")
    print("-" * 70)

    all_subjects = find_all_oasis_sessions(oasis_dir, label_mode=args.label_mode, verbose=args.verbose)

    if not all_subjects:
        print("\nNo subjects found – check OASIS directory structure.")
        return

    n_total = len(all_subjects)
    n_patients = len({s["patient_id"] for s in all_subjects})
    class_names = get_class_names(args.label_mode)
    print(f"\nTotal valid sessions : {n_total}")
    print(f"  Unique patients    : {n_patients}")
    print("  Label counts       : " + " ".join(
        f"{name}={sum(1 for s in all_subjects if s['class_name'] == name)}" for name in class_names
    ))
    print(f"  MR1 sessions       : {sum(1 for s in all_subjects if s['is_mr1'])}")
    print(f"  MR2+ sessions      : {sum(1 for s in all_subjects if not s['is_mr1'])}")

    cdr_counts: dict[float, int] = {}
    for s in all_subjects:
        cdr_counts[s["cdr"]] = cdr_counts.get(s["cdr"], 0) + 1
    print("\nCDR distribution:")
    for cdr in sorted(cdr_counts):
        print(f"  CDR {cdr}: {cdr_counts[cdr]} subject(s)")

    # --------------------------------------------------------------- splits
    print("\n" + "-" * 70)
    print("Building stratified grouped CV splits…")
    print("-" * 70)

    cv_splits = create_cv_splits(
        all_subjects,
        n_folds=args.n_folds,
        seed=args.seed,
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

    # ---------------------------------------------------------- metadata
    summary = save_fold_metadata(cv_splits, output_dir)

    # ---------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nDirectory structure:")
    print("  _cache/")
    print("    <session_id>.nii.gz")
    print("  split_info/")
    print("    folds.json")
    print("    folds.csv")
    print("    summary.json")
    print("    cdr_distribution.json")
    for fold_idx in range(args.n_folds):
        fk = f"fold_{fold_idx}"
        fs = summary["folds"][fk]
        print(f"\n  {fk}:")
        for split_name in ("train", "test"):
            st = fs[split_name]
            print(
                f"    {split_name}: "
                + " ".join(f"{name}={st['counts'].get(name, 0)}" for name in class_names)
                + f" total={st['total']}"
            )

    print("\n" + "=" * 70)
    print("Next steps")
    print("=" * 70)
    print("\n1. Export 2D slices from cache + split_info:")
    print("   python src/data/v2/postprocess_oasis_data.py --input_dir <output_dir> --output_dir <2d_output_dir>")
    print("\n2. Point training configs to fold_{k}/train and fold_{k}/test.")
    print("   If fold_{k}/val does not exist, the training code will create an internal validation split from train/.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OASIS 3D MRI data with grouped 5-fold CV metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--oasis_dir", type=str, default="./data/OASIS",
        help="Root OASIS directory containing disc1, disc2, … sub-folders",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/processed_oasis_3d_cv5_v2",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for StratifiedGroupKFold",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="normal_vs_abnormal",
        choices=VALID_LABEL_MODES,
        help="CDR-to-label mapping used for split metadata and exported class folders",
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
        "--verbose", action="store_true",
        help="Print reasons for each skipped subject",
    )

    main(parser.parse_args())
