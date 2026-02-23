"""
Data preprocessing script for OASIS Alzheimer's Disease dataset
Processes all OASIS discs and creates train/validation/test splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.helpers.oasis_metadata import (  # noqa: E402
    find_all_oasis_subjects,
    save_split_info,
    save_kfold_info,
    split_by_subject,
    make_kfold_splits,
)
from src.modules.oasis_preprocess import (  # noqa: E402
    convert_and_copy_subjects,
    convert_and_copy_subjects_jpg,
)


def main(args):
    print("=" * 70)
    print("OASIS Alzheimer's Disease Dataset Preprocessing")
    print("=" * 70)

    oasis_dir = Path(args.oasis_dir)

    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")

    print(f"\nOASIS directory: {oasis_dir}")

    print("\n" + "-" * 70)
    print("Scanning for OASIS subjects...")
    print("-" * 70)

    all_subjects = find_all_oasis_subjects(oasis_dir, verbose=args.verbose)

    if len(all_subjects) == 0:
        print("\nNo subjects found! Check your OASIS directory structure.")
        return

    print(f"\n\nTotal valid subjects found: {len(all_subjects)}")
    print(f"  Normal (CDR=0): {sum(1 for s in all_subjects if s['label'] == 0)}")
    print(f"  Alzheimer (CDR>0): {sum(1 for s in all_subjects if s['label'] == 1)}")

    cdr_counts = {}
    for s in all_subjects:
        cdr = s['cdr']
        cdr_counts[cdr] = cdr_counts.get(cdr, 0) + 1

    print("\nCDR Distribution:")
    for cdr in sorted(cdr_counts.keys()):
        print(f"  CDR {cdr}: {cdr_counts[cdr]} subjects")

    if args.k_folds and args.k_folds > 1 and not args.single_dir:
        print("\nNote: --k_folds enabled; forcing --single_dir to avoid data duplication.")
        args.single_dir = True

    output_dir = Path(args.output_dir)

    if args.k_folds and args.k_folds > 1:
        print("\n" + "-" * 70)
        print(f"Creating stratified {args.k_folds}-fold splits (subject-level)...")
        print("-" * 70)
        fold_seed = args.fold_seed if args.fold_seed is not None else args.seed
        folds, test_data = make_kfold_splits(
            all_subjects,
            k_folds=int(args.k_folds),
            test_ratio=float(args.test_ratio),
            val_ratio=float(args.fold_val_ratio),
            random_seed=int(fold_seed),
        )

        print(f"\nTest set: {len(test_data)} subjects")
        print(f"  Normal: {sum(1 for x in test_data if x['label'] == 0)}")
        print(f"  Alzheimer: {sum(1 for x in test_data if x['label'] == 1)}")
        for fold in folds:
            train_data = fold['train']
            val_data = fold['val']
            print(f"\nFold {fold['fold']}:")
            print(f"  Train: {len(train_data)} (N={sum(1 for x in train_data if x['label'] == 0)}, "
                  f"A={sum(1 for x in train_data if x['label'] == 1)})")
            print(f"  Val:   {len(val_data)} (N={sum(1 for x in val_data if x['label'] == 0)}, "
                  f"A={sum(1 for x in val_data if x['label'] == 1)})")
    else:
        print("\n" + "-" * 70)
        print("Splitting data by subject...")
        print("-" * 70)

        train_data, val_data, test_data = split_by_subject(
            all_subjects,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )

        print(f"\nTrain set: {len(train_data)} subjects")
        print(f"  Normal: {sum(1 for x in train_data if x['label'] == 0)}")
        print(f"  Alzheimer: {sum(1 for x in train_data if x['label'] == 1)}")

        print(f"\nValidation set: {len(val_data)} subjects")
        print(f"  Normal: {sum(1 for x in val_data if x['label'] == 0)}")
        print(f"  Alzheimer: {sum(1 for x in val_data if x['label'] == 1)}")

        print(f"\nTest set: {len(test_data)} subjects")
        print(f"  Normal: {sum(1 for x in test_data if x['label'] == 0)}")
        print(f"  Alzheimer: {sum(1 for x in test_data if x['label'] == 1)}")

    if not args.dry_run:
        print("\n" + "-" * 70)
        print("Converting and copying files to output directory...")
        print("-" * 70)

        if args.single_dir:
            split_name = "all"
            if args.export_2d_jpg:
                all_count = convert_and_copy_subjects_jpg(
                    all_subjects,
                    output_dir,
                    split_name,
                    image_size=args.image_size,
                    central_slices=args.central_slices,
                    reorient_ras=args.reorient_ras,
                    resample_mm=args.resample_mm,
                    slice_axis=args.slice_axis,
                    rotate_k=args.rotate_k,
                    crop_to_nonblank=args.crop_to_nonblank,
                    nonblank_intensity_threshold=args.nonblank_intensity_threshold,
                    nonblank_min_ratio=args.nonblank_min_ratio,
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
            else:
                all_count = convert_and_copy_subjects(
                    all_subjects,
                    output_dir,
                    split_name,
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
        else:
            if args.export_2d_jpg:
                train_count = convert_and_copy_subjects_jpg(
                    train_data,
                    output_dir,
                    'train',
                    image_size=args.image_size,
                    central_slices=args.central_slices,
                    reorient_ras=args.reorient_ras,
                    resample_mm=args.resample_mm,
                    slice_axis=args.slice_axis,
                    rotate_k=args.rotate_k,
                    crop_to_nonblank=args.crop_to_nonblank,
                    nonblank_intensity_threshold=args.nonblank_intensity_threshold,
                    nonblank_min_ratio=args.nonblank_min_ratio,
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
                val_count = convert_and_copy_subjects_jpg(
                    val_data,
                    output_dir,
                    'val',
                    image_size=args.image_size,
                    central_slices=args.central_slices,
                    reorient_ras=args.reorient_ras,
                    resample_mm=args.resample_mm,
                    slice_axis=args.slice_axis,
                    rotate_k=args.rotate_k,
                    crop_to_nonblank=args.crop_to_nonblank,
                    nonblank_intensity_threshold=args.nonblank_intensity_threshold,
                    nonblank_min_ratio=args.nonblank_min_ratio,
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
                test_count = convert_and_copy_subjects_jpg(
                    test_data,
                    output_dir,
                    'test',
                    image_size=args.image_size,
                    central_slices=args.central_slices,
                    reorient_ras=args.reorient_ras,
                    resample_mm=args.resample_mm,
                    slice_axis=args.slice_axis,
                    rotate_k=args.rotate_k,
                    crop_to_nonblank=args.crop_to_nonblank,
                    nonblank_intensity_threshold=args.nonblank_intensity_threshold,
                    nonblank_min_ratio=args.nonblank_min_ratio,
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
            else:
                train_count = convert_and_copy_subjects(
                    train_data,
                    output_dir,
                    'train',
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
                val_count = convert_and_copy_subjects(
                    val_data,
                    output_dir,
                    'val',
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )
                test_count = convert_and_copy_subjects(
                    test_data,
                    output_dir,
                    'test',
                    hdbet=args.hdbet,
                    hdbet_device=args.hdbet_device,
                    hdbet_fast=args.hdbet_fast,
                    hdbet_tta=args.hdbet_tta,
                )

        if args.k_folds and args.k_folds > 1:
            save_kfold_info(folds, test_data, output_dir, random_seed=fold_seed)
        elif not args.single_dir:
            save_split_info(train_data, val_data, test_data, output_dir)
        else:
            save_split_info(train_data, val_data, test_data, output_dir)

        print("\n" + "=" * 70)
        print("Dataset preprocessing completed!")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        if args.single_dir:
            print("\nDataset structure:")
            print("  all/")
            print(f"    normal/     ({all_count['normal']} files)")
            print(f"    alzheimer/  ({all_count['alzheimer']} files)")
            print("  split_info/")
            if args.k_folds and args.k_folds > 1:
                print("    folds.json")
                print("    folds_summary.json")
                if test_data:
                    print("    test_split.json")
                    print("    test_split.csv")
            else:
                print("    train_split.json")
                print("    val_split.json")
                print("    test_split.json")
                print("    summary.json")
                print("    cdr_distribution.json")
            if all_count['errors'] > 0:
                print(f"\nWarning: {all_count['errors']} subjects failed to convert")
        else:
            print("\nDataset structure:")
            print("  train/")
            print(f"    normal/     ({train_count['normal']} files)")
            print(f"    alzheimer/  ({train_count['alzheimer']} files)")
            print("  val/")
            print(f"    normal/     ({val_count['normal']} files)")
            print(f"    alzheimer/  ({val_count['alzheimer']} files)")
            print("  test/")
            print(f"    normal/     ({test_count['normal']} files)")
            print(f"    alzheimer/  ({test_count['alzheimer']} files)")
            print("  split_info/")
            print("    train_split.json")
            print("    val_split.json")
            print("    test_split.json")
            print("    summary.json")
            print("    cdr_distribution.json")

            if train_count['errors'] + val_count['errors'] + test_count['errors'] > 0:
                total_errors = train_count['errors'] + val_count['errors'] + test_count['errors']
                print(f"\nWarning: {total_errors} subjects failed to convert")
    else:
        print("\n" + "=" * 70)
        print("Dry run completed - no files were converted/copied")
        print("Remove --dry_run flag to process files")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess OASIS dataset for Alzheimer classification'
    )

    parser.add_argument(
        '--oasis_dir',
        type=str,
        default='./data/OASIS',
        help='Path to OASIS directory containing disc1, disc2, etc.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed_oasis',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--k_folds',
        type=int,
        default=0,
        help='Number of stratified folds to create (subject-level). Set >1 to enable.'
    )
    parser.add_argument(
        '--single_dir',
        action='store_true',
        help='Store all data under output_dir/all (no train/val/test folders)'
    )
    parser.add_argument(
        '--fold_seed',
        type=int,
        default=None,
        help='Seed for k-fold splitting (default: use --seed)'
    )
    parser.add_argument(
        '--fold_val_ratio',
        type=float,
        default=0.1,
        help='Validation ratio within trainval pool for each fold (default: 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run without converting/copying files (for testing)'
    )

    parser.add_argument(
        '--export_2d_jpg',
        action='store_true',
        help='Export 2D JPG slices instead of NIfTI volumes'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Output JPG size (square). Used when --export_2d_jpg is set'
    )

    parser.add_argument(
        '--central_slices',
        type=int,
        default=120,
        help='Number of central slices to export (JPG mode). Set <=0 to export all slices in the axis.'
    )
    parser.add_argument(
        '--slice_axis',
        type=int,
        default=2,
        help='Slice axis for JPG export: 0, 1, or 2 (default: 2)'
    )
    parser.add_argument(
        '--rotate_k',
        type=int,
        default=0,
        help='Rotate slices by 90*k degrees (default: 0). Set 0 to disable.'
    )
    parser.add_argument(
        '--crop_to_nonblank',
        action='store_true',
        help='Crop exported JPG stack to first/last non-blank slice'
    )
    parser.add_argument(
        '--nonblank_intensity_threshold',
        type=int,
        default=5,
        help='Pixel intensity threshold used for non-blank detection'
    )
    parser.add_argument(
        '--nonblank_min_ratio',
        type=float,
        default=0.001,
        help='Minimum ratio of pixels above threshold to count slice as non-blank'
    )

    parser.add_argument(
        '--resample_mm',
        type=float,
        default=1.0,
        help='Resample voxel size in mm (JPG mode). Set <= 0 to disable'
    )

    parser.add_argument(
        '--reorient_ras',
        action='store_true',
        help='Reorient to RAS+ before slicing (JPG mode)'
    )
    parser.add_argument(
        '--no_reorient_ras',
        dest='reorient_ras',
        action='store_false',
        help='Disable reorientation to RAS+ (JPG mode)'
    )
    parser.set_defaults(reorient_ras=True)

    parser.add_argument(
        '--hdbet',
        action='store_true',
        help='Apply HD-BET skull stripping before conversion/slicing'
    )
    parser.add_argument(
        '--hdbet_device',
        default='cpu',
        help='HD-BET device (cpu or cuda)'
    )
    parser.add_argument(
        '--hdbet_fast',
        action='store_true',
        help='Use HD-BET fast mode'
    )
    parser.add_argument(
        '--hdbet_tta',
        action='store_true',
        help='Use HD-BET test-time augmentation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about skipped subjects'
    )

    args = parser.parse_args()
    main(args)
