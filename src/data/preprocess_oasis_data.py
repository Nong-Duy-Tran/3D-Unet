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
    split_by_subject,
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

        output_dir = Path(args.output_dir)

        if args.export_2d_jpg:
            train_count = convert_and_copy_subjects_jpg(
                train_data,
                output_dir,
                'train',
                image_size=args.image_size,
                central_slices=args.central_slices,
                reorient_ras=args.reorient_ras,
                resample_mm=args.resample_mm,
            )
            val_count = convert_and_copy_subjects_jpg(
                val_data,
                output_dir,
                'val',
                image_size=args.image_size,
                central_slices=args.central_slices,
                reorient_ras=args.reorient_ras,
                resample_mm=args.resample_mm,
            )
            test_count = convert_and_copy_subjects_jpg(
                test_data,
                output_dir,
                'test',
                image_size=args.image_size,
                central_slices=args.central_slices,
                reorient_ras=args.reorient_ras,
                resample_mm=args.resample_mm,
            )
        else:
            train_count = convert_and_copy_subjects(train_data, output_dir, 'train')
            val_count = convert_and_copy_subjects(val_data, output_dir, 'val')
            test_count = convert_and_copy_subjects(test_data, output_dir, 'test')

        summary = save_split_info(train_data, val_data, test_data, output_dir)

        print("\n" + "=" * 70)
        print("Dataset preprocessing completed!")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
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
        help='Number of central axial slices to export (JPG mode)'
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
        '--verbose',
        action='store_true',
        help='Print detailed information about skipped subjects'
    )

    args = parser.parse_args()
    main(args)
