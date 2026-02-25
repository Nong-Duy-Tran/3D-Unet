"""
Data preprocessing script for OASIS with 5-Fold Cross-Validation
Creates 5 independent train/val/test splits for nested cross-validation

This implements the outer loop of nested CV:
- Each fold has 64% train / 16% val / 20% test (80/20 outer, then inner 80/20)
- All 416 subjects will be tested exactly once across the 5 folds
- Stratified splitting maintains class balance
- Subject-level splitting prevents data leakage
- Inner validation split for early stopping

Note: OASIS SUBJ_111 images are NOT skull-stripped (fill ratio ~39%).
Use --skull_strip to apply HD-BET and remove skull/eyes/neck tissue.
"""
import os
import json
import shutil
import subprocess
import tempfile
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import random
import nibabel as nib
import numpy as np


def parse_txt_metadata(txt_path):
    """
    Extract CDR (Clinical Dementia Rating) from the OASIS text file
    
    Args:
        txt_path: Path to subject metadata text file
    
    Returns:
        float CDR or None
        CDR = 0: Normal/Healthy
        CDR > 0: Various stages of dementia/Alzheimer's
        None (empty CDR): Treated as 0 (young healthy controls)
    """
    cdr = None
    cdr_found = False
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('CDR:'):
                    cdr_found = True
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            cdr = float(parts[1])
                        except ValueError:
                            # CDR field exists but empty -> treat as 0 (young healthy control)
                            cdr = 0.0
                    else:
                        # CDR field exists but empty -> treat as 0 (young healthy control)
                        cdr = 0.0
                    break
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return cdr


def find_all_oasis_subjects(oasis_root, verbose=False):
    """
    Recursively find all OASIS subjects across all discs
    
    Args:
        oasis_root: Root OASIS directory containing disc1, disc2, etc.
        verbose: Print detailed information about skipped subjects
    
    Returns:
        List of (subject_path, subject_id, cdr, label) tuples
    """
    subjects = []
    oasis_path = Path(oasis_root)
    
    # Process all disc folders
    disc_folders = sorted([d for d in oasis_path.iterdir() if d.is_dir() and d.name.startswith('disc')])
    
    print(f"\nFound {len(disc_folders)} disc folders")
    
    # Track statistics
    stats = {
        'total_dirs': 0,
        'no_txt': 0,
        'no_cdr': 0,
        'no_subj_path': 0,
        'no_img_file': 0,
        'success': 0
    }
    
    for disc_folder in disc_folders:
        print(f"\nProcessing {disc_folder.name}...")
        
        # Find all subject directories in this disc
        subject_dirs = [d for d in disc_folder.iterdir() if d.is_dir() and d.name.startswith('OAS1_')]
        stats['total_dirs'] += len(subject_dirs)
        
        for subject_dir in tqdm(subject_dirs, desc=f"  {disc_folder.name}"):
            subject_id = subject_dir.name
            
            # 1. Find and parse metadata
            txt_file = subject_dir / f"{subject_id}.txt"
            if not txt_file.exists():
                stats['no_txt'] += 1
                if verbose:
                    print(f"  No .txt file: {subject_id}")
                continue
            
            cdr = parse_txt_metadata(txt_file)
            
            # Skip subjects without CDR information
            if cdr is None:
                stats['no_cdr'] += 1
                if verbose:
                    print(f"  No CDR found: {subject_id}")
                continue
            
            # Determine label: CDR = 0 -> Normal, CDR > 0 -> Alzheimer's
            label = 0 if cdr == 0.0 else 1
            class_name = 'normal' if label == 0 else 'alzheimer'
            
            # 2. Find the subject registered brain image
            subj_path = subject_dir / 'PROCESSED' / 'MPRAGE' / 'SUBJ_111'
            
            if not subj_path.exists():
                stats['no_subj_path'] += 1
                if verbose:
                    print(f"  No SUBJ_111 path: {subject_id}")
                continue
            
            # Find the subject image file
            img_file = None
            for f in subj_path.iterdir():
                if f.name.endswith('sbj_111.img') or f.name.endswith('sbj_111.4dfp.img'):
                    img_file = f
                    break
            
            if not img_file:
                stats['no_img_file'] += 1
                if verbose:
                    print(f"  No masked image: {subject_id}")
                continue
            
            stats['success'] += 1
            subjects.append({
                'subject_id': subject_id,
                'subject_path': str(subject_dir),
                'img_file': str(img_file),
                'cdr': cdr,
                'label': label,
                'class_name': class_name,
                'disc': disc_folder.name
            })
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Scan Summary:")
    print(f"  Total subject directories: {stats['total_dirs']}")
    print(f"  Successfully processed: {stats['success']}")
    print(f"\nSkipped:")
    print(f"  Missing .txt file: {stats['no_txt']}")
    print(f"  Missing CDR in .txt: {stats['no_cdr']}")
    print(f"  Missing SUBJ_111 folder: {stats['no_subj_path']}")
    print(f"  Missing subject image: {stats['no_img_file']}")
    print(f"{'='*70}")
    
    # Remove duplicate subjects (keep only first scan, usually MR1)
    unique_subjects = {}
    for subj in subjects:
        # Extract base subject ID (without _MR1, _MR2, etc.)
        base_id = subj['subject_id'].rsplit('_', 1)[0]
        
        if base_id not in unique_subjects:
            unique_subjects[base_id] = subj
        else:
            # Keep MR1 over MR2 if available
            if '_MR1' in subj['subject_id']:
                unique_subjects[base_id] = subj
    
    subjects = list(unique_subjects.values())
    
    if len(unique_subjects) < stats['success']:
        print(f"\nNote: Found {stats['success'] - len(unique_subjects)} duplicate scans.")
        print(f"      Keeping one scan per subject: {len(subjects)} unique subjects")
    
    return subjects


def create_stratified_cv_splits(data, n_folds=5, val_ratio=0.2, random_seed=42):
    """
    Create stratified K-fold splits for cross-validation with inner validation split
    Ensures balanced class distribution across all folds
    
    For each fold:
    - Outer test set: 20% of full dataset
    - From remaining 80%:
      - Training set: 80% (64% of full dataset)
      - Validation set: 20% (16% of full dataset)
    
    Args:
        data: List of subject dictionaries
        n_folds: Number of folds
        val_ratio: Ratio of training data to use for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (train_data, val_data, test_data) tuples for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Convert to arrays for sklearn
    subjects = np.array(data)
    labels = np.array([s['label'] for s in data])
    
    print(f"\n{'='*70}")
    print(f"Creating {n_folds}-Fold Stratified Cross-Validation Splits")
    print(f"with Inner Validation Split (val_ratio={val_ratio})")
    print(f"{'='*70}")
    print(f"\nTotal subjects: {len(data)}")
    print(f"  Normal (CDR=0): {sum(labels == 0)}")
    print(f"  Alzheimer (CDR>0): {sum(labels == 1)}")
    print(f"\nClass ratio: {sum(labels == 0)}/{sum(labels == 1)} = {sum(labels == 0)/sum(labels == 1):.2f}:1")
    
    # Create stratified K-fold splitter for outer loop
    outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Generate splits
    cv_splits = []
    for fold_idx, (trainval_indices, test_indices) in enumerate(outer_skf.split(subjects, labels)):
        # Get train+val and test data
        trainval_data = subjects[trainval_indices]
        trainval_labels = labels[trainval_indices]
        test_data = subjects[test_indices].tolist()
        test_labels = labels[test_indices]
        
        # Further split train+val into train and val using stratification
        # Use a different seed for each fold to ensure different val splits
        inner_seed = random_seed + fold_idx
        train_indices_inner, val_indices_inner = train_test_split(
            np.arange(len(trainval_data)),
            test_size=val_ratio,
            stratify=trainval_labels,
            random_state=inner_seed
        )
        
        train_data = trainval_data[train_indices_inner].tolist()
        val_data = trainval_data[val_indices_inner].tolist()
        
        train_labels_inner = trainval_labels[train_indices_inner]
        val_labels_inner = trainval_labels[val_indices_inner]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_data)} subjects (Normal: {sum(train_labels_inner == 0)}, Alzheimer: {sum(train_labels_inner == 1)})")
        print(f"  Val:   {len(val_data)} subjects (Normal: {sum(val_labels_inner == 0)}, Alzheimer: {sum(val_labels_inner == 1)})")
        print(f"  Test:  {len(test_data)} subjects (Normal: {sum(test_labels == 0)}, Alzheimer: {sum(test_labels == 1)})")
        print(f"  Train ratio: {sum(train_labels_inner == 0)}/{sum(train_labels_inner == 1)} = {sum(train_labels_inner == 0)/sum(train_labels_inner == 1):.2f}:1")
        print(f"  Val ratio:   {sum(val_labels_inner == 0)}/{sum(val_labels_inner == 1)} = {sum(val_labels_inner == 0)/sum(val_labels_inner == 1):.2f}:1")
        print(f"  Test ratio:  {sum(test_labels == 0)}/{sum(test_labels == 1)} = {sum(test_labels == 0)/sum(test_labels == 1):.2f}:1")
        
        cv_splits.append((train_data, val_data, test_data))
    
    print(f"\n{'='*70}")
    print(f"✓ All {n_folds} folds maintain similar class distributions!")
    print(f"{'='*70}")
    
    return cv_splits


def skull_strip_with_hdbet(img, subject_id, device='cpu'):
    """
    Apply HD-BET skull stripping to remove skull, eyes, and neck tissue.

    OASIS SUBJ_111 images are NOT skull-stripped (fill ratio ~39%).
    HD-BET removes non-brain tissue, leaving only brain parenchyma (~15-20% fill).

    Args:
        img: nibabel image
        subject_id: Subject ID for logging
        device: 'cuda' or 'cpu'

    Returns:
        nibabel image with skull stripped, or original img if HD-BET fails
    """
    hd_bet_bin = '/home/ntq/miniconda3/envs/3dunet/bin/hd-bet'
    if not os.path.exists(hd_bet_bin):
        print(f"  Warning: hd-bet not found at {hd_bet_bin}, skipping skull stripping")
        return img

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, f'{subject_id}_input.nii.gz')
            out_path = os.path.join(tmpdir, f'{subject_id}_bet.nii.gz')

            nib.save(img, in_path)

            cmd = [
                hd_bet_bin,
                '-i', in_path,
                '-o', out_path,
                '-device', device,
                '--disable_tta',
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                print(f"  Warning: HD-BET failed for {subject_id}: {result.stderr[:200]}")
                return img

            if not os.path.exists(out_path):
                print(f"  Warning: HD-BET output not found for {subject_id}")
                return img

            # Load data into memory before tmpdir is deleted (nibabel uses lazy loading)
            bet_img = nib.load(out_path)
            data = bet_img.get_fdata()
            return nib.Nifti1Image(data, bet_img.affine, bet_img.header)

    except subprocess.TimeoutExpired:
        print(f"  Warning: HD-BET timed out for {subject_id}")
        return img
    except Exception as e:
        print(f"  Warning: HD-BET error for {subject_id}: {e}")
        return img


def convert_and_copy_subjects(data, output_dir, split_name, skull_strip=False, bet_device='cpu'):
    """
    Convert Analyze format to NIfTI and copy files to output directory

    Args:
        data: List of subject dictionaries
        output_dir: Output directory
        split_name: 'train', 'val', or 'test'
        skull_strip: Apply HD-BET skull stripping
        bet_device: Device for HD-BET ('cpu' or 'cuda')
    """
    split_dir = Path(output_dir) / split_name

    # Create class directories
    normal_dir = split_dir / 'normal'
    alzheimer_dir = split_dir / 'alzheimer'
    normal_dir.mkdir(parents=True, exist_ok=True)
    alzheimer_dir.mkdir(parents=True, exist_ok=True)

    file_count = {'normal': 0, 'alzheimer': 0, 'errors': 0}

    for item in data:
        try:
            img_file = Path(item['img_file'])
            subject_id = item['subject_id']
            class_name = item['class_name']

            # Load Analyze format image
            img = nib.load(img_file)

            # Apply HD-BET skull stripping if requested
            # OASIS SUBJ_111 fill ratio ~39% confirms non-brain tissue is present
            if skull_strip:
                img = skull_strip_with_hdbet(img, subject_id, device=bet_device)

            # Determine output path
            out_filename = f"{subject_id}.nii.gz"
            if class_name == 'alzheimer':
                out_path = alzheimer_dir / out_filename
            else:
                out_path = normal_dir / out_filename

            # Save as compressed NIfTI
            nib.save(img, out_path)
            file_count[class_name] += 1

        except Exception as e:
            print(f"\nError processing {item['subject_id']}: {e}")
            file_count['errors'] += 1

    return file_count


def save_fold_info(cv_splits, output_dir):
    """
    Save fold information to JSON and CSV files
    
    Args:
        cv_splits: List of (train_data, val_data, test_data) tuples
        output_dir: Output directory
    """
    info_dir = Path(output_dir) / 'split_info'
    info_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'n_folds': len(cv_splits),
        'random_seed': 42,
        'stratified': True,
        'inner_validation': True,
        'folds': {}
    }
    
    cdr_stats = {}
    
    for fold_idx, (train_data, val_data, test_data) in enumerate(cv_splits):
        fold_key = f'fold_{fold_idx}'
        
        # Save train split
        train_json_path = info_dir / f'{fold_key}_train.json'
        with open(train_json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        train_csv_path = info_dir / f'{fold_key}_train.csv'
        pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
        
        # Save val split
        val_json_path = info_dir / f'{fold_key}_val.json'
        with open(val_json_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        val_csv_path = info_dir / f'{fold_key}_val.csv'
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        
        # Save test split
        test_json_path = info_dir / f'{fold_key}_test.json'
        with open(test_json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        test_csv_path = info_dir / f'{fold_key}_test.csv'
        pd.DataFrame(test_data).to_csv(test_csv_path, index=False)
        
        # Calculate statistics
        summary['folds'][fold_key] = {
            'train': {
                'total': len(train_data),
                'normal': sum(1 for x in train_data if x['label'] == 0),
                'alzheimer': sum(1 for x in train_data if x['label'] == 1),
            },
            'val': {
                'total': len(val_data),
                'normal': sum(1 for x in val_data if x['label'] == 0),
                'alzheimer': sum(1 for x in val_data if x['label'] == 1),
            },
            'test': {
                'total': len(test_data),
                'normal': sum(1 for x in test_data if x['label'] == 0),
                'alzheimer': sum(1 for x in test_data if x['label'] == 1),
            }
        }
        
        # CDR distribution
        cdr_stats[fold_key] = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        for item in train_data:
            cdr = item['cdr']
            cdr_stats[fold_key]['train'][cdr] = cdr_stats[fold_key]['train'].get(cdr, 0) + 1
        
        for item in val_data:
            cdr = item['cdr']
            cdr_stats[fold_key]['val'][cdr] = cdr_stats[fold_key]['val'].get(cdr, 0) + 1
        
        for item in test_data:
            cdr = item['cdr']
            cdr_stats[fold_key]['test'][cdr] = cdr_stats[fold_key]['test'].get(cdr, 0) + 1
    
    # Save summary
    summary_path = info_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save CDR distribution
    cdr_path = info_dir / 'cdr_distribution.json'
    with open(cdr_path, 'w') as f:
        json.dump(cdr_stats, f, indent=2)
    
    print(f"\nSplit information saved to {info_dir}")
    return summary


def main(args):
    print("=" * 70)
    print("OASIS 5-Fold Cross-Validation Dataset Preprocessing")
    print("=" * 70)
    
    # Paths
    oasis_dir = Path(args.oasis_dir)
    
    # Verify directory exists
    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")
    
    print(f"\nOASIS directory: {oasis_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Random seed: {args.seed}")
    print(f"Skull stripping (HD-BET): {'ENABLED (device=' + args.bet_device + ')' if args.skull_strip else 'DISABLED (non-brain tissue present!)'}")
    
    # Find all subjects
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
    
    # CDR distribution
    cdr_counts = {}
    for s in all_subjects:
        cdr = s['cdr']
        cdr_counts[cdr] = cdr_counts.get(cdr, 0) + 1
    
    print("\nCDR Distribution:")
    for cdr in sorted(cdr_counts.keys()):
        print(f"  CDR {cdr}: {cdr_counts[cdr]} subjects")
    
    # Create stratified CV splits
    print("\n" + "-" * 70)
    print("Creating stratified cross-validation splits...")
    print("-" * 70)
    
    cv_splits = create_stratified_cv_splits(
        all_subjects,
        n_folds=args.n_folds,
        random_seed=args.seed
    )
    
    # Convert and copy files if requested
    if not args.dry_run:
        print("\n" + "-" * 70)
        print("Converting and copying files to output directory...")
        print("-" * 70)
        
        output_dir = Path(args.output_dir)
        
        for fold_idx, (train_data, val_data, test_data) in enumerate(cv_splits):
            fold_dir = output_dir / f'fold_{fold_idx}'
            
            print(f"\n{'='*70}")
            print(f"Processing Fold {fold_idx}")
            print(f"{'='*70}")
            
            print(f"  Converting {len(train_data)} training subjects...")
            train_count = convert_and_copy_subjects(train_data, fold_dir, 'train',
                                                    skull_strip=args.skull_strip,
                                                    bet_device=args.bet_device)
            print(f"    Normal: {train_count['normal']}, Alzheimer: {train_count['alzheimer']}, Errors: {train_count['errors']}")
            
            print(f"  Converting {len(val_data)} validation subjects...")
            val_count = convert_and_copy_subjects(val_data, fold_dir, 'val',
                                                  skull_strip=args.skull_strip,
                                                  bet_device=args.bet_device)
            print(f"    Normal: {val_count['normal']}, Alzheimer: {val_count['alzheimer']}, Errors: {val_count['errors']}")
            
            print(f"  Converting {len(test_data)} test subjects...")
            test_count = convert_and_copy_subjects(test_data, fold_dir, 'test',
                                                   skull_strip=args.skull_strip,
                                                   bet_device=args.bet_device)
            print(f"    Normal: {test_count['normal']}, Alzheimer: {test_count['alzheimer']}, Errors: {test_count['errors']}")
        
        # Save split information
        summary = save_fold_info(cv_splits, output_dir)
        
        print("\n" + "=" * 70)
        print("Dataset preprocessing completed!")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print("\nDataset structure:")
        for fold_idx in range(args.n_folds):
            fold_stats = summary['folds'][f'fold_{fold_idx}']
            print(f"\nfold_{fold_idx}/")
            print(f"  train/")
            print(f"    normal/     ({fold_stats['train']['normal']} files)")
            print(f"    alzheimer/  ({fold_stats['train']['alzheimer']} files)")
            print(f"  val/")
            print(f"    normal/     ({fold_stats['val']['normal']} files)")
            print(f"    alzheimer/  ({fold_stats['val']['alzheimer']} files)")
            print(f"  test/")
            print(f"    normal/     ({fold_stats['test']['normal']} files)")
            print(f"    alzheimer/  ({fold_stats['test']['alzheimer']} files)")
        
        print(f"\nsplit_info/")
        print(f"  fold_0_train.json, fold_0_val.json, fold_0_test.json")
        print(f"  fold_1_train.json, fold_1_val.json, fold_1_test.json")
        print(f"  ...")
        print(f"  summary.json")
        print(f"  cdr_distribution.json")
        
        print("\n" + "=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print("\n1. Train each fold independently:")
        print("   bash script/train_cv5_fold.sh 0  # Train fold 0")
        print("   bash script/train_cv5_fold.sh 1  # Train fold 1")
        print("   ...")
        print("\n2. Evaluate all folds:")
        print("   bash script/evaluate_cv5_all.sh")
        print("\n3. Aggregate results:")
        print("   python aggregate_cv5_results.py")
        
    else:
        print("\n" + "=" * 70)
        print("Dry run completed - no files were converted/copied")
        print("Remove --dry_run flag to process files")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess OASIS dataset with 5-fold cross-validation for nested CV'
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
        default='./data/processed_oasis_cv5',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--skull_strip',
        action='store_true',
        default=False,
        help='Apply HD-BET skull stripping to remove skull/eyes/neck tissue.'
    )

    parser.add_argument(
        '--bet_device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for HD-BET skull stripping (default: cpu). Use cuda for GPU acceleration.'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run without converting/copying files (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about skipped subjects'
    )
    
    args = parser.parse_args()
    main(args)
