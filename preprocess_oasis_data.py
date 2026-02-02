"""
Data preprocessing script for OASIS Alzheimer's Disease dataset
Processes all OASIS discs and creates train/validation/test splits
"""
import os
import json
import shutil
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import nibabel as nib


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
        'no_t88_path': 0,
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
            
            # 2. Find the T88 registered, masked brain image
            t88_path = subject_dir / 'PROCESSED' / 'MPRAGE' / 'T88_111'
            
            if not t88_path.exists():
                stats['no_t88_path'] += 1
                if verbose:
                    print(f"  No T88_111 path: {subject_id}")
                continue
            
            # Find the masked image file
            img_file = None
            for f in t88_path.iterdir():
                if f.name.endswith('masked_gfc.img'):
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
    print(f"  Missing T88_111 folder: {stats['no_t88_path']}")
    print(f"  Missing masked image: {stats['no_img_file']}")
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


def split_by_subject(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by subject to avoid data leakage
    Ensures all scans from same subject are in same split
    Also ensures balanced class distribution across splits
    
    Args:
        data: List of subject dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data
    """
    random.seed(random_seed)
    
    # Separate by class
    normal_subjects = [s for s in data if s['label'] == 0]
    alzheimer_subjects = [s for s in data if s['label'] == 1]
    
    print(f"\nTotal subjects: {len(data)}")
    print(f"  Normal (CDR=0): {len(normal_subjects)}")
    print(f"  Alzheimer (CDR>0): {len(alzheimer_subjects)}")
    
    # Shuffle
    random.shuffle(normal_subjects)
    random.shuffle(alzheimer_subjects)
    
    # Split each class independently to maintain balance
    def split_class(subjects, train_r, val_r, test_r):
        n = len(subjects)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        train = subjects[:n_train]
        val = subjects[n_train:n_train + n_val]
        test = subjects[n_train + n_val:]
        
        return train, val, test
    
    normal_train, normal_val, normal_test = split_class(normal_subjects, train_ratio, val_ratio, test_ratio)
    alzheimer_train, alzheimer_val, alzheimer_test = split_class(alzheimer_subjects, train_ratio, val_ratio, test_ratio)
    
    # Combine
    train_data = normal_train + alzheimer_train
    val_data = normal_val + alzheimer_val
    test_data = normal_test + alzheimer_test
    
    # Shuffle the combined splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def convert_and_copy_subjects(data, output_dir, split_name):
    """
    Convert Analyze format to NIfTI and copy files to output directory
    
    Args:
        data: List of subject dictionaries
        output_dir: Output directory
        split_name: 'train', 'val', or 'test'
    """
    split_dir = Path(output_dir) / split_name
    
    # Create class directories
    normal_dir = split_dir / 'normal'
    alzheimer_dir = split_dir / 'alzheimer'
    normal_dir.mkdir(parents=True, exist_ok=True)
    alzheimer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(data)} subjects for {split_name} split...")
    
    file_count = {'normal': 0, 'alzheimer': 0, 'errors': 0}
    
    for item in tqdm(data):
        try:
            img_file = Path(item['img_file'])
            subject_id = item['subject_id']
            class_name = item['class_name']
            
            # Load Analyze format image
            img = nib.load(img_file)
            
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
    
    print(f"  Normal: {file_count['normal']}, Alzheimer: {file_count['alzheimer']}, Errors: {file_count['errors']}")
    
    return file_count


def save_split_info(train_data, val_data, test_data, output_dir):
    """
    Save split information to JSON and CSV files
    
    Args:
        train_data, val_data, test_data: Split data lists
        output_dir: Output directory
    """
    info_dir = Path(output_dir) / 'split_info'
    info_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        # JSON format
        json_path = info_dir / f'{split_name}_split.json'
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # CSV format
        csv_path = info_dir / f'{split_name}_split.csv'
        df = pd.DataFrame(split_data)
        df.to_csv(csv_path, index=False)
    
    # Calculate statistics
    summary = {
        'train': {
            'total': len(train_data),
            'normal': sum(1 for x in train_data if x['label'] == 0),
            'alzheimer': sum(1 for x in train_data if x['label'] == 1),
            'subjects': len(train_data)
        },
        'val': {
            'total': len(val_data),
            'normal': sum(1 for x in val_data if x['label'] == 0),
            'alzheimer': sum(1 for x in val_data if x['label'] == 1),
            'subjects': len(val_data)
        },
        'test': {
            'total': len(test_data),
            'normal': sum(1 for x in test_data if x['label'] == 0),
            'alzheimer': sum(1 for x in test_data if x['label'] == 1),
            'subjects': len(test_data)
        }
    }
    
    summary_path = info_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save CDR distribution
    cdr_stats = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for split_name, split_data in splits.items():
        cdr_counts = {}
        for item in split_data:
            cdr = item['cdr']
            cdr_counts[cdr] = cdr_counts.get(cdr, 0) + 1
        cdr_stats[split_name] = cdr_counts
    
    cdr_path = info_dir / 'cdr_distribution.json'
    with open(cdr_path, 'w') as f:
        json.dump(cdr_stats, f, indent=2)
    
    print(f"\nSplit information saved to {info_dir}")
    return summary


def main(args):
    print("=" * 70)
    print("OASIS Alzheimer's Disease Dataset Preprocessing")
    print("=" * 70)
    
    # Paths
    oasis_dir = Path(args.oasis_dir)
    
    # Verify directory exists
    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")
    
    print(f"\nOASIS directory: {oasis_dir}")
    
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
    
    # Split data
    print("\n" + "-" * 70)
    print("Splitting data by subject...")
    print("-" * 70)
    
    train_data, val_data, test_data = split_by_subject(
        all_subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
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
    
    # Convert and copy files if requested
    if not args.dry_run:
        print("\n" + "-" * 70)
        print("Converting and copying files to output directory...")
        print("-" * 70)
        
        output_dir = Path(args.output_dir)
        
        # Convert and copy files
        train_count = convert_and_copy_subjects(train_data, output_dir, 'train')
        val_count = convert_and_copy_subjects(val_data, output_dir, 'val')
        test_count = convert_and_copy_subjects(test_data, output_dir, 'test')
        
        # Save split information
        summary = save_split_info(train_data, val_data, test_data, output_dir)
        
        print("\n" + "=" * 70)
        print("Dataset preprocessing completed!")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print("\nDataset structure:")
        print(f"  train/")
        print(f"    normal/     ({train_count['normal']} files)")
        print(f"    alzheimer/  ({train_count['alzheimer']} files)")
        print(f"  val/")
        print(f"    normal/     ({val_count['normal']} files)")
        print(f"    alzheimer/  ({val_count['alzheimer']} files)")
        print(f"  test/")
        print(f"    normal/     ({test_count['normal']} files)")
        print(f"    alzheimer/  ({test_count['alzheimer']} files)")
        print(f"  split_info/")
        print(f"    train_split.json")
        print(f"    val_split.json")
        print(f"    test_split.json")
        print(f"    summary.json")
        print(f"    cdr_distribution.json")
        
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
        '--verbose',
        action='store_true',
        help='Print detailed information about skipped subjects'
    )
    
    args = parser.parse_args()
    main(args)
