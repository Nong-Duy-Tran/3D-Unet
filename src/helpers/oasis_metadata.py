from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_txt_metadata(txt_path: Path):
    """
    Extract CDR (Clinical Dementia Rating) from the OASIS text file

    Returns:
        float CDR or None
        CDR = 0: Normal/Healthy
        CDR > 0: Various stages of dementia/Alzheimer's
        None (empty CDR): Treated as 0 (young healthy controls)
    """
    cdr = None
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if line.startswith('CDR:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            cdr = float(parts[1])
                        except ValueError:
                            # CDR field exists but empty -> treat as 0
                            cdr = 0.0
                    else:
                        # CDR field exists but empty -> treat as 0
                        cdr = 0.0
                    break
    except Exception as exc:
        print(f"Error reading {txt_path}: {exc}")
    return cdr


def find_all_oasis_subjects(oasis_root, verbose: bool = False):
    """
    Recursively find all OASIS subjects across all discs.

    Returns:
        List of subject dicts with keys:
        subject_id, subject_path, img_file, cdr, label, class_name, disc
    """
    subjects = []
    oasis_path = Path(oasis_root)

    disc_folders = sorted([d for d in oasis_path.iterdir() if d.is_dir() and d.name.startswith('disc')])

    print(f"\nFound {len(disc_folders)} disc folders")

    stats = {
        'total_dirs': 0,
        'no_txt': 0,
        'no_cdr': 0,
        'no_subj_path': 0,
        'no_img_file': 0,
        'success': 0,
    }

    for disc_folder in disc_folders:
        print(f"\nProcessing {disc_folder.name}...")

        subject_dirs = [d for d in disc_folder.iterdir() if d.is_dir() and d.name.startswith('OAS1_')]
        stats['total_dirs'] += len(subject_dirs)

        for subject_dir in tqdm(subject_dirs, desc=f"  {disc_folder.name}"):
            subject_id = subject_dir.name

            txt_file = subject_dir / f"{subject_id}.txt"
            if not txt_file.exists():
                stats['no_txt'] += 1
                if verbose:
                    print(f"  No .txt file: {subject_id}")
                continue

            cdr = parse_txt_metadata(txt_file)

            if cdr is None:
                stats['no_cdr'] += 1
                if verbose:
                    print(f"  No CDR found: {subject_id}")
                continue

            label = 0 if cdr == 0.0 else 1
            class_name = 'normal' if label == 0 else 'alzheimer'

            subj_path = subject_dir / 'PROCESSED' / 'MPRAGE' / 'SUBJ_111'
            if not subj_path.exists():
                stats['no_subj_path'] += 1
                if verbose:
                    print(f"  No SUBJ_111 path: {subject_id}")
                continue

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
            subjects.append(
                {
                    'subject_id': subject_id,
                    'subject_path': str(subject_dir),
                    'img_file': str(img_file),
                    'cdr': cdr,
                    'label': label,
                    'class_name': class_name,
                    'disc': disc_folder.name,
                }
            )

    print(f"\n{'=' * 70}")
    print("Scan Summary:")
    print(f"  Total subject directories: {stats['total_dirs']}")
    print(f"  Successfully processed: {stats['success']}")
    print(f"\nSkipped:")
    print(f"  Missing .txt file: {stats['no_txt']}")
    print(f"  Missing CDR in .txt: {stats['no_cdr']}")
    print(f"  Missing SUBJ_111 folder: {stats['no_subj_path']}")
    print(f"  Missing subject image: {stats['no_img_file']}")
    print(f"{'=' * 70}")

    unique_subjects = {}
    for subj in subjects:
        base_id = subj['subject_id'].rsplit('_', 1)[0]
        if base_id not in unique_subjects:
            unique_subjects[base_id] = subj
        else:
            if '_MR1' in subj['subject_id']:
                unique_subjects[base_id] = subj

    subjects = list(unique_subjects.values())

    if len(unique_subjects) < stats['success']:
        print(f"\nNote: Found {stats['success'] - len(unique_subjects)} duplicate scans.")
        print(f"      Keeping one scan per subject: {len(subjects)} unique subjects")

    return subjects


def split_by_subject(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by subject to avoid data leakage.
    Ensures balanced class distribution across splits.
    """
    random.seed(random_seed)

    normal_subjects = [s for s in data if s['label'] == 0]
    alzheimer_subjects = [s for s in data if s['label'] == 1]

    print(f"\nTotal subjects: {len(data)}")
    print(f"  Normal (CDR=0): {len(normal_subjects)}")
    print(f"  Alzheimer (CDR>0): {len(alzheimer_subjects)}")

    random.shuffle(normal_subjects)
    random.shuffle(alzheimer_subjects)

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

    train_data = normal_train + alzheimer_train
    val_data = normal_val + alzheimer_val
    test_data = normal_test + alzheimer_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def save_split_info(train_data, val_data, test_data, output_dir):
    """
    Save split information to JSON and CSV files.
    """
    info_dir = Path(output_dir) / 'split_info'
    info_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }

    for split_name, split_data in splits.items():
        json_path = info_dir / f'{split_name}_split.json'
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        csv_path = info_dir / f'{split_name}_split.csv'
        df = pd.DataFrame(split_data)
        df.to_csv(csv_path, index=False)

    summary = {
        'train': {
            'total': len(train_data),
            'normal': sum(1 for x in train_data if x['label'] == 0),
            'alzheimer': sum(1 for x in train_data if x['label'] == 1),
            'subjects': len(train_data),
        },
        'val': {
            'total': len(val_data),
            'normal': sum(1 for x in val_data if x['label'] == 0),
            'alzheimer': sum(1 for x in val_data if x['label'] == 1),
            'subjects': len(val_data),
        },
        'test': {
            'total': len(test_data),
            'normal': sum(1 for x in test_data if x['label'] == 0),
            'alzheimer': sum(1 for x in test_data if x['label'] == 1),
            'subjects': len(test_data),
        },
    }

    summary_path = info_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    cdr_stats = {
        'train': {},
        'val': {},
        'test': {},
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


def _split_summary(items):
    return {
        'total': len(items),
        'normal': sum(1 for x in items if x['label'] == 0),
        'alzheimer': sum(1 for x in items if x['label'] == 1),
        'subjects': len(items),
    }


def make_kfold_splits(data, k_folds=5, test_ratio=0.2, val_ratio=0.1, random_seed=42):
    """
    Split data into:
      - fixed held-out test set (stratified) with size test_ratio
      - k repeated stratified train/val splits on the remaining data

    For each fold, val set is val_ratio of the remaining data (stratified).
    Returns: folds, test_data
    folds is a list of dicts: {fold, train, val}
    """
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    from sklearn.model_selection import StratifiedShuffleSplit

    random.seed(random_seed)
    labels = [s['label'] for s in data]

    if test_ratio and test_ratio > 0:
        test_split = StratifiedShuffleSplit(
            n_splits=1, test_size=float(test_ratio), random_state=random_seed
        )
        trainval_idx, test_idx = next(test_split.split(list(range(len(data))), labels))
        trainval_data = [data[i] for i in trainval_idx]
        test_data = [data[i] for i in test_idx]
    else:
        trainval_data = list(data)
        test_data = []

    trainval_labels = [s['label'] for s in trainval_data]
    val_split = StratifiedShuffleSplit(
        n_splits=k_folds, test_size=float(val_ratio), random_state=random_seed
    )

    fold_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(val_split.split(list(range(len(trainval_data))), trainval_labels)):
        train_data = [trainval_data[i] for i in train_idx]
        val_data = [trainval_data[i] for i in val_idx]
        fold_splits.append({'fold': fold_idx, 'train': train_data, 'val': val_data})

    return fold_splits, test_data


def save_kfold_info(folds, test_data, output_dir, random_seed=42):
    """
    Save k-fold split information to JSON and CSV files.
    """
    info_dir = Path(output_dir) / 'split_info'
    info_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'seed': random_seed,
        'k_folds': len(folds),
        'test': test_data,
        'folds': [],
    }
    summary = {
        'seed': random_seed,
        'k_folds': len(folds),
        'test': _split_summary(test_data),
        'folds': [],
    }

    for fold in folds:
        fold_idx = int(fold['fold'])
        train_data = fold['train']
        val_data = fold['val']
        payload['folds'].append(
            {
                'fold': fold_idx,
                'train': train_data,
                'val': val_data,
            }
        )
        summary['folds'].append(
            {
                'fold': fold_idx,
                'train': _split_summary(train_data),
                'val': _split_summary(val_data),
            }
        )

    folds_path = info_dir / 'folds.json'
    with open(folds_path, 'w') as f:
        json.dump(payload, f, indent=2)

    summary_path = info_dir / 'folds_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if test_data:
        test_json_path = info_dir / 'test_split.json'
        with open(test_json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        test_csv_path = info_dir / 'test_split.csv'
        pd.DataFrame(test_data).to_csv(test_csv_path, index=False)

    print(f"\nK-fold split information saved to {info_dir}")
    return summary
