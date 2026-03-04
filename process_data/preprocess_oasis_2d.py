"""
2D Data preprocessing script for OASIS Alzheimer's Disease dataset
Extracts 120 center axial slices with smart cropping and padding

Input: Pre-processed skull-stripped data from processed_oasis_cv5_skullstrip
  Structure: fold_X/{train,val,test}/{normal,alzheimer}/*.nii.gz
  - Skull stripping already applied (HD-BET)
  - 1mm isotropic voxel spacing
  - NIfTI format (.nii.gz)

This script applies the 2D slice extraction pipeline:
- RAS orientation (standardized)
- Smart cropping: removes black regions (front of nose, back of head)
- Center padding: maintains brain at center with target shape
- 1mm voxel standardization via resizing (if needed)
- Intensity normalization (percentile-based)
- Preserves existing 5-fold cross-validation splits
"""
import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

try:
    from skimage.transform import resize
except ImportError:
    print("Warning: scikit-image not installed. Install with: pip install scikit-image")
    resize = None


def find_subjects_from_preprocessed_folds(input_dir, verbose=False):
    """
    Read subjects from a pre-processed fold directory structure.
    Skull stripping has already been applied to this data.

    Expected structure:
      input_dir/fold_X/{train,val,test}/{normal,alzheimer}/*.nii.gz

    Args:
        input_dir: Path to the pre-processed directory (e.g. processed_oasis_cv5_skullstrip)
        verbose: Print per-subject details

    Returns:
        (all_subjects, folds): all unique subjects and pre-defined fold splits
    """
    input_path = Path(input_dir)
    fold_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('fold_')],
        key=lambda d: int(d.name.split('_')[1])
    )

    if not fold_dirs:
        raise FileNotFoundError(f"No fold_X directories found in {input_dir}")

    print(f"\nFound {len(fold_dirs)} fold directories")

    all_subjects_map = {}  # subject_id -> subject dict
    folds = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split('_')[1])
        split_subjects = {}

        for split in ['train', 'val', 'test']:
            split_dir = fold_dir / split
            split_subjects[split] = []

            if not split_dir.exists():
                continue

            for class_name in ['normal', 'alzheimer']:
                label = 0 if class_name == 'normal' else 1
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue

                for img_file in sorted(class_dir.glob('*.nii.gz')):
                    subject_id = img_file.name.replace('.nii.gz', '')
                    subj = {
                        'subject_id': subject_id,
                        'img_file': str(img_file),
                        'label': label,
                        'class_name': class_name,
                        'cdr': 0.0 if label == 0 else 0.5,
                    }
                    all_subjects_map[subject_id] = subj
                    split_subjects[split].append(subj)

        # Print fold stats
        print(f"\nFold {fold_idx}:")
        for split in ['train', 'val', 'test']:
            subjs = split_subjects[split]
            if subjs:
                lbls = np.array([s['label'] for s in subjs])
                print(f"  {split:5s}: {len(subjs):3d} subjects "
                      f"(Normal: {np.sum(lbls == 0)}, Alzheimer: {np.sum(lbls == 1)})")

        folds.append({
            'fold': fold_idx,
            'train_subjects': split_subjects['train'],
            'val_subjects':   split_subjects['val'],
            'test_subjects':  split_subjects['test'],
            'train_indices':  list(range(len(split_subjects['train']))),
            'val_indices':    list(range(len(split_subjects['val']))),
        })

    all_subjects = list(all_subjects_map.values())
    print(f"\nTotal unique subjects: {len(all_subjects)}")
    print(f"  Normal:    {sum(1 for s in all_subjects if s['label'] == 0)}")
    print(f"  Alzheimer: {sum(1 for s in all_subjects if s['label'] == 1)}")

    return all_subjects, folds


def to_ras_orientation(img):
    """
    Convert image to RAS+ orientation (Right-Anterior-Superior)
    Standard neuroimaging orientation
    """
    # Get current orientation
    ornt = nib.orientations.io_orientation(img.affine)
    # Convert to RAS+
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    # Calculate transformation
    transform = nib.orientations.ornt_transform(ornt, ras_ornt)
    # Apply transformation - squeeze to remove any singleton dimensions
    data = nib.orientations.apply_orientation(img.get_fdata(), transform)
    data = np.squeeze(data)  # Remove singleton dimensions (e.g., shape (176,208,160,1) -> (176,208,160))
    
    # Update affine
    new_affine = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)
    
    return nib.Nifti1Image(data, new_affine)


def analyze_axial_content(data, num_slices=120):
    """
    Analyze axial slice content to find optimal bounding box
    
    Args:
        data: 3D array in RAS orientation (X, Y, Z) where Y is axial direction
        num_slices: Number of center slices to analyze
    
    Returns:
        Dictionary with bounding box stats across slices
    """
    y_dim = data.shape[1]
    start_idx = (y_dim - num_slices) // 2
    end_idx = start_idx + num_slices
    
    # Analyze each axial slice to find content boundaries
    x_mins, x_maxs = [], []
    z_mins, z_maxs = [], []
    
    for y_idx in range(start_idx, end_idx):
        slice_2d = data[:, y_idx, :]  # Axial slice (X, Z)
        
        # Find non-zero content
        mask = slice_2d > slice_2d.mean() * 0.1  # Threshold at 10% of mean
        if mask.sum() > 0:
            x_coords, z_coords = np.where(mask)
            if len(x_coords) > 0:
                x_mins.append(x_coords.min())
                x_maxs.append(x_coords.max())
                z_mins.append(z_coords.min())
                z_maxs.append(z_coords.max())
    
    # Use median boundaries to avoid outliers
    if len(x_mins) > 0:
        x_min = int(np.percentile(x_mins, 5))  # 5th percentile for robustness
        x_max = int(np.percentile(x_maxs, 95))  # 95th percentile
        z_min = int(np.percentile(z_mins, 5))
        z_max = int(np.percentile(z_maxs, 95))
        
        width = x_max - x_min + 1
        height = z_max - z_min + 1
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'z_min': z_min, 'z_max': z_max,
            'width': width, 'height': height,
            'max_dim': max(width, height)
        }
    
    return None


def crop_and_pad_axial_slices(data, num_slices=120, target_size=224):
    """
    Crop axial slices to remove black regions, then pad to square target size
    while keeping brain centered
    
    Args:
        data: 3D array in RAS orientation (X, Y, Z)
        num_slices: Number of center axial slices
        target_size: Target square size (e.g., 224, 240, 256)
    
    Returns:
        4D array (num_slices, target_size, target_size) with cropped and padded axial slices
    """
    # First, extract center axial slices
    y_dim = data.shape[1]
    if y_dim < num_slices:
        # Pad if not enough slices
        pad_before = (num_slices - y_dim) // 2
        pad_after = num_slices - y_dim - pad_before
        data = np.pad(data, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
    else:
        # Extract center slices
        start_idx = (y_dim - num_slices) // 2
        data = data[:, start_idx:start_idx + num_slices, :]
    
    # Analyze content to find optimal bounding box
    bbox = analyze_axial_content(data, num_slices)
    
    if bbox is None:
        # Fallback: just resize without cropping
        print("  Warning: Could not detect content, using full image")
        x_min, x_max = 0, data.shape[0] - 1
        z_min, z_max = 0, data.shape[2] - 1
    else:
        x_min, x_max = bbox['x_min'], bbox['x_max']
        z_min, z_max = bbox['z_min'], bbox['z_max']
        print(f"  Content bounding box: X[{x_min}:{x_max}], Z[{z_min}:{z_max}], Size: {bbox['width']}x{bbox['height']}")
    
    # Crop all slices using the bounding box
    cropped_slices = []
    for y_idx in range(num_slices):
        slice_2d = data[:, y_idx, :]  # (X, Z)
        cropped = slice_2d[x_min:x_max+1, z_min:z_max+1]
        cropped_slices.append(cropped)
    
    # Now resize and pad each slice to target_size x target_size
    processed_slices = []
    for cropped in cropped_slices:
        h, w = cropped.shape
        
        # Resize to fit within target_size while preserving aspect ratio
        # This ensures 1mm voxel spacing is approximately maintained
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        if new_h > 0 and new_w > 0:
            resized = resize(cropped, (new_h, new_w), anti_aliasing=True, preserve_range=True)
        else:
            resized = cropped
        
        # Pad to target_size x target_size with brain centered
        padded = np.zeros((target_size, target_size), dtype=np.float32)
        start_h = (target_size - resized.shape[0]) // 2
        start_w = (target_size - resized.shape[1]) // 2
        padded[start_h:start_h+resized.shape[0], start_w:start_w+resized.shape[1]] = resized
        
        processed_slices.append(padded)
    
    # Stack into (num_slices, target_size, target_size)
    result = np.stack(processed_slices, axis=0)
    return result


def center_crop_or_pad(data, target_shape):
    """
    Center crop or pad data to target shape
    """
    current_shape = data.shape
    result = np.zeros(target_shape, dtype=data.dtype)
    
    # Calculate start indices for each dimension
    starts_src = []
    starts_tgt = []
    sizes = []
    
    for i in range(len(target_shape)):
        if current_shape[i] >= target_shape[i]:
            # Crop
            start = (current_shape[i] - target_shape[i]) // 2
            starts_src.append(start)
            starts_tgt.append(0)
            sizes.append(target_shape[i])
        else:
            # Pad
            start = (target_shape[i] - current_shape[i]) // 2
            starts_src.append(0)
            starts_tgt.append(start)
            sizes.append(current_shape[i])
    
    # Copy data
    result[
        starts_tgt[0]:starts_tgt[0]+sizes[0],
        starts_tgt[1]:starts_tgt[1]+sizes[1],
        starts_tgt[2]:starts_tgt[2]+sizes[2]
    ] = data[
        starts_src[0]:starts_src[0]+sizes[0],
        starts_src[1]:starts_src[1]+sizes[1],
        starts_src[2]:starts_src[2]+sizes[2]
    ]
    
    return result


def normalize_intensity(data, method='zscore'):
    """
    Normalize intensity values
    
    Args:
        data: 3D numpy array
        method: 'zscore', 'minmax', or 'percentile'
    
    Returns:
        Normalized data
    """
    # Mask for non-zero values
    mask = data > 0
    
    if method == 'zscore':
        mean = data[mask].mean()
        std = data[mask].std()
        data[mask] = (data[mask] - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = data[mask].min()
        max_val = data[mask].max()
        data[mask] = (data[mask] - min_val) / (max_val - min_val + 1e-8)
    elif method == 'percentile':
        p1, p99 = np.percentile(data[mask], [1, 99])
        data[mask] = np.clip(data[mask], p1, p99)
        data[mask] = (data[mask] - p1) / (p99 - p1 + 1e-8)
    
    return data


def investigate_optimal_size(data, num_slices=120):
    """
    Investigate optimal target size by analyzing actual brain content dimensions
    
    Args:
        data: 3D array in RAS orientation
        num_slices: Number of slices to analyze
    
    Returns:
        Recommended target size
    """
    bbox = analyze_axial_content(data, num_slices)
    
    if bbox is None:
        return 224  # Default fallback
    
    max_dim = bbox['max_dim']
    
    # Round up to nearest common size
    if max_dim <= 200:
        return 224
    elif max_dim <= 220:
        return 240
    elif max_dim <= 240:
        return 256
    else:
        return 256  # Cap at 256


def standardize_to_1mm(data, original_voxel_size=(1.0, 1.0, 1.0)):
    """
    Standardize voxel spacing to 1mm isotropic
    
    Args:
        data: 3D array
        original_voxel_size: Original voxel dimensions (x, y, z) in mm
    
    Returns:
        Resampled data with 1mm voxel spacing
    """
    # Calculate zoom factors to achieve 1mm spacing
    zoom_factors = original_voxel_size
    
    # Only resample if not already 1mm
    if not np.allclose(zoom_factors, [1.0, 1.0, 1.0]):
        print(f"  Resampling from {original_voxel_size} to 1mm isotropic")
        data = ndimage.zoom(data, zoom_factors, order=1, mode='constant', cval=0.0)
    
    return data


def process_single_subject(img_file, subject_id, args):
    """
    Process a single subject: load, preprocess, extract 2D slices
    
    Returns:
        Processed 4D array (num_slices, img_size, img_size) or None if failed
    """
    try:
        # Load image (Analyze format)
        img = nib.load(img_file)
        
        # Convert to RAS orientation (REQUIRED for consistent anatomical orientation)
        if args.ras_orient:
            img = to_ras_orientation(img)

        # Get data and squeeze to remove singleton dimensions
        # NIfTI format may have singleton dimensions
        data = np.squeeze(img.get_fdata()).astype(np.float32)
        
        # Get voxel size for 1mm standardization
        voxel_size = img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else (1.0, 1.0, 1.0)
        
        # Standardize to 1mm isotropic voxels
        data = standardize_to_1mm(data, voxel_size)
        
        # Investigate optimal size (only print for first subject if verbose)
        if args.verbose:
            recommended_size = investigate_optimal_size(data, args.num_slices)
            print(f"  Recommended target size for {subject_id}: {recommended_size}x{recommended_size}")
        
        # Crop and pad axial slices intelligently
        # This removes black regions (front of nose, back of head) and pads to target size
        data = crop_and_pad_axial_slices(data, num_slices=args.num_slices, target_size=args.img_size)
        
        # Normalize intensity
        data = normalize_intensity(data, method=args.norm_method)
        
        # Transpose to (img_size, img_size, num_slices) for consistency with old format
        # This makes it compatible with existing dataset loading code
        data = np.transpose(data, (1, 2, 0))

        # Final crop/pad to guarantee exact target shape (img_size, img_size, num_slices)
        data = center_crop_or_pad(data, (args.img_size, args.img_size, args.num_slices))
        
        return data
        
    except Exception as e:
        print(f"\nError processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_processed_data(subjects, folds, output_dir, args):
    """
    Process and save all subjects in numpy format for efficient loading
    
    Saves:
    - fold_0/ to fold_4/ directories with train.npz and val.npz files
    - Each .npz contains: images (N, img_size, img_size, num_slices), labels (N,), subject_ids (N,)
    - Shape: (N, H, W, num_slices) where H=W=img_size, and last dimension is number of axial slices
    - Axial slices are intelligently cropped and padded with brain centered
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Processing and saving subjects...")
    print("="*70)
    
    # Process all subjects first
    processed_data = {}
    failed_subjects = []
    
    print("\nProcessing all subjects...")
    for subject in tqdm(subjects, desc="Processing"):
        img_file = subject['img_file']
        subject_id = subject['subject_id']
        
        data = process_single_subject(img_file, subject_id, args)
        
        if data is not None:
            processed_data[subject_id] = data
        else:
            failed_subjects.append(subject_id)
    
    print(f"\nSuccessfully processed: {len(processed_data)}/{len(subjects)} subjects")
    if failed_subjects:
        print(f"Failed subjects: {len(failed_subjects)}")
        if args.verbose:
            for sid in failed_subjects:
                print(f"  - {sid}")
    
    # Save each fold
    print("\nSaving folds...")
    for fold_data in tqdm(folds, total=len(folds)):
        fold_idx = fold_data['fold']
        fold_dir = output_path / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare train data
        train_subjects = fold_data['train_subjects']
        train_images = []
        train_labels = []
        train_ids = []
        
        for subj in train_subjects:
            sid = subj['subject_id']
            if sid in processed_data:
                train_images.append(processed_data[sid])
                train_labels.append(subj['label'])
                train_ids.append(sid)
        
        # Prepare val data
        val_subjects = fold_data['val_subjects']
        val_images = []
        val_labels = []
        val_ids = []
        
        for subj in val_subjects:
            sid = subj['subject_id']
            if sid in processed_data:
                val_images.append(processed_data[sid])
                val_labels.append(subj['label'])
                val_ids.append(sid)
        
        # Convert to numpy arrays
        train_images = np.array(train_images, dtype=np.float32)  # Shape: (N, 224, 224, 120)
        train_labels = np.array(train_labels, dtype=np.int64)
        val_images = np.array(val_images, dtype=np.float32)
        val_labels = np.array(val_labels, dtype=np.int64)
        
        # Prepare test data (if present in pre-processed folds)
        test_subjects = fold_data.get('test_subjects', [])
        test_images = []
        test_labels = []
        test_ids = []
        for subj in test_subjects:
            sid = subj['subject_id']
            if sid in processed_data:
                test_images.append(processed_data[sid])
                test_labels.append(subj['label'])
                test_ids.append(sid)

        # Convert to numpy arrays
        test_images = np.array(test_images, dtype=np.float32) if test_images else np.array([], dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.int64)

        # Save as compressed numpy files
        np.savez_compressed(
            fold_dir / 'train.npz',
            images=train_images,
            labels=train_labels,
            subject_ids=train_ids
        )
        
        np.savez_compressed(
            fold_dir / 'val.npz',
            images=val_images,
            labels=val_labels,
            subject_ids=val_ids
        )

        if len(test_ids) > 0:
            np.savez_compressed(
                fold_dir / 'test.npz',
                images=test_images,
                labels=test_labels,
                subject_ids=test_ids
            )
        
        # Save metadata as JSON
        metadata = {
            'fold': fold_idx,
            'train': {
                'num_subjects': len(train_ids),
                'num_normal': int(np.sum(train_labels == 0)),
                'num_alzheimer': int(np.sum(train_labels == 1)),
                'subject_ids': train_ids
            },
            'val': {
                'num_subjects': len(val_ids),
                'num_normal': int(np.sum(val_labels == 0)),
                'num_alzheimer': int(np.sum(val_labels == 1)),
                'subject_ids': val_ids
            },
            'test': {
                'num_subjects': len(test_ids),
                'num_normal': int(np.sum(test_labels == 0)) if len(test_ids) > 0 else 0,
                'num_alzheimer': int(np.sum(test_labels == 1)) if len(test_ids) > 0 else 0,
                'subject_ids': test_ids
            },
            'preprocessing': {
                'img_size': args.img_size,
                'num_slices': args.num_slices,
                'ras_orient': args.ras_orient,
                'skull_strip': 'pre-applied',
                'norm_method': args.norm_method
            }
        }
        
        with open(fold_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Fold {fold_idx}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Save overall summary
    summary = {
        'total_subjects': len(subjects),
        'processed_subjects': len(processed_data),
        'failed_subjects': len(failed_subjects),
        'num_folds': len(folds),
        'image_shape': [args.img_size, args.img_size, args.num_slices],
        'preprocessing_config': {
            'img_size': args.img_size,
            'num_slices': args.num_slices,
            'ras_orient': args.ras_orient,
            'skull_strip': 'pre-applied (input data already skull-stripped)',
            'smart_cropping': True,
            'center_padding': True,
            'voxel_standardization_1mm': True,
            'norm_method': args.norm_method,
            'original_axial_shape': '(256, 256, 160)',
            'notes': 'Smart crop/pad with brain centered; skull stripping pre-applied via HD-BET'
        }
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Data preprocessing completed!")
    print(f"{'='*70}")
    print(f"Output directory: {output_path}")
    print(f"Data shape: ({args.img_size}, {args.img_size}, {args.num_slices})")
    print(f"Format: .npz (compressed numpy)")


def main(args):
    print("=" * 70)
    print("OASIS 2D Alzheimer's Disease Dataset Preprocessing")
    print("=" * 70)
    
    # Verify dependencies
    if resize is None:
        print("\nError: scikit-image is required but not installed.")
        print("Install with: pip install scikit-image")
        return
    
    oasis_dir = Path(args.oasis_dir)
    
    if not oasis_dir.exists():
        raise FileNotFoundError(f"OASIS directory not found: {oasis_dir}")
    
    print(f"\nInput directory: {oasis_dir}")
    print(f"Preprocessing options:")
    print(f"  - Target image size: {args.img_size}x{args.img_size}")
    print(f"  - Number of center axial slices: {args.num_slices}")
    print(f"  - RAS orientation: {args.ras_orient}")
    print(f"  - Skull stripping: PRE-APPLIED (data already skull-stripped)")
    print(f"  - Smart cropping: Enabled (removes black regions)")
    print(f"  - Center padding: Enabled (brain centered in frame)")
    print(f"  - 1mm voxel standardization: Enabled")
    print(f"  - Intensity normalization: {args.norm_method}")

    # Load subjects and pre-defined folds from the preprocessed directory
    print("\n" + "-" * 70)
    print("Loading subjects from pre-processed fold structure...")
    print("-" * 70)

    all_subjects, folds = find_subjects_from_preprocessed_folds(oasis_dir, verbose=args.verbose)

    if len(all_subjects) == 0:
        print("\nNo subjects found! Check the pre-processed directory structure.")
        return
    
    # Process and save data
    if not args.dry_run:
        save_processed_data(all_subjects, folds, args.output_dir, args)
    else:
        print("\n" + "=" * 70)
        print("Dry run completed - no files were processed/saved")
        print("Remove --dry_run flag to process files")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess OASIS dataset for 2D Alzheimer classification'
    )
    
    parser.add_argument(
        '--oasis_dir',
        type=str,
        default='./data/processed_oasis_cv5_skullstrip',
        help='Path to pre-processed skull-stripped directory with fold_X/{train,val,test}/{normal,alzheimer}/*.nii.gz structure'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed_oasis_2d_cv5',
        help='Output directory for processed 2D data'
    )
    
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Target image size (default: 224x224)'
    )
    
    parser.add_argument(
        '--num_slices',
        type=int,
        default=120,
        help='Number of center axial slices to extract (default: 120)'
    )
    
    parser.add_argument(
        '--ras_orient',
        action='store_true',
        default=True,
        help='Convert to RAS+ orientation (STRONGLY RECOMMENDED for consistent anatomy)'
    )
    
    parser.add_argument(
        '--norm_method',
        type=str,
        default='percentile',
        choices=['zscore', 'minmax', 'percentile'],
        help='Intensity normalization method (default: percentile)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run without processing files (for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    main(args)
