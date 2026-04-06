"""Analyze processed OASIS NIfTI volumes and recommend 2D target size.

This script:
1. Loads processed NIfTI files from cache folder
2. Visualizes shape distributions for sagittal/coronal/axial axes
3. Analyzes axial content dimensions and recommends target size
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Add process_data to path
sys.path.insert(0, str(Path(__file__).parent / 'process_data'))

from process_data.preprocess_oasis_2d import (
    analyze_axial_content, 
    investigate_optimal_size,
)


def collect_cached_nifti_files(cache_root, max_subjects=400):
    """Collect NIfTI files from processed cache folders."""
    if not cache_root.exists():
        return []

    nifti_files = sorted(cache_root.glob('*.nii.gz'))
    if max_subjects is not None:
        nifti_files = nifti_files[:max_subjects]

    return nifti_files


def visualize_nifti_shape_distribution(nifti_paths, output_path):
    """Create and save distribution plots of NIfTI volume shapes.

    Axis order is assumed to already be RAS: [sagittal, coronal, axial].
    """
    if not nifti_paths:
        print("No NIfTI files provided for shape visualization.")
        return

    shapes = []
    for nifti_path in nifti_paths:
        try:
            img = nib.load(str(nifti_path))
            if len(img.shape) < 3:
                continue
            shapes.append(img.shape[:3])
        except Exception as e:
            print(f"Skipping {nifti_path.name} due to error: {e}")

    if not shapes:
        print("No valid 3D NIfTI shapes found for visualization.")
        return

    shapes_arr = np.array(shapes)
    print(f"abc: {shapes_arr}")
    sagittal_sizes = shapes_arr[:, 0]
    coronal_sizes = shapes_arr[:, 1]
    axial_sizes = shapes_arr[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axis_data = [
        (sagittal_sizes, 'Sagittal Size (dim 0)'),
        (coronal_sizes, 'Coronal Size (dim 1)'),
        (axial_sizes, 'Axial Size (dim 2)'),
    ]

    for idx, (values, title) in enumerate(axis_data):
        axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.75)
        axes[idx].set_xlabel('Pixels')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(title)
        axes[idx].axvline(
            np.median(values),
            color='red',
            linestyle='--',
            label=f"Median: {np.median(values):.1f}",
        )
        axes[idx].legend()

    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved axis-size distribution to: {output_path}")


def analyze_subject(img_file, subject_id, num_slices=120):
    """Analyze a single subject's axial content"""
    try:
        print(f"\n{'='*70}")
        print(f"Analyzing: {subject_id}")
        print(f"{'='*70}")
        
        # Load image
        img = nib.load(img_file)
        print(f"Canonical shape: {img.shape}")
        print(f"Voxel size: {img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else 'Unknown'}")

        # The processed cache volumes are already in RAS [sagittal, coronal, axial].
        data = np.squeeze(img.get_fdata()).astype(np.float32)
        if data.ndim != 3:
            print(f"Skipping {subject_id}: expected 3D volume, got shape {data.shape}")
            return None
        
        # Analyze axial content
        valid_num_slices = min(num_slices, data.shape[2])
        bbox = analyze_axial_content(data, valid_num_slices)
        
        if bbox:
            print(f"\nAxial slice content analysis (across {valid_num_slices} center slices):")
            print(f"  X range: [{bbox['x_min']}, {bbox['x_max']}]  Width: {bbox['width']} pixels")
            y_min = bbox.get('y_min', bbox.get('z_min'))
            y_max = bbox.get('y_max', bbox.get('z_max'))
            print(f"  Y range: [{y_min}, {y_max}]  Height: {bbox['height']} pixels")
            print(f"  Max dimension: {bbox['max_dim']} pixels")
            
            # Recommend size
            recommended = investigate_optimal_size(data, valid_num_slices)
            print(f"\nRecommended target size: {recommended}x{recommended}")
            
            # Show what percentage of target will be content vs padding
            for target in [224, 240, 256]:
                content_ratio = (bbox['max_dim'] / target) * 100
                padding_ratio = 100 - content_ratio
                print(f"  {target}x{target}: Brain={content_ratio:.1f}%, Padding={padding_ratio:.1f}%")
            
            return bbox
        else:
            print("Could not analyze content!")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main analysis function"""
    print("="*70)
    print("OASIS Axial Slice Optimal Size Analysis")
    print("="*70)
    
    # Path to processed cache data
    project_root = Path(__file__).parent
    cache_root = project_root / 'data' / 'processed_oasis_3d_cv5_v2' / '_cached'
    if not cache_root.exists():
        cache_root = project_root / 'data' / 'processed_oasis_3d_cv5_v2' / '_cache'

    if not cache_root.exists():
        print(f"\nError: Processed cache directory not found at {cache_root}")
        print("Expected either .../processed_oasis_3d_cv5_v2/_cached or _cache")
        return

    nifti_paths = collect_cached_nifti_files(cache_root, max_subjects=400)
    if not nifti_paths:
        print(f"\nNo .nii.gz files found in {cache_root}")
        return

    print(f"\nFound {len(nifti_paths)} processed NIfTI subjects to analyze")

    # Visualize NIfTI axis-size distributions.
    shape_plot_path = project_root / 'docs' / 'nifti_axis_size_distribution.png'
    visualize_nifti_shape_distribution(nifti_paths, shape_plot_path)
    
    # Analyze each subject
    results = []
    for path in nifti_paths:
        subject_id = path.stem.replace('.nii', '')
        bbox = analyze_subject(str(path), subject_id)
        if bbox:
            results.append(bbox)
    
    # Summary statistics
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        
        widths = [r['width'] for r in results]
        heights = [r['height'] for r in results]
        max_dims = [r['max_dim'] for r in results]
        
        print(f"\nWidth statistics:")
        print(f"  Min: {np.min(widths):.0f}, Max: {np.max(widths):.0f}")
        print(f"  Mean: {np.mean(widths):.1f}, Median: {np.median(widths):.1f}")
        print(f"  Std: {np.std(widths):.1f}")
        
        print(f"\nHeight statistics:")
        print(f"  Min: {np.min(heights):.0f}, Max: {np.max(heights):.0f}")
        print(f"  Mean: {np.mean(heights):.1f}, Median: {np.median(heights):.1f}")
        print(f"  Std: {np.std(heights):.1f}")
        
        print(f"\nMax dimension statistics:")
        print(f"  Min: {np.min(max_dims):.0f}, Max: {np.max(max_dims):.0f}")
        print(f"  Mean: {np.mean(max_dims):.1f}, Median: {np.median(max_dims):.1f}")
        print(f"  Std: {np.std(max_dims):.1f}")
        
        # Recommendation
        median_max = np.median(max_dims)
        print(f"\n{'='*70}")
        print("RECOMMENDATION")
        print(f"{'='*70}")
        
        if median_max <= 200:
            recommended = 224
        elif median_max <= 220:
            recommended = 240
        else:
            recommended = 256
        
        print(f"\nBased on median max dimension of {median_max:.1f} pixels:")
        print(f"  → Recommended target size: {recommended}x{recommended}")
        
        # Show distribution
        print(f"\nSubjects by recommended size:")
        for target in [224, 240, 256]:
            count = sum(1 for d in max_dims if d <= target and (target == 256 or d > target - 32))
            print(f"  {target}x{target}: {count} subjects")
        
        # Create visualization
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].hist(widths, bins=20, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Width (pixels)')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Axial Slice Width Distribution')
            axes[0].axvline(np.median(widths), color='red', linestyle='--', label=f'Median: {np.median(widths):.1f}')
            axes[0].legend()
            
            axes[1].hist(heights, bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Height (pixels)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Axial Slice Height Distribution')
            axes[1].axvline(np.median(heights), color='red', linestyle='--', label=f'Median: {np.median(heights):.1f}')
            axes[1].legend()
            
            axes[2].hist(max_dims, bins=20, edgecolor='black', alpha=0.7)
            axes[2].set_xlabel('Max Dimension (pixels)')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Max Dimension Distribution')
            axes[2].axvline(np.median(max_dims), color='red', linestyle='--', label=f'Median: {np.median(max_dims):.1f}')
            for target in [224, 240, 256]:
                axes[2].axvline(target, color='green', linestyle=':', alpha=0.5, label=f'{target}')
            axes[2].legend()
            
            plt.tight_layout()
            
            output_path = Path(__file__).parent / 'docs' / 'optimal_size_analysis.png'
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
            
        except Exception as e:
            print(f"\nCould not create visualization: {e}")


if __name__ == '__main__':
    main()
