"""
Script to analyze OASIS data and recommend optimal target size
for axial slice preprocessing

This script:
1. Loads sample OASIS subjects
2. Analyzes axial slice content dimensions across 120 centered slices
3. Recommends optimal target size (224, 240, or 256)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add process_data to path
sys.path.insert(0, str(Path(__file__).parent / 'process_data'))

from process_data.preprocess_oasis_2d import (
    to_ras_orientation, 
    analyze_axial_content, 
    investigate_optimal_size,
    standardize_to_1mm
)


def analyze_subject(img_file, subject_id, num_slices=120):
    """Analyze a single subject's axial content"""
    try:
        print(f"\n{'='*70}")
        print(f"Analyzing: {subject_id}")
        print(f"{'='*70}")
        
        # Load image
        img = nib.load(img_file)
        print(f"Original shape: {img.shape}")
        print(f"Original voxel size: {img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else 'Unknown'}")
        
        # Convert to RAS
        img = to_ras_orientation(img)
        data = np.squeeze(img.get_fdata()).astype(np.float32)
        print(f"After RAS + squeeze: {data.shape}")
        
        # Standardize to 1mm
        voxel_size = img.header.get_zooms()[:3] if hasattr(img.header, 'get_zooms') else (1.0, 1.0, 1.0)
        data = standardize_to_1mm(data, voxel_size)
        print(f"After 1mm standardization: {data.shape}")
        
        # Analyze axial content
        bbox = analyze_axial_content(data, num_slices)
        
        if bbox:
            print(f"\nAxial slice content analysis (across {num_slices} center slices):")
            print(f"  X range: [{bbox['x_min']}, {bbox['x_max']}]  Width: {bbox['width']} pixels")
            print(f"  Z range: [{bbox['z_min']}, {bbox['z_max']}]  Height: {bbox['height']} pixels")
            print(f"  Max dimension: {bbox['max_dim']} pixels")
            
            # Recommend size
            recommended = investigate_optimal_size(data, num_slices)
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
    
    # Path to OASIS data
    oasis_root = Path(__file__).parent / 'data' / 'OASIS'
    
    if not oasis_root.exists():
        print(f"\nError: OASIS directory not found at {oasis_root}")
        print("Please update the path in this script.")
        return
    
    # Find some sample subjects
    sample_subjects = []
    for disc_folder in sorted(oasis_root.iterdir()):
        if disc_folder.is_dir() and disc_folder.name.startswith('disc'):
            for subject_dir in disc_folder.iterdir():
                if subject_dir.is_dir() and subject_dir.name.startswith('OAS1_'):
                    subj_path = subject_dir / 'PROCESSED' / 'MPRAGE' / 'SUBJ_111'
                    if subj_path.exists():
                        for f in subj_path.iterdir():
                            if f.name.endswith('sbj_111.img'):
                                sample_subjects.append({
                                    'id': subject_dir.name,
                                    'path': str(f)
                                })
                                break
                
                if len(sample_subjects) >= 400:  # Analyze 10 subjects
                    break
        
        if len(sample_subjects) >= 400:
            break
    
    if not sample_subjects:
        print("\nNo subjects found! Check OASIS directory structure.")
        return
    
    print(f"\nFound {len(sample_subjects)} sample subjects to analyze")
    
    # Analyze each subject
    results = []
    for subj in sample_subjects:
        bbox = analyze_subject(subj['path'], subj['id'])
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
