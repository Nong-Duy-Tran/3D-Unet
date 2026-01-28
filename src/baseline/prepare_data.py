"""
Data preparation script
Converts NIfTI files to HDF5 format for pytorch-3dunet compatibility
"""
import os
import h5py
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm


def convert_nifti_to_hdf5(nifti_path, output_path, label):
    """
    Convert a single NIfTI file to HDF5 format
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        output_path: Output .h5 file path
        label: Classification label (0 for normal, 1 for alzheimer)
    """
    # Load NIfTI
    nii_img = nib.load(nifti_path)
    img_data = nii_img.get_fdata()
    
    # Ensure 3D
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('raw', data=img_data, compression='gzip')
        f.create_dataset('label', data=label)


def prepare_dataset(input_dir, output_dir, split='train'):
    """
    Prepare dataset by converting NIfTI files to HDF5
    
    Args:
        input_dir: Input directory with class subdirectories
        output_dir: Output directory for HDF5 files
        split: 'train' or 'val'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    classes = {
        'normal': 0,
        'alzheimer': 1
    }
    
    file_count = 0
    
    for class_name, label in classes.items():
        class_dir = os.path.join(input_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith(('.nii', '.nii.gz'))]
        
        print(f"\nProcessing {class_name} ({len(files)} files)...")
        
        for filename in tqdm(files):
            nifti_path = os.path.join(class_dir, filename)
            
            # Create output filename
            base_name = os.path.splitext(filename)[0]
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]
            
            output_filename = f"{split}_{class_name}_{base_name}.h5"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                convert_nifti_to_hdf5(nifti_path, output_path, label)
                file_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal files converted: {file_count}")


def main(args):
    print("=" * 60)
    print("MRI Data Preparation for Alzheimer's Classification")
    print("=" * 60)
    
    # Expected input structure:
    # input_dir/
    #   train/
    #     normal/
    #     alzheimer/
    #   val/
    #     normal/
    #     alzheimer/
    
    # Prepare training data
    train_input = os.path.join(args.input_dir, 'train')
    train_output = os.path.join(args.output_dir, 'train')
    
    if os.path.exists(train_input):
        print(f"\nPreparing training data...")
        prepare_dataset(train_input, train_output, split='train')
    else:
        print(f"Training directory not found: {train_input}")
    
    # Prepare validation data
    val_input = os.path.join(args.input_dir, 'val')
    val_output = os.path.join(args.output_dir, 'val')
    
    if os.path.exists(val_input):
        print(f"\nPreparing validation data...")
        prepare_dataset(val_input, val_output, split='val')
    else:
        print(f"Validation directory not found: {val_input}")
    
    print("\n" + "=" * 60)
    print("Data preparation completed!")
    print(f"HDF5 files saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare MRI data for training')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing train/val subdirectories')
    parser.add_argument('--output_dir', type=str, default='./data_hdf5',
                       help='Output directory for HDF5 files')
    
    args = parser.parse_args()
    main(args)
