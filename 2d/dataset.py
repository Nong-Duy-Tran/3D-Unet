"""
2D Dataset for OASIS Alzheimer's Disease classification
Handles preprocessed 2D slices (224x224x120) stored in .npz format
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class OASIS2DDataset(Dataset):
    """
    Dataset for 2D OASIS brain MRI classification
    
    Each sample consists of 120 axial slices of size 224x224
    
    Args:
        data_dir: Directory containing fold_X/train.npz or fold_X/val.npz
        fold: Fold number (0-4)
        split: 'train' or 'val'
        transform: Optional transform to apply to each slice
        slice_transform: Optional transform to apply to all slices together
    """
    
    def __init__(self, data_dir, fold=0, split='train', transform=None, slice_transform=None):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.transform = transform
        self.slice_transform = slice_transform
        
        # Load data from .npz file
        fold_dir = self.data_dir / f'fold_{fold}'
        npz_path = fold_dir / f'{split}.npz'
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        
        # Load numpy arrays
        data = np.load(npz_path)
        self.images = data['images']  # Shape: (N, 224, 224, 120)
        self.labels = data['labels']  # Shape: (N,)
        self.subject_ids = data['subject_ids']  # List of subject IDs
        
        # Load metadata
        metadata_path = fold_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print(f"Loaded {split} data for fold {fold}: {len(self.images)} subjects")
        print(f"  Shape: {self.images.shape}")
        print(f"  Normal: {np.sum(self.labels == 0)}, Alzheimer: {np.sum(self.labels == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (120, 1, 224, 224) - 120 slices, each with 1 channel
            label: Tensor scalar (0 or 1)
            subject_id: String subject ID
        """
        # Get image and label
        image = self.images[idx]  # Shape: (224, 224, 120)
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        
        # Apply slice-wise transform if provided
        if self.transform is not None:
            # Transform each slice independently
            slices = []
            for i in range(image.shape[2]):
                slice_2d = image[:, :, i]
                # Add channel dimension for transform
                slice_2d = slice_2d[np.newaxis, :, :]  # (1, 224, 224)
                slice_2d = self.transform(slice_2d)
                slices.append(slice_2d)
            image = torch.stack(slices, dim=0)  # (120, 1, 224, 224)
        else:
            # Convert to tensor
            # Rearrange from (224, 224, 120) to (120, 1, 224, 224)
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()
        
        # Apply transform to all slices together if provided
        if self.slice_transform is not None:
            image = self.slice_transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label, subject_id
    
    def get_class_distribution(self):
        """Get class distribution for the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_subject_info(self, idx):
        """Get detailed information about a subject"""
        return {
            'subject_id': self.subject_ids[idx],
            'label': int(self.labels[idx]),
            'class_name': 'normal' if self.labels[idx] == 0 else 'alzheimer',
            'image_shape': self.images[idx].shape
        }


class OASIS2DSliceDataset(Dataset):
    """
    Dataset that treats each slice independently (for slice-level models)
    
    Each sample is a single 224x224 slice
    Useful for 2D CNN models that don't use attention across slices
    
    Args:
        data_dir: Directory containing fold_X/train.npz or fold_X/val.npz
        fold: Fold number (0-4)
        split: 'train' or 'val'
        transform: Optional transform to apply
    """
    
    def __init__(self, data_dir, fold=0, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.transform = transform
        
        # Load data from .npz file
        fold_dir = self.data_dir / f'fold_{fold}'
        npz_path = fold_dir / f'{split}.npz'
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        
        # Load numpy arrays
        data = np.load(npz_path)
        images = data['images']  # Shape: (N, 224, 224, 120)
        labels = data['labels']  # Shape: (N,)
        subject_ids = data['subject_ids']
        
        # Flatten slices: each slice becomes a separate sample
        # (N, 224, 224, 120) -> (N*120, 224, 224)
        self.slices = images.reshape(-1, images.shape[1], images.shape[2])
        
        # Repeat labels for each slice
        self.labels = np.repeat(labels, images.shape[3])
        
        # Track which subject each slice belongs to
        self.subject_indices = np.repeat(np.arange(len(labels)), images.shape[3])
        self.subject_ids = [subject_ids[i] for i in self.subject_indices]
        
        print(f"Loaded {split} slice data for fold {fold}: {len(self.slices)} slices from {len(images)} subjects")
        print(f"  Normal: {np.sum(self.labels == 0)}, Alzheimer: {np.sum(self.labels == 1)}")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        """
        Returns:
            slice: Tensor of shape (1, 224, 224)
            label: Tensor scalar (0 or 1)
        """
        slice_2d = self.slices[idx]
        label = self.labels[idx]
        
        # Add channel dimension
        slice_2d = slice_2d[np.newaxis, :, :]  # (1, 224, 224)
        
        if self.transform is not None:
            slice_2d = self.transform(slice_2d)
        else:
            slice_2d = torch.from_numpy(slice_2d).float()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return slice_2d, label


def get_dataloader(data_dir, fold=0, split='train', batch_size=4, 
                   num_workers=4, transform=None, slice_transform=None,
                   slice_level=False):
    """
    Factory function to create dataloader
    
    Args:
        data_dir: Directory containing processed data
        fold: Fold number (0-4)
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        transform: Transform to apply to each slice
        slice_transform: Transform to apply to all slices together
        slice_level: If True, use slice-level dataset instead of volume-level
    
    Returns:
        DataLoader
    """
    if slice_level:
        dataset = OASIS2DSliceDataset(
            data_dir=data_dir,
            fold=fold,
            split=split,
            transform=transform
        )
    else:
        dataset = OASIS2DDataset(
            data_dir=data_dir,
            fold=fold,
            split=split,
            transform=transform,
            slice_transform=slice_transform
        )
    
    # Shuffle only for training
    shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_cv5',
                        help='Directory containing processed 2D data')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    args = parser.parse_args()
    
    print("Testing OASIS2DDataset...")
    print("=" * 70)
    
    # Test volume-level dataset
    dataset = OASIS2DDataset(args.data_dir, fold=args.fold, split='train')
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test loading a sample
    image, label, subject_id = dataset[0]
    print(f"\nSample 0:")
    print(f"  Subject ID: {subject_id}")
    print(f"  Image shape: {image.shape}")  # Should be (120, 1, 224, 224)
    print(f"  Label: {label.item()} ({'Normal' if label == 0 else 'Alzheimer'})")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Test subject info
    info = dataset.get_subject_info(0)
    print(f"\nSubject info: {info}")
    
    # Test dataloader
    print("\n" + "=" * 70)
    print("Testing DataLoader...")
    dataloader = get_dataloader(args.data_dir, fold=args.fold, split='train', 
                                batch_size=2, num_workers=0)
    
    for i, (images, labels, subject_ids) in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Images shape: {images.shape}")  # Should be (2, 120, 1, 224, 224)
        print(f"  Labels: {labels}")
        print(f"  Subject IDs: {subject_ids}")
        
        if i >= 1:  # Only test 2 batches
            break
    
    # Test slice-level dataset
    print("\n" + "=" * 70)
    print("Testing OASIS2DSliceDataset...")
    
    slice_dataset = OASIS2DSliceDataset(args.data_dir, fold=args.fold, split='train')
    print(f"\nSlice dataset size: {len(slice_dataset)}")
    
    slice_img, slice_label = slice_dataset[0]
    print(f"\nSample slice:")
    print(f"  Shape: {slice_img.shape}")  # Should be (1, 224, 224)
    print(f"  Label: {slice_label.item()}")
    
    print("\n" + "=" * 70)
    print("Dataset tests completed successfully!")
