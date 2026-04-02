"""
Dataset for 3D MRI classification
Uses MONAI dictionary transforms for preprocessing
"""
import os
import torch
import numpy as np

# Fix for MONAI's MAX_SEED overflow issue with numpy >= 2.0
import monai
import monai.transforms.transform
import monai.utils.misc
monai.utils.misc.MAX_SEED = 4294967295
monai.transforms.transform.MAX_SEED = 4294967295

from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Resized,
    NormalizeIntensityd, RandFlipd, RandRotated, Rand3DElasticd,
    RandBiasFieldd, RandGaussianNoised, ToTensord
)

class MRIClassificationDataset(Dataset):
    """
    Dataset for MRI classification from NIfTI files
    
    Args:
        data_dir: Directory containing class subdirectories
        fold: Fold number
        split: Train or val split
        target_shape: Target shape for resizing (D, H, W)
    """
    
    def __init__(self, data_dir, fold=0, split='train', target_shape=(96, 96, 96)):
        self.data_dir = data_dir
        self.fold = fold
        self.split = split
        self.root_dir = os.path.join(data_dir, f'fold_{fold}', split)
        self.target_shape = target_shape
        
        self.samples = []
        self.labels = []
        
        # Load samples
        if os.path.exists(os.path.join(self.root_dir, 'ad')) and os.path.exists(os.path.join(self.root_dir, 'nonad')):
            classes = ['nonad', 'ad']
            class_to_idx = {'nonad': 0, 'ad': 1}
        elif os.path.exists(os.path.join(self.root_dir, 'lgg')) and os.path.exists(os.path.join(self.root_dir, 'hgg')):
            classes = ['lgg', 'hgg']
            class_to_idx = {'lgg': 0, 'hgg': 1}
        else:
            classes = ['normal', 'nonnormal']
            class_to_idx = {'normal': 0, 'nonnormal': 1}
        
        if os.path.exists(self.root_dir):
            for class_name in classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: {class_dir} not found")
                    continue
                
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.nii', '.nii.gz')):
                        self.samples.append(os.path.join(class_dir, filename))
                        self.labels.append(class_to_idx[class_name])
            
            print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
            if len(self.samples) > 0:
                print(f"  {classes[0]}: {self.labels.count(0)}, {classes[1]}: {self.labels.count(1)}")
        else:
            print(f"Warning: Root dir {self.root_dir} not found")

        # Setup MONAI transforms
        if self.split == 'train':
            self.transform = Compose([
                LoadImaged(keys=['image']),
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=self.target_shape, mode='trilinear'),
                NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
                RandRotated(keys=['image'], range_x=0.1, prob=0.5, mode='bilinear'),
                Rand3DElasticd(keys=['image'], sigma_range=(5,8), magnitude_range=(100,200), prob=0.2),
                RandBiasFieldd(keys=['image'], prob=0.3),
                RandGaussianNoised(keys=['image'], prob=0.2),
                EnsureTyped(keys=['image']),
                ToTensord(keys=['image'])
            ])
        else:
            self.transform = Compose([
                LoadImaged(keys=['image']),
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=self.target_shape, mode='trilinear'),
                NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                EnsureTyped(keys=['image']),
                ToTensord(keys=['image'])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Apply MONAI dictionary transforms
        transformed = self.transform({'image': self.samples[idx]})
        
        img_tensor = transformed['image']
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_tensor, label_tensor


def get_dataloader(data_dir, fold=0, split='train', batch_size=4, num_workers=4,
                   target_shape=(96, 96, 96)):
    """
    Create dataloader
    
    Args:
        data_dir: Data directory containing train and val directories
        fold: Fold number (0-4)
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of workers
        target_shape: Target MRI shape
    
    Returns:
        DataLoader
    """
    
    dataset = MRIClassificationDataset(data_dir, fold=fold, split=split, target_shape=target_shape)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    # Test dataset
    print("Testing MRI Classification Dataset...")
    
    # Test with dummy data directory
    data_dir = "../data/OASIS"
    if os.path.exists(data_dir):
        try:
            dataset = MRIClassificationDataset(data_dir, fold=0, split='train', target_shape=(96, 96, 96))
            if len(dataset) > 0:
                img, label = dataset[0]
                print(f"Image shape: {img.shape}")
                print(f"Label: {label}")
        except Exception as e:
            print(f"Error initializing dataset: {e}")
