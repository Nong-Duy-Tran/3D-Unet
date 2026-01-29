"""
Dataset for 3D MRI classification
Supports both NIfTI files and HDF5 format
"""
import os
import h5py
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.ndimage import zoom, rotate
import random


class MRIClassificationDataset(Dataset):
    """
    Dataset for MRI classification from NIfTI files
    
    Args:
        root_dir: Directory containing class subdirectories
        target_shape: Target shape for resizing (D, H, W)
        augment: Whether to apply augmentation
        normalize: Whether to normalize data
    """
    
    def __init__(
        self,
        root_dir,
        target_shape=(64, 64, 64),
        augment=False,
        normalize=True,
        use_2d=False,
        num_slices=8,
        slice_axis=0,
        slice_strategy="uniform",
        resize_2d=None,
    ):
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.augment = augment
        self.normalize = normalize
        self.use_2d = use_2d
        self.num_slices = num_slices
        self.slice_axis = slice_axis
        self.slice_strategy = slice_strategy
        self.resize_2d = resize_2d
        
        self.samples = []
        self.labels = []
        
        # Load samples
        classes = ['normal', 'alzheimer']
        class_to_idx = {'normal': 0, 'alzheimer': 1}
        
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith(('.nii', '.nii.gz')):
                    self.samples.append(os.path.join(class_dir, filename))
                    self.labels.append(class_to_idx[class_name])
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        if len(self.samples) > 0:
            print(f"  Normal: {self.labels.count(0)}, Alzheimer: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load MRI
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        nii_img = nib.load(img_path)
        img_data = nii_img.get_fdata()
        
        # Ensure 3D
        if img_data.ndim == 4:
            img_data = img_data[..., 0]
        
        # Resize
        img_data = self._resize(img_data)
        
        # Normalize
        if self.normalize:
            img_data = self._normalize(img_data)
        
        # Augment
        if self.augment:
            img_data = self._augment(img_data)
        
        # To tensor
        if self.use_2d:
            slices = self._extract_slices(img_data)
            img_tensor = torch.from_numpy(slices).float().unsqueeze(1)  # (S, 1, H, W)
        else:
            img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor
    
    def _resize(self, img):
        """Resize image to target shape"""
        zoom_factors = [t / s for t, s in zip(self.target_shape, img.shape)]
        return zoom(img, zoom_factors, order=1)
    
    def _normalize(self, img):
        """Z-score normalization"""
        img = img.astype(np.float32)
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
        return img
    
    def _augment(self, img):
        """Data augmentation"""
        # Random flips
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            img = rotate(img, angle, axes=axes, reshape=False, order=1)
        
        # Random intensity
        if random.random() > 0.5:
            img = img * random.uniform(0.9, 1.1)
        
        return img

    def _slice_indices(self, size):
        if self.num_slices <= 1:
            return [size // 2]
        if self.slice_strategy == "random":
            return [random.randint(0, size - 1) for _ in range(self.num_slices)]
        if self.slice_strategy == "center":
            center = size // 2
            half = self.num_slices // 2
            start = max(0, center - half)
            indices = list(range(start, start + self.num_slices))
            return [min(size - 1, max(0, i)) for i in indices]
        # uniform
        return [int(round(i)) for i in np.linspace(0, size - 1, self.num_slices)]

    def _extract_slices(self, img):
        axis = int(self.slice_axis)
        if axis == 0:
            size = img.shape[0]
        elif axis == 1:
            size = img.shape[1]
        else:
            size = img.shape[2]
        indices = self._slice_indices(size)
        slices = []
        for idx in indices:
            if axis == 0:
                slices.append(img[idx, :, :])
            elif axis == 1:
                slices.append(img[:, idx, :])
            else:
                slices.append(img[:, :, idx])
        slices = np.stack(slices, axis=0)
        if self.resize_2d is not None:
            target_h, target_w = self.resize_2d
            zoom_factors = [target_h / slices.shape[1], target_w / slices.shape[2]]
            resized = []
            for s in slices:
                resized.append(zoom(s, zoom_factors, order=1))
            slices = np.stack(resized, axis=0)
        return slices


class MRIClassificationHDF5Dataset(Dataset):
    """
    Dataset for MRI classification from HDF5 files
    Each HDF5 file should contain 'raw' (3D MRI) and 'label' (0 or 1)
    
    Args:
        file_paths: List of HDF5 file paths
        target_shape: Target shape for resizing
        augment: Whether to apply augmentation
    """
    
    def __init__(
        self,
        file_paths,
        target_shape=(64, 64, 64),
        augment=False,
        use_2d=False,
        num_slices=8,
        slice_axis=0,
        slice_strategy="uniform",
        resize_2d=None,
    ):
        self.file_paths = file_paths
        self.target_shape = target_shape
        self.augment = augment
        self.use_2d = use_2d
        self.num_slices = num_slices
        self.slice_axis = slice_axis
        self.slice_strategy = slice_strategy
        self.resize_2d = resize_2d
        self.labels = []

        for file_path in self.file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    self.labels.append(int(f['label'][()]))
            except Exception:
                self.labels.append(0)

        print(f"Loaded {len(self.file_paths)} HDF5 files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Load raw data
            img_data = f['raw'][:]
            label = int(f['label'][()])
            
            # Ensure 3D
            if img_data.ndim == 4:
                img_data = img_data[0]  # Take first channel
            
            # Resize if needed
            if img_data.shape != self.target_shape:
                zoom_factors = [t / s for t, s in zip(self.target_shape, img_data.shape)]
                img_data = zoom(img_data, zoom_factors, order=1)
            
            # Normalize
            img_data = img_data.astype(np.float32)
            mean = np.mean(img_data)
            std = np.std(img_data)
            if std > 0:
                img_data = (img_data - mean) / std
            
            # Augment
            if self.augment:
                img_data = self._augment(img_data)
        
        if self.use_2d:
            slices = self._extract_slices(img_data)
            img_tensor = torch.from_numpy(slices).float().unsqueeze(1)
        else:
            img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

    def _augment(self, img):
        """Data augmentation"""
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
        return img

    def _slice_indices(self, size):
        if self.num_slices <= 1:
            return [size // 2]
        if self.slice_strategy == "random":
            return [random.randint(0, size - 1) for _ in range(self.num_slices)]
        if self.slice_strategy == "center":
            center = size // 2
            half = self.num_slices // 2
            start = max(0, center - half)
            indices = list(range(start, start + self.num_slices))
            return [min(size - 1, max(0, i)) for i in indices]
        return [int(round(i)) for i in np.linspace(0, size - 1, self.num_slices)]

    def _extract_slices(self, img):
        axis = int(self.slice_axis)
        if axis == 0:
            size = img.shape[0]
        elif axis == 1:
            size = img.shape[1]
        else:
            size = img.shape[2]
        indices = self._slice_indices(size)
        slices = []
        for idx in indices:
            if axis == 0:
                slices.append(img[idx, :, :])
            elif axis == 1:
                slices.append(img[:, idx, :])
            else:
                slices.append(img[:, :, idx])
        slices = np.stack(slices, axis=0)
        if self.resize_2d is not None:
            target_h, target_w = self.resize_2d
            zoom_factors = [target_h / slices.shape[1], target_w / slices.shape[2]]
            resized = []
            for s in slices:
                resized.append(zoom(s, zoom_factors, order=1))
            slices = np.stack(resized, axis=0)
        return slices


def get_dataloaders(train_dir, val_dir, batch_size=4, num_workers=4,
                   target_shape=(64, 64, 64), use_hdf5=False,
                   weighted_sampler=False, use_2d=False,
                   num_slices=8, slice_axis=0, resize_2d=None,
                   slice_strategy_train="random", slice_strategy_val="uniform"):
    """
    Create dataloaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of workers
        target_shape: Target MRI shape
        use_hdf5: Use HDF5 dataset instead of NIfTI
        weighted_sampler: Use weighted sampler for class imbalance
        use_2d: Return 2D slices instead of 3D volumes
        num_slices: Number of slices per volume (2D mode)
        slice_axis: Axis to slice along (0, 1, 2)
        slice_strategy_train: Slice sampling strategy for train
        slice_strategy_val: Slice sampling strategy for val
        resize_2d: Optional (H, W) to resize 2D slices
    
    Returns:
        train_loader, val_loader
    """
    
    if use_hdf5:
        # Get HDF5 file paths
        train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]
        val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.h5')]
        
        train_dataset = MRIClassificationHDF5Dataset(
            train_files,
            target_shape,
            augment=True,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_train,
            resize_2d=resize_2d,
        )
        val_dataset = MRIClassificationHDF5Dataset(
            val_files,
            target_shape,
            augment=False,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_val,
            resize_2d=resize_2d,
        )
    else:
        train_dataset = MRIClassificationDataset(
            train_dir,
            target_shape,
            augment=True,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_train,
            resize_2d=resize_2d,
        )
        val_dataset = MRIClassificationDataset(
            val_dir,
            target_shape,
            augment=False,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_val,
            resize_2d=resize_2d,
        )
    
    sampler = None
    shuffle = True
    if weighted_sampler:
        if not hasattr(train_dataset, "labels") or not train_dataset.labels:
            raise ValueError("Weighted sampler requires dataset labels to be available.")
        labels = np.array(train_dataset.labels, dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing MRI Classification Dataset...")
    
    # Test with dummy data directory
    train_dir = "../data/train"
    if os.path.exists(train_dir):
        dataset = MRIClassificationDataset(train_dir, target_shape=(64, 64, 64))
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Label: {label}")
