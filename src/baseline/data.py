"""
Dataset for 3D MRI classification (NIfTI only).
"""
import os
import random
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import rotate, zoom
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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
        transform_2d=None,
        rgb_mode=False,
        classes=None,
        class_map=None,
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
        self.transform_2d = transform_2d
        self.rgb_mode = rgb_mode
        self.classes = classes
        self.class_map = class_map
        
        self.samples = []
        self.labels = []
        
        # Load samples
        if self.class_map:
            class_to_idx = dict(self.class_map)
            classes = list(self.class_map.keys())
        elif self.classes:
            classes = list(self.classes)
            class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            classes = ["normal", "alzheimer"]
            class_to_idx = {"normal": 0, "alzheimer": 1}
        
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
            counts = {name: 0 for name in classes}
            for label in self.labels:
                for name, idx in class_to_idx.items():
                    if label == idx:
                        counts[name] += 1
                        break
            print("  " + ", ".join([f"{k}: {v}" for k, v in counts.items()]))
    
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
            if self.transform_2d is not None:
                slice_tensors = []
                for s in slices:
                    s_min, s_max = float(s.min()), float(s.max())
                    if s_max > s_min:
                        s = (s - s_min) / (s_max - s_min)
                    s = (s * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(s, mode="L")
                    if self.rgb_mode:
                        pil_img = pil_img.convert("RGB")
                    t = self.transform_2d(pil_img)
                    if self.rgb_mode and t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    slice_tensors.append(t)
                img_tensor = torch.stack(slice_tensors, dim=0)
            else:
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


_JPG_SLICE_RE = re.compile(r"^(?P<stem>.+)_(?P<view>[a-z]+)_(?P<idx>-?\d+)$", re.IGNORECASE)


class MRIVolumeJPGDataset(Dataset):
    """
    Dataset for 3D volume classification from stacked JPG slices.

    Expected layout: root_dir/<class_name>/*.jpg with filenames like
    "{subject_id}_ax_012.jpg" (view in {ax,sag,cor}).
    """

    def __init__(
        self,
        root_dir: str,
        classes: list[str] | None = None,
        class_map: dict[str, int] | None = None,
        view: str = "ax",
        num_slices: int = 64,
        slice_strategy: str = "uniform",
        image_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
        subject_ids: set[str] | None = None,
    ):
        self.root_dir = root_dir
        self.view = view.lower()
        self.num_slices = int(num_slices) if num_slices is not None else 0
        self.slice_strategy = slice_strategy
        self.image_size = int(image_size)
        self.augment = augment
        self.normalize = normalize
        self.subject_ids = set(subject_ids) if subject_ids else None

        self.samples: list[list[str]] = []
        self.labels: list[int] = []

        class_dirs = []
        if class_map:
            for name in class_map:
                class_dirs.append(name)
        elif classes:
            class_dirs = list(classes)
        else:
            class_dirs = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])

        for class_name in class_dirs:
            class_dir = Path(root_dir) / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue

            grouped: dict[str, list[tuple[int, str]]] = {}
            for path in class_dir.glob("*.jpg"):
                stem = path.stem
                match = _JPG_SLICE_RE.match(stem)
                if not match:
                    continue
                view = match.group("view").lower()
                if self.view and view != self.view:
                    continue
                subject_id = match.group("stem")
                idx = int(match.group("idx"))
                grouped.setdefault(subject_id, []).append((idx, str(path)))

            for subject_id, items in grouped.items():
                if self.subject_ids is not None and subject_id not in self.subject_ids:
                    continue
                items_sorted = sorted(items, key=lambda x: x[0])
                if class_map:
                    label = class_map[class_name]
                else:
                    label = class_dirs.index(class_name)
                self.samples.append([p for _, p in items_sorted])
                self.labels.append(label)

        print(f"Loaded {len(self.samples)} subjects from {root_dir} (view={self.view})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        slice_paths = self.samples[idx]
        label = self.labels[idx]

        selected = self._select_slices(slice_paths)
        volume = self._load_slices(selected)

        if self.normalize:
            volume = self._normalize(volume)
        if self.augment:
            volume = self._augment(volume)

        # Return (S, C, H, W) for 2D slice models.
        img_tensor = torch.from_numpy(volume).float().unsqueeze(1)  # (D, 1, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

    def _select_slices(self, slice_paths: list[str]) -> list[str]:
        total = len(slice_paths)
        if total == 0:
            raise ValueError("No slices found for subject.")
        if self.num_slices <= 0 or self.num_slices == total:
            return slice_paths
        if total < self.num_slices:
            # pad by repeating last slice
            pad_count = self.num_slices - total
            return slice_paths + [slice_paths[-1]] * pad_count

        if self.slice_strategy == "random":
            indices = np.random.choice(total, size=self.num_slices, replace=False)
            indices = sorted(indices.tolist())
        elif self.slice_strategy == "center":
            center = total // 2
            half = self.num_slices // 2
            start = max(0, center - half)
            indices = list(range(start, start + self.num_slices))
            indices = [min(total - 1, max(0, i)) for i in indices]
        else:
            indices = [int(round(i)) for i in np.linspace(0, total - 1, self.num_slices)]

        return [slice_paths[i] for i in indices]

    def _load_slices(self, slice_paths: list[str]) -> np.ndarray:
        slices = []
        for path in slice_paths:
            img = Image.open(path).convert("L")
            if self.image_size:
                img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            slices.append(arr)
        return np.stack(slices, axis=0)

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        mean = float(volume.mean())
        std = float(volume.std())
        if std > 0:
            volume = (volume - mean) / std
        return volume

    @staticmethod
    def _augment(volume: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()
        if random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()
        if random.random() > 0.5:
            volume = np.flip(volume, axis=0).copy()
        if random.random() > 0.5:
            volume = volume * random.uniform(0.9, 1.1)
        return volume


def get_dataloaders(train_dir, val_dir, batch_size=4, num_workers=4,
                   target_shape=(64, 64, 64),
                   weighted_sampler=False, use_2d=False,
                   num_slices=8, slice_axis=0, resize_2d=None,
                   slice_strategy_train="random", slice_strategy_val="uniform",
                   imagenet_norm=False, randaugment=False,
                   randaugment_ops=2, randaugment_mag=9, rgb_mode=False,
                   normalize=True,
                   data_format="nifti",
                   classes=None,
                   class_map=None,
                   jpg_view="ax",
                   image_size=None,
                   train_subject_ids=None,
                   val_subject_ids=None):
    """
    Create dataloaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of workers
        target_shape: Target MRI shape
        weighted_sampler: Use weighted sampler for class imbalance
        use_2d: Return 2D slices instead of 3D volumes
        num_slices: Number of slices per volume (2D mode)
        slice_axis: Axis to slice along (0, 1, 2)
        slice_strategy_train: Slice sampling strategy for train
        slice_strategy_val: Slice sampling strategy for val
        resize_2d: Optional (H, W) to resize 2D slices
        imagenet_norm: Apply ImageNet normalization (2D mode)
        randaugment: Use RandAugment (2D mode)
        randaugment_ops: RandAugment ops count
        randaugment_mag: RandAugment magnitude
        rgb_mode: Convert grayscale to 3-channel
        normalize: Apply z-score normalization to 3D volumes
        data_format: 'nifti' or 'jpg'
        classes: Optional list of class folder names (jpg)
        class_map: Optional mapping {folder_name: label} (jpg)
        jpg_view: Slice view to stack from jpgs (ax|sag|cor)
        image_size: Resize jpg slices to this size (jpg)
        train_subject_ids: Optional set/list of subject IDs to include in train
        val_subject_ids: Optional set/list of subject IDs to include in val
    
    Returns:
        train_loader, val_loader
    """
    
    transform_2d_train = None
    transform_2d_val = None
    if use_2d and imagenet_norm:
        ops = []
        if resize_2d is not None:
            ops.append(T.Resize(tuple(resize_2d)))
        if randaugment:
            ops.append(T.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag))
        ops.append(T.ToTensor())
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_2d_train = T.Compose(ops)

        ops_val = []
        if resize_2d is not None:
            ops_val.append(T.Resize(tuple(resize_2d)))
        ops_val.append(T.ToTensor())
        ops_val.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_2d_val = T.Compose(ops_val)

    if data_format == "jpg":
        if image_size is None:
            raise ValueError("image_size must be set when data_format='jpg'")
        train_dataset = MRIVolumeJPGDataset(
            train_dir,
            classes=classes,
            class_map=class_map,
            view=jpg_view,
            num_slices=num_slices,
            slice_strategy=slice_strategy_train,
            image_size=image_size,
            augment=True,
            normalize=normalize,
            subject_ids=train_subject_ids,
        )
        val_dataset = MRIVolumeJPGDataset(
            val_dir,
            classes=classes,
            class_map=class_map,
            view=jpg_view,
            num_slices=num_slices,
            slice_strategy=slice_strategy_val,
            image_size=image_size,
            augment=False,
            normalize=normalize,
            subject_ids=val_subject_ids,
        )
    else:
        train_dataset = MRIClassificationDataset(
            train_dir,
            target_shape,
            augment=True,
            normalize=normalize,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_train,
            resize_2d=resize_2d,
            transform_2d=transform_2d_train,
            rgb_mode=rgb_mode,
            classes=classes,
            class_map=class_map,
        )
        val_dataset = MRIClassificationDataset(
            val_dir,
            target_shape,
            augment=False,
            normalize=normalize,
            use_2d=use_2d,
            num_slices=num_slices,
            slice_axis=slice_axis,
            slice_strategy=slice_strategy_val,
            resize_2d=resize_2d,
            transform_2d=transform_2d_val,
            rgb_mode=rgb_mode,
            classes=classes,
            class_map=class_map,
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
