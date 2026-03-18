import os
import torch
import json
import argparse
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from pathlib import Path


class _ImageAccessor:
    """Lazy per-subject image accessor returning (S, H, W) numpy arrays."""
    def __init__(self, dataset):
        self._ds = dataset

    def __getitem__(self, idx):
        subject_id = self._ds.subject_ids[idx]
        slices = []
        for v in self._ds.load_views:
            for fpath in self._ds.file_index[subject_id][v]:
                img = np.array(Image.open(fpath))  # (H, W), already grayscale
                slices.append(img)
        return np.stack(slices, axis=0)  # (S, H, W)


class OASIS2DDataset(Dataset):
    """
    Dataset for OASIS MRI classification (Alzheimer vs Normal) from JPEG images.

    Directory layout expected:
        data_dir/
            axial/fold_{fold}/{split}/normal/*.jpg
            axial/fold_{fold}/{split}/nonnormal/*.jpg
            coronal/fold_{fold}/{split}/normal/*.jpg
            coronal/fold_{fold}/{split}/nonnormal/*.jpg
            sagittal/fold_{fold}/{split}/normal/*.jpg
            sagittal/fold_{fold}/{split}/nonnormal/*.jpg

    File naming convention:
        {subject_id}_{view}_{slice_index:03d}.jpg
        e.g. OAS1_0001_MR1_axial_007.jpg

    Each subject has 80 slices per view.
    view='axial'    → 80 slices
    view='coronal'  → 80 slices
    view='sagittal' → 80 slices
    view='all'      → 240 slices (axial + coronal + sagittal)

    __getitem__ returns: (S, 1, H, W) float32 tensor, label, subject_id

    Args:
        data_dir      : Root directory (contains axial/, coronal/, sagittal/).
        fold          : Fold index (0–4).
        split         : 'train', 'val', or 'test'.
        view          : 'all', 'axial', 'coronal', or 'sagittal'.
        one_transform : Optional transform applied to each PIL slice image.
        all_transform : Optional transform applied to the full (S, 1, H, W) tensor.
    """

    VIEWS = ['axial', 'coronal', 'sagittal']
    CLASSES = {'normal': 0, 'nonnormal': 1}

    def __init__(self, data_dir, fold=0, split='train', view='all',
                 one_transform=None, all_transform=None):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.view = view
        self.one_transform = one_transform
        self.all_transform = all_transform

        self.load_views = self.VIEWS if view == 'all' else [view]

        # ------------------------------------------------------------------ #
        # Collect subjects and labels from the primary view folder            #
        # ------------------------------------------------------------------ #
        primary_view = self.load_views[0]
        subject_label_map = {}  # subject_id -> label

        for cls_name, label in self.CLASSES.items():
            cls_dir = self.data_dir / primary_view / f'fold_{fold}' / split / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Directory not found: {cls_dir}")
            for fpath in sorted(cls_dir.glob('*.jpg')):
                # Filename: OAS1_0001_MR1_axial_000.jpg
                # subject_id is everything before the last two '_'-delimited parts
                parts = fpath.stem.split('_')   # e.g. ['OAS1', '0001', 'MR1', 'axial', '000']
                subject_id = '_'.join(parts[:3])  # OAS1_0001_MR1
                subject_label_map[subject_id] = label

        self.subject_ids = sorted(subject_label_map.keys())
        self.labels = np.array(
            [subject_label_map[s] for s in self.subject_ids], dtype=np.int64
        )

        # ------------------------------------------------------------------ #
        # Build file index: subject_id -> {view: [sorted Path list]}         #
        # ------------------------------------------------------------------ #
        self.file_index = {}
        for subject_id in self.subject_ids:
            label = subject_label_map[subject_id]
            cls_name = 'normal' if label == 0 else 'nonnormal'
            self.file_index[subject_id] = {}
            for v in self.load_views:
                view_dir = self.data_dir / v / f'fold_{fold}' / split / cls_name
                files = sorted(view_dir.glob(f'{subject_id}_{v}_*.jpg'))
                if not files:
                    raise FileNotFoundError(
                        f"No JPEG files found for subject '{subject_id}' "
                        f"view '{v}' in {view_dir}"
                    )
                self.file_index[subject_id][v] = files

        # ------------------------------------------------------------------ #
        # Lazy images accessor (for external visualization code)              #
        # ------------------------------------------------------------------ #
        self.images = _ImageAccessor(self)

        # Slice count (from first subject, first view)
        first_subj = self.subject_ids[0]
        self._n_slices_per_view = len(self.file_index[first_subj][self.load_views[0]])
        self._n_slices_total = self._n_slices_per_view * len(self.load_views)

        print(f"\n--- Loaded {split} Fold {fold} (view={view}) ---")
        print(f"Subjects      : {len(self.subject_ids)}")
        print(f"Slices/subject: {self._n_slices_total}  "
              f"({self._n_slices_per_view} per view × {len(self.load_views)} views)")
        print(f"  Normal     : {np.sum(self.labels == 0)}")
        print(f"  Alzheimer  : {np.sum(self.labels == 1)}")

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                       #
    # ---------------------------------------------------------------------- #

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image      : Tensor (S, 1, H, W) float32 in [0, 1].
            label      : Long tensor scalar (0=Normal, 1=Alzheimer).
            subject_id : Subject ID string.
        """
        subject_id = self.subject_ids[idx]
        label = int(self.labels[idx])

        slices = []
        for v in self.load_views:
            for fpath in self.file_index[subject_id][v]:
                pil_img = Image.open(fpath)  # already 'L' (grayscale)

                if self.one_transform is not None:
                    pil_img = self.one_transform(pil_img)

                if isinstance(pil_img, Image.Image):
                    arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W)
                    slice_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
                elif torch.is_tensor(pil_img):
                    slice_tensor = pil_img.float()
                    if slice_tensor.ndim == 2:
                        slice_tensor = slice_tensor.unsqueeze(0)
                else:
                    raise TypeError(f"Unsupported type after one_transform: {type(pil_img)}")

                slices.append(slice_tensor)

        image = torch.stack(slices, dim=0)  # (S, 1, H, W)

        if self.all_transform is not None:
            image = self.all_transform(image)

        return image, torch.tensor(label, dtype=torch.long), subject_id

    # ---------------------------------------------------------------------- #
    # Helper methods                                                          #
    # ---------------------------------------------------------------------- #

    def get_class_distribution(self):
        """Return counts of Normal and Alzheimer subjects."""
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        return {
            'normal':    dist.get(0, 0),
            'alzheimer': dist.get(1, 0),
        }

    def get_subject_info(self, idx):
        """Return a summary dict for the subject at the given index."""
        return {
            'subject_id': self.subject_ids[idx],
            'label':      int(self.labels[idx]),
            'class_name': 'Normal' if self.labels[idx] == 0 else 'Alzheimer',
            'n_slices':   self._n_slices_total,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OASIS2DDataset (JPEG)")
    parser.add_argument(
        "--data_dir", type=str,
        default="./data/processed_oasis_2d_jpeg",
        help="Root directory containing axial/, coronal/, sagittal/ sub-folders"
    )
    parser.add_argument("--fold",  type=int, default=0)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--view",  type=str, default="axial",
                        choices=["all", "axial", "coronal", "sagittal"])
    args = parser.parse_args()

    print("=" * 70)
    print("OASIS2DDataset (JPEG) – Test Script")
    print("=" * 70)

    dataset = OASIS2DDataset(
        data_dir=args.data_dir,
        fold=args.fold,
        split=args.split,
        view=args.view,
    )

    print(f"\nTotal subjects : {len(dataset)}")

    dist = dataset.get_class_distribution()
    print(f"Class distribution: Normal={dist['normal']}, Alzheimer={dist['alzheimer']}")

    image, label, subject_id = dataset[0]
    print(f"\nSample 0:")
    print(f"  Subject ID  : {subject_id}")
    print(f"  Image shape : {image.shape}")
    print(f"  Label       : {label.item()} ({'Normal' if label.item() == 0 else 'Alzheimer'})")
    print(f"  Dtype       : {image.dtype}")
    print(f"  Value range : [{image.min():.3f}, {image.max():.3f}]")

    # Lazy images accessor
    images_np = dataset.images[0]
    print(f"\n  images[0] (numpy): shape={images_np.shape}, dtype={images_np.dtype}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
