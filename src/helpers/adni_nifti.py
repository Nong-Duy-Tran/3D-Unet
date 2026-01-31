"""Helpers for ADNI NIfTI processing (no HDF5)."""
from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib


def extract_image_id(path: Path) -> str | None:
    match = re.search(r"I\d{4,}", path.name)
    if match:
        return match.group(0)
    for part in path.parts:
        match = re.search(r"I\d{4,}", part)
        if match:
            return match.group(0)
    return None


def load_nifti(path: Path, reorient_ras: bool = True):
    nii_img = nib.load(str(path))
    if reorient_ras:
        nii_img = nib.as_closest_canonical(nii_img)
    img_data = nii_img.get_fdata()
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    return img_data
