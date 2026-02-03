from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import nibabel as nib
from nibabel.processing import resample_to_output

from src.utils.image_ops import center_slice_indices, to_uint8


def load_subject_volume(
    img_path: Path,
    reorient_ras: bool = True,
    resample_mm: float | None = 1.0,
) -> np.ndarray:
    img = nib.load(str(img_path))
    if reorient_ras:
        img = nib.as_closest_canonical(img)
    if resample_mm and resample_mm > 0:
        zooms = img.header.get_zooms()[:3]
        if any(abs(z - resample_mm) > 1e-3 for z in zooms):
            img = resample_to_output(img, voxel_sizes=(resample_mm, resample_mm, resample_mm), order=1)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data


def convert_and_copy_subjects(data, output_dir, split_name):
    """
    Convert Analyze format to NIfTI and copy files to output directory.
    """
    split_dir = Path(output_dir) / split_name

    normal_dir = split_dir / 'normal'
    alzheimer_dir = split_dir / 'alzheimer'
    normal_dir.mkdir(parents=True, exist_ok=True)
    alzheimer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(data)} subjects for {split_name} split...")

    file_count = {'normal': 0, 'alzheimer': 0, 'errors': 0}

    for item in tqdm(data):
        try:
            img_file = Path(item['img_file'])
            subject_id = item['subject_id']
            class_name = item['class_name']

            img = nib.load(img_file)

            out_filename = f"{subject_id}.nii.gz"
            if class_name == 'alzheimer':
                out_path = alzheimer_dir / out_filename
            else:
                out_path = normal_dir / out_filename

            nib.save(img, out_path)
            file_count[class_name] += 1

        except Exception as exc:
            print(f"\nError processing {item['subject_id']}: {exc}")
            file_count['errors'] += 1

    print(f"  Normal: {file_count['normal']}, Alzheimer: {file_count['alzheimer']}, Errors: {file_count['errors']}")

    return file_count


def convert_and_copy_subjects_jpg(
    data,
    output_dir,
    split_name,
    image_size=224,
    central_slices=120,
    reorient_ras=True,
    resample_mm=1.0,
):
    """
    Convert Analyze format to JPG slices and copy files to output directory.
    """
    split_dir = Path(output_dir) / split_name

    normal_dir = split_dir / 'normal'
    alzheimer_dir = split_dir / 'alzheimer'
    normal_dir.mkdir(parents=True, exist_ok=True)
    alzheimer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(data)} subjects for {split_name} split (JPG)...")

    file_count = {'normal': 0, 'alzheimer': 0, 'errors': 0}

    for item in tqdm(data):
        try:
            img_file = Path(item['img_file'])
            subject_id = item['subject_id']
            class_name = item['class_name']

            volume = load_subject_volume(
                img_file,
                reorient_ras=reorient_ras,
                resample_mm=resample_mm,
            )
            volume_u8 = to_uint8(volume)

            z_total = volume_u8.shape[2]
            indices = center_slice_indices(z_total, int(central_slices))

            if class_name == 'alzheimer':
                out_dir = alzheimer_dir
            else:
                out_dir = normal_dir

            for idx in indices:
                slice_2d = np.rot90(volume_u8[:, :, idx])
                img = Image.fromarray(slice_2d, mode="L")
                img = img.resize((image_size, image_size), resample=Image.BILINEAR)
                out_path = out_dir / f"{subject_id}_ax_{idx}.jpg"
                img.save(out_path)

            file_count[class_name] += len(indices)

        except Exception as exc:
            print(f"\nError processing {item['subject_id']}: {exc}")
            file_count['errors'] += 1

    print(f"  Normal: {file_count['normal']}, Alzheimer: {file_count['alzheimer']}, Errors: {file_count['errors']}")
    return file_count
