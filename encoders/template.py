from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
import torch

SLICE_PROMPT = (
    "You are a radiology expert analyzing this brain sMRI slice. "
    "Summarize your findings related to Alzheimer's disease-related brain regions."
)

VOLUME_PROMPT = (
    "You are a radiology expert analyzing brain sMRI slices. "
    "Summarize the severity of brain region findings related to Alzheimer's disease. "
    "Output one short paragraph."
)


def to_rgb_like_image(slice_image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    """Convert a grayscale sMRI slice to the 3-channel image format expected by the VLM."""
    if isinstance(slice_image, Image.Image):
        return slice_image.convert("RGB")

    if isinstance(slice_image, torch.Tensor):
        array = slice_image.detach().cpu().numpy()
    else:
        array = np.asarray(slice_image)

    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {array.shape}.")

    array = array.astype(np.float32)
    if array.max() > 1.0 or array.min() < 0.0:
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        else:
            array = np.zeros_like(array, dtype=np.float32)

    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.repeat(array[..., None], 3, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def build_slice_messages(
    slice_image: Image.Image | np.ndarray | torch.Tensor,
    prompt: str = SLICE_PROMPT,
) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": to_rgb_like_image(slice_image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_volume_messages(
    slice_images: Sequence[Image.Image | np.ndarray | torch.Tensor],
    prompt: str = VOLUME_PROMPT,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "image", "image": to_rgb_like_image(slice_image)}
        for slice_image in slice_images
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_text_messages(prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
