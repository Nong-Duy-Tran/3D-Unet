from __future__ import annotations

import numpy as np


def to_uint8(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    vmin, vmax = np.percentile(volume, (1.0, 99.0))
    if vmax <= vmin:
        vmin, vmax = float(volume.min()), float(volume.max())
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    volume = (volume * 255.0).clip(0, 255).astype(np.uint8)
    return volume


def center_slice_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count <= 0:
        return list(range(total))
    if total >= count:
        start = max(0, total // 2 - count // 2)
        end = min(total, start + count)
        if end - start < count:
            start = max(0, end - count)
        return list(range(start, end))
    indices = list(range(total))
    while len(indices) < count:
        indices.append(total - 1)
    return indices
