import numpy as np
from PIL import Image



def to_uint8(volume: np.ndarray) -> np.ndarray:
    """Clip to 1st–99th percentile of non-zero voxels, then scale to uint8."""
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    p1, p99 = np.percentile(nonzero, [1, 99])
    if p99 > p1:
        out = np.clip(volume, p1, p99) # remove outliers
        out = (out - p1) / (p99 - p1) * 255.0
    else:
        out = np.zeros_like(volume, dtype=np.float32)
    return out.astype(np.uint8)


