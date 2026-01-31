import collections
from typing import Any

import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

from src.modules.pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("Dataset")


class RandomScaler:
    """
    Randomly scales the raw and label patches.

    Args:
        scale_range (int): the maximum absolute value of the scaling factor,
            i.e. patches coordinates will be randomly shifted in the range [-scale_range, scale_range]
        patch_shape (tuple): the shape of the patch DxHxW
        volume_shape (tuple): the shape of the volume DxHxW
        execution_probability (float): the probability of executing the scaling
        seed (int): random seed
    """

    def __init__(
        self,
        scale_range: int,
        patch_shape: tuple,
        volume_shape: tuple,
        execution_probability: float = 0.5,
        seed: int = 47,
    ):
        self.scale_range = scale_range
        self.patch_shape = patch_shape
        self.volume_shape = volume_shape
        self.execution_probability = execution_probability
        self.rs = np.random.RandomState(seed)

    def randomize_indices(self, raw_idx: tuple, label_idx: tuple) -> tuple[tuple, tuple]:
        # execute scaling with a given probability
        if self.rs.uniform() < self.execution_probability:
            return raw_idx, label_idx

        # select random offsets for scaling
        offsets = [self.rs.randint(self.scale_range) for _ in range(3)]
        # change offset sign at random
        if self.rs.rand() > 0.5:
            offsets = [-o for o in offsets]
        # apply offsets to the start or end of the slice at random
        is_start = self.rs.rand() > 0.5
        raw_idx = self._apply_offsets(raw_idx, offsets, is_start)
        label_idx = self._apply_offsets(label_idx, offsets, is_start)

        # assert spatial dimensions are the same
        if len(raw_idx) == 4:
            raw_idx_spacial = raw_idx[1:]
        else:
            raw_idx_spacial = raw_idx
        if len(label_idx) == 4:
            label_idx_spacial = label_idx[1:]
        else:
            label_idx_spacial = label_idx
        assert raw_idx_spacial == label_idx_spacial, (
            f"Raw and label indices are different: {raw_idx_spacial} != {label_idx_spacial}"
        )

        return raw_idx, label_idx

    def rescale_patches(self, raw_patch: torch.Tensor, label_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # compute zoom factors
        if raw_patch.ndim == 4:
            raw_shape = raw_patch.shape[1:]
        else:
            raw_shape = raw_patch.shape

        # if raw_shape equal to self.patch_shape just return the patches
        if raw_shape == self.patch_shape:
            return raw_patch, label_patch

        # rescale patches back to the original shape
        if raw_patch.ndim == 4:
            # add batch dimension
            raw_patch = raw_patch.unsqueeze(0)
            remove_dims = 1
        else:
            # add batch and channels dimensions
            raw_patch = raw_patch.unsqueeze(0).unsqueeze(0)
            remove_dims = 2

        # interpolate raw patch
        raw_patch = interpolate(raw_patch, self.patch_shape, mode="trilinear")
        # remove additional dimensions
        for _ in range(remove_dims):
            raw_patch = raw_patch.squeeze(0)

        if label_patch.ndim == 4:
            label_patch = label_patch.unsqueeze(0)
            remove_dims = 1
        else:
            label_patch = label_patch.unsqueeze(0).unsqueeze(0)
            remove_dims = 2

        label_dtype = label_patch.dtype
        # check if label patch is of torch int type
        if label_dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            # convert to float for interpolation
            label_patch = label_patch.float()

        # interpolate label patch
        label_patch = interpolate(label_patch, self.patch_shape, mode="nearest")

        # remove additional dimensions
        for _ in range(remove_dims):
            label_patch = label_patch.squeeze(0)

        # convert back to int if necessary
        if label_dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            if label_dtype == torch.int64:
                label_patch = label_patch.long()
            else:
                label_patch = label_patch.int()

        return raw_patch, label_patch

    def _apply_offsets(self, idx: tuple, offsets: list, is_start: bool) -> tuple:
        if len(idx) == 4:
            spatial_idx = idx[1:]
        else:
            spatial_idx = idx

        new_idx = []
        for i, o, s in zip(spatial_idx, offsets, self.volume_shape, strict=True):
            if is_start:
                # prevent negative start
                start = max(0, i.start + o)
                stop = i.stop
            else:
                start = i.start
                # prevent stop exceeding the volume shape
                stop = min(s, i.stop + o)

            new_idx.append(slice(start, stop))

        if len(idx) == 4:
            return (idx[0],) + tuple(new_idx)

        return tuple(new_idx)


class ConfigDataset(Dataset):
    """
    Abstract class for datasets that are configured via a dictionary.
    """

    def __getitem__(self, index) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config: dict, phase: str) -> list[Dataset]:
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config: dataset configuration
            phase: one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch: list) -> Any:
        """Default collate_fn. Override in child class for non-standard datasets.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Collated batch.
        """
        return default_prediction_collate(batch)

def get_train_loaders(config: dict) -> dict[str, torch.utils.data.DataLoader]:
    raise NotImplementedError("HDF5 training pipeline was removed. Use a custom loader for your dataset.")


def get_test_loaders(config: dict) -> torch.utils.data.DataLoader:
    raise NotImplementedError("HDF5 test pipeline was removed. Use a custom loader for your dataset.")


def default_prediction_collate(batch: list) -> Any:
    """Default collate_fn to form a mini-batch of Tensor(s).

    Args:
        batch: List of samples from the dataset.

    Returns:
        Collated batch.
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch, strict=True)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(type(batch[0])))


def calculate_stats(img: np.array, skip: bool = False) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    if not skip:
        pmin, pmax, mean, std = np.percentile(img, 1), np.percentile(img, 99.6), np.mean(img), np.std(img)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {"pmin": pmin, "pmax": pmax, "mean": mean, "std": std}


def mirror_pad(image: np.ndarray, padding_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode="reflect")


def remove_padding(m: np.ndarray, padding_shape: int | tuple[int, ...] | None) -> np.ndarray:
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]
