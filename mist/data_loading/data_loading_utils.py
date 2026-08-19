"""Utility functions for data loading."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nvidia.dali import fn
from nvidia.dali import math
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.tensors import TensorGPU

from mist.data_loading.data_loading_constants import DataLoadingConstants as constants


def get_numpy_reader(
        files: list[str],
        shard_id: int,
        num_shards: int,
        seed: int,
        shuffle: bool,
) -> ops.readers.Numpy:
    """Creates and returns a DALI Numpy reader operator that reads numpy files.

    Args:
        files: List with file paths to numpy files (i.e., /path/to/file.npy)
        shard_id: The ID of the current shard, used for distributed data
            loading.
        num_shards: Total number of shards for splitting the data among workers.
        seed: Random seed for shuffling or any other randomness in the reader.
        shuffle: Whether to shuffle the data after each epoch.

    Returns:
        A DALI numpy reader operator configured with the provided parameters.
    """
    return ops.readers.Numpy(
        seed=seed,
        files=files,
        device="cpu",  # Reading happens on the CPU.
        read_ahead=True,  # Preload the data to speed up the reading process.
        shard_id=shard_id,  # Which shard of the data this instance will read.
        pad_last_batch=True,  # Pad last batch to for consistent batch sizes.
        num_shards=num_shards,  # Number of shards to split the dataset.
        dont_use_mmap=True,  # Disable memory mapping for reading files.
        shuffle_after_epoch=shuffle  # Shuffle the data after every epoch.
    )


def random_augmentation(
        probability: float,
        augmented_data: TensorGPU,
        original_data: TensorGPU,
) -> TensorGPU:
    """Apply random augmentation to the data based on a given probability.

    This function returns the augmented version of the original data with a
    user defined probability.

    Args:
        probability: The probability of applying the augmentation.
        augmented_data: The augmented version of the data.
        original_data: The original data.

    Returns:
        The augmented version of the data if the flip_coin function returns true
        with the user defined probability.
    """
    # Generate a condition using a coin flip based on the provided probability.
    condition = fn.cast(
        fn.random.coin_flip(probability=probability),
        dtype=types.DALIDataType.BOOL
    )

    # Invert the condition (negation) for the alternative case.
    neg_condition = condition ^ True

    # Return augmented data if condition is true.
    return condition * augmented_data + neg_condition * original_data


def noise_fn(img: TensorGPU) -> TensorGPU:
    """Apply random noise to the image data.

    This function applies random noise to the image data using a Gaussian
    noise function. The standard deviation of the noise is randomly selected
    from a range of 0.0 to 0.33.

    Args:
        img: The image data to apply noise to.

    Returns:
        The image data with random noise applied with a probability of 0.15.
    """
    # Generate random noise with a standard deviation between 0.0 and 0.33,
    # then clamp to the original image range to avoid out-of-range values.
    img_noised = math.clamp(
        img + fn.random.normal(img, stddev=fn.random.uniform(
            range=(
                constants.NOISE_FN_RANGE_MIN,
                constants.NOISE_FN_RANGE_MAX,
            )
        )),
        fn.reductions.min(img),
        fn.reductions.max(img),
    )

    # Return the augmented image data with a probability of 0.15.
    return random_augmentation(
        constants.NOISE_FN_PROBABILITY, img_noised, img
    )


def blur_fn(img: TensorGPU) -> TensorGPU:
    """Apply random Gaussian blur to the image data.

    This function applies random Gaussian blur to the image data. The sigma
    value for the Gaussian blur is randomly selected from a range of 0.5 to
    1.5.

    Args:
        img: The image data to apply Gaussian blur to.

    Returns:
        The image data with random Gaussian blur applied with a probability
        of 0.15.
    """
    # Apply random Gaussian blur with a sigma between 0.5 and 1.5,
    # then clamp to the original image range to avoid out-of-range values.
    img_blurred = math.clamp(
        fn.gaussian_blur(
            img, sigma=fn.random.uniform(
                range=(
                    constants.BLUR_FN_RANGE_MIN,
                    constants.BLUR_FN_RANGE_MAX,
                )
            )
        ),
        fn.reductions.min(img),
        fn.reductions.max(img),
    )

    # Return the augmented image data with a probability of 0.15.
    return random_augmentation(
        constants.BLUR_FN_PROBABILITY, img_blurred, img
    )


def brightness_fn(img: TensorGPU) -> TensorGPU:
    """Apply random brightness scaling to the image data.

    This function applies random brightness scaling to the image data. The
    brightness scale is randomly selected from a range of 0.7 to 1.3.

    Args:
        img: The image data to apply brightness scaling to.

    Returns:
        The image data with random brightness scaling applied with a
        probability of 0.15.
    """
    # Generate a random brightness scale between 0.7 and 1.3 with a
    # probability of 0.15. Otherwise, the brightness scale is 1.0.
    brightness_scale = random_augmentation(
        constants.BRIGHTNESS_FN_PROBABILITY,
        fn.random.uniform(
            range=(
                constants.BRIGHTNESS_FN_RANGE_MIN,
                constants.BRIGHTNESS_FN_RANGE_MAX,
            )
        ),
        1.0,
    )

    # Return the image data with the random brightness scale applied.
    return img * brightness_scale


def contrast_fn(img: TensorGPU) -> TensorGPU:
    """Apply random contrast scaling to the image data.

    This function applies random contrast scaling to the image data. The
    scaling factor is randomly selected from a range of 0.65 to 1.5. The
    minimum and maximum values of the image data are used to clamp the
    contrast scaling. This function is applied with a probability of 0.15.

    Args:
        img: The image data to apply contrast scaling to.

    Returns:
        The image data with random contrast scaling applied with a
        probability of 0.15.
    """
    # Get the minimum and maximum values of the image data.
    min_, max_ = fn.reductions.min(img), fn.reductions.max(img)

    # Generate a random contrast scaling factor between 0.65 and 1.5 with
    # a probability of 0.15. Otherwise, the scaling factor is 1.0.
    scale = random_augmentation(
        constants.CONTRAST_FN_PROBABILITY,
        fn.random.uniform(
            range=(
                constants.CONTRAST_FN_RANGE_MIN,
                constants.CONTRAST_FN_RANGE_MAX,
            )
        ),
        1.0,
    )

    # Scale the image data and clamp the values between the minimum and
    # maximum values of the original image data.
    img = math.clamp(img * scale, min_, max_)
    return img

def cutout_fn(img: TensorGPU) -> TensorGPU:
    """Apply random masking/cutouts of the image
    """
    erase_d = fn.random.uniform(range=(5,10))
    erase_h = fn.random.uniform(range=(5.,10.))
    erase_w = fn.random.uniform(range=(5.,10.))
    anchor_x = fn.random.uniform(range=(0.2,0.8))
    anchor_y = fn.random.uniform(range=(0.2,0.8))
    anchor_z = fn.random.uniform(range=(0.2,0.8))
    shape_node = fn.stack(erase_d,erase_h,erase_w)
    anchor_node = fn.stack(anchor_z,anchor_y, anchor_x)
    img_cutout = fn.erase(
        img,
        anchor=anchor_node,
        shape=shape_node,
        axis_names="DHW",
        fill_value=0.0,
        normalized_anchor=True
    )
    # Return the augmented image data with a probability of 0.4.
    return random_augmentation(
        constants.CUTOUT_FN_PROBABILITY, img_cutout, img
    )


def channel_dropout_fn(img: TensorGPU, n_channels: int) -> TensorGPU:
    """Randomly replace entire channels of a multi-channel image with zeros.

    Each channel is independently zeroed out with probability
    CHANNEL_DROPOUT_FN_PROBABILITY. If every channel would be dropped for a
    given sample, the first channel is kept instead so the network never sees
    an all-zero input.

    Args:
        img: The image data (DHWC layout) to apply channel dropout to.
        n_channels: The number of channels in the image, i.e., the size of
            the last (C) axis.

    Returns:
        The image data with entire channels randomly replaced by zeros.
    """
    if n_channels < 2:
        return img

    keep_flags = [
        fn.cast(
            fn.random.coin_flip(
                probability=1.0 - constants.CHANNEL_DROPOUT_FN_PROBABILITY
            ),
            dtype=types.DALIDataType.FLOAT,
        )
        for _ in range(n_channels)
    ]

    # Guarantee at least one channel survives: if every channel was dropped
    # for this sample, force the first channel back on.
    any_kept = keep_flags[0]
    for flag in keep_flags[1:]:
        any_kept = any_kept + flag
    all_dropped = fn.cast(any_kept == 0.0, dtype=types.DALIDataType.FLOAT)
    keep_flags[0] = keep_flags[0] + all_dropped

    channels = [
        fn.slice(img, c, 1, axes=[3]) * flag
        for c, flag in enumerate(keep_flags)
    ]
    return fn.cat(*channels, axis=3)


def resolve_fixed_rotation_axis(
        target_spacing: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Resolve the fixed rotation axis vector from a dataset's target spacing.

    This is a plain Python (not DALI-graph) computation, since target_spacing
    is static, known once at pipeline-construction time. If target_spacing is
    anisotropic (max spacing / min spacing exceeds
    ROTATION_AXIS_ANISOTROPY_THRESHOLD), the rotation axis is fixed to the
    coarsest (lowest-resolution) spatial dimension, so the rotation plane
    stays within the two higher-resolution dimensions. Otherwise (isotropic
    or no spacing information), it falls back to
    ROTATION_DEFAULT_ARRAY_AXIS.

    Args:
        target_spacing: The dataset's target voxel spacing, ordered to match
            the DHWC array's spatial axes (index 0 = D, 1 = H, 2 = W), or
            None if unavailable.

    Returns:
        The rotation axis vector in DALI's (x, y, z) axis convention, ready
        to pass to fn.transforms.rotation(axis=...).
    """
    array_axis = constants.ROTATION_DEFAULT_ARRAY_AXIS
    if target_spacing is not None:
        spacing = np.asarray(target_spacing, dtype=np.float64)
        anisotropy_ratio = spacing.max() / spacing.min()
        if anisotropy_ratio > constants.ROTATION_AXIS_ANISOTROPY_THRESHOLD:
            array_axis = int(np.argmax(spacing))
    return constants.ROTATION_AXIS_BY_ARRAY_AXIS[array_axis]


def validate_train_and_eval_inputs(
        imgs: list[str],
        lbls: list[str],
        dtms: list[str] | None = None,
) -> None:
    """Validate that the input data is correct.

    Ensures that images, labels, and optional DTM data are provided and that
    the lengths of the image, label, and DTM lists match.

    Args:
        imgs: List of image file paths.
        lbls: List of label file paths.
        dtms: Optional list of DTM data file paths. Defaults to None.

    Raises:
        ValueError: If the number of images, labels, or DTMs are incorrect.
    """
    if not imgs:
        raise ValueError("No images found!")

    if not lbls:
        raise ValueError("No labels found!")

    if len(imgs) != len(lbls):
        raise ValueError("Number of images and labels do not match!")

    if dtms is not None:
        if not dtms:
            raise ValueError("No DTM data found!")
        if len(imgs) != len(dtms):
            raise ValueError("Number of images and DTMs do not match!")


def is_valid_generic_pipeline_input(input_data: Any) -> bool:
    """Check if the input data is a valid generic pipeline input.

    Args:
        input_data: The input data to check.

    Returns:
        True if the input data is a valid generic pipeline input, False otherwise.
    """
    if not isinstance(input_data, Sequence) or isinstance(input_data, str):
        return False  # Must be a sequence but not a single string.

    if len(input_data) == 0:
        return False  # Empty lists are not valid.

    return all(
        isinstance(item, str) and
        item.endswith(".npy") and
        Path(item).is_file() for item in input_data
    )
