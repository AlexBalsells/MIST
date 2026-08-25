"""Dataclass for constants used in data loading."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadingConstants:
    """Dataclass for constants used in data loading."""
    # Noise function constants.
    NOISE_FN_RANGE_MIN = 0.0
    NOISE_FN_RANGE_MAX = 0.33
    NOISE_FN_PROBABILITY = 0.15

    # Blur function constants.
    BLUR_FN_RANGE_MIN = 0.5
    BLUR_FN_RANGE_MAX = 1.5
    BLUR_FN_PROBABILITY = 0.15

    # Brightness function constants.
    BRIGHTNESS_FN_RANGE_MIN = 0.7
    BRIGHTNESS_FN_RANGE_MAX = 1.3
    BRIGHTNESS_FN_PROBABILITY = 0.15

    # Contrast function constants.
    CONTRAST_FN_RANGE_MIN = 0.65
    CONTRAST_FN_RANGE_MAX = 1.5
    CONTRAST_FN_PROBABILITY = 0.15

    # Zoom function constants.
    ZOOM_FN_RANGE_MIN = 0.7
    ZOOM_FN_RANGE_MAX = 1.0
    ZOOM_FN_PROBABILITY = 0.15

    # Flip function constants.
    HORIZONTAL_FLIP_PROBABILITY = 0.5
    VERTICAL_FLIP_PROBABILITY = 0.5
    DEPTH_FLIP_PROBABILITY = 0.5

    # Rotation function constants
    ROTATION_FN_RANGE_MIN = -10.
    ROTATION_FN_RANGE_MAX = 10.
    ROTATION_FN_PROBABILITY = 0.15

    # Rotation axis vectors, one per DHWC array spatial axis (index 0 = D,
    # 1 = H, 2 = W), expressed in DALI's (x, y, z) axis convention for
    # `fn.transforms.rotation`. For DHWC-layout volumetric data, DALI's
    # (x, y, z) maps to (W, H, D) (i.e., reversed relative to the DHWC
    # storage/layout order). Rotating about a given array axis leaves that
    # axis invariant and confines the rotation to the plane of the other two.
    ROTATION_AXIS_BY_ARRAY_AXIS = (
        (0.0, 0.0, 1.0),  # array axis 0 (D) -> DALI z.
        (0.0, 1.0, 0.0),  # array axis 1 (H) -> DALI y.
        (1.0, 0.0, 0.0),  # array axis 2 (W) -> DALI x.
    )

    # Fallback array axis (see ROTATION_AXIS_BY_ARRAY_AXIS) used to fix the
    # rotation axis when spacing is unavailable or the volume is isotropic
    # (i.e., no single axis is clearly lower-resolution). 2 = last spatial
    # dimension (W), so rotation happens within the D-H plane by default.
    ROTATION_DEFAULT_ARRAY_AXIS = 2

    # If (max spacing / min spacing) for a volume's target spacing exceeds
    # this threshold, the volume is considered anisotropic and the rotation
    # axis is fixed to its coarsest (lowest-resolution) spatial dimension
    # instead of ROTATION_DEFAULT_ARRAY_AXIS. Mirrors the anisotropy bar used
    # elsewhere in MIST for patch-size selection
    # (analyzer_constants.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD).
    ROTATION_AXIS_ANISOTROPY_THRESHOLD = 3.0

    # Cutout function probability
    CUTOUT_FN_PROBABILITY = 0.5

    # Channel dropout function constants. This is the probability that any
    # given channel is independently zeroed out (not an overall probability
    # of applying the augmentation at all).
    CHANNEL_DROPOUT_FN_PROBABILITY = 0.33

    # Slice dropout function constants. SLICE_DROPOUT_FN_PROBABILITY is the
    # probability that any given slice along the randomly chosen spatial
    # axis is independently zeroed out (not an overall probability of
    # applying the augmentation at all).
    SLICE_DROPOUT_FN_PROBABILITY = 0.15
