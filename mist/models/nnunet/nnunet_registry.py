"""Create variants of nnUNet model."""
# MIST imports.
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.model_registry import register_model


def _create_nnunet_base(
    use_pocket_model: bool, use_residual_blocks: bool = True, **kwargs
) -> NNUNet:
    """Shared factory logic for nnUNet and nnunet-pocket.

    Args:
        use_pocket_model: Whether to use constant-width filters (pocket mode).
        use_residual_blocks: Whether to use residual conv blocks.
        **kwargs: Model configuration, must include in_channels, out_channels,
            patch_size, and target_spacing.

    Returns:
        An instance of the NNUNet model.
    """
    required_keys = ["in_channels", "out_channels", "patch_size", "target_spacing"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    return NNUNet(
        in_channels=kwargs["in_channels"],
        out_channels=kwargs["out_channels"],
        patch_size=kwargs["patch_size"],
        target_spacing=kwargs["target_spacing"],
        use_residual_blocks=use_residual_blocks,
        use_deep_supervision=True,
        use_pocket_model=use_pocket_model,
    )


@register_model("nnunet")
def create_nnunet(**kwargs) -> NNUNet:
    """Create a standard nnUNet model with residual blocks and deep supervision."""
    return _create_nnunet_base(use_pocket_model=False, **kwargs)


@register_model("nnunet-pocket")
def create_nnunet_pocket(**kwargs) -> NNUNet:
    """Create a pocket nnUNet with constant 32-filter width across all depths."""
    return _create_nnunet_base(use_pocket_model=True, **kwargs)


@register_model("nnunet-plain")
def create_nnunet_plain(**kwargs) -> NNUNet:
    """Create a standard nnUNet with plain (non-residual) conv blocks.

    This matches the original nnU-Net v2 PlainConvUNet topology, so weights
    trained in nnU-Net can be converted and loaded into it.
    """
    return _create_nnunet_base(
        use_pocket_model=False, use_residual_blocks=False, **kwargs
    )
