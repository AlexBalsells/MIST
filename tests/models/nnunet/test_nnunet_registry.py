"""Unit tests for the nnUNet model factory function."""
import pytest

# MIST imports.
from mist.models.nnunet.nnunet_registry import (
    create_nnunet,
    create_nnunet_plain,
    create_nnunet_pocket,
)
from mist.models.nnunet.mist_nnunet import NNUNet


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for the NNUNet model."""
    return {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": (128, 128, 128),
        "target_spacing": (1.0, 1.0, 1.0),
    }


def test_create_nnunet_success(valid_config):
    """Test that create_nnunet returns an NNUNet instance for valid config."""
    model = create_nnunet(**valid_config)
    assert isinstance(model, NNUNet)


def test_create_nnunet_pocket_success(valid_config):
    """Test that create_nnunet_pocket returns an NNUNet instance."""
    model = create_nnunet_pocket(**valid_config)
    assert isinstance(model, NNUNet)


def test_create_nnunet_plain_is_non_residual(valid_config):
    """nnunet-plain uses plain conv blocks (no residual conv3/norm3 layers)."""
    plain = create_nnunet_plain(**valid_config)
    residual = create_nnunet(**valid_config)
    assert isinstance(plain, NNUNet)
    assert plain.unet.use_deep_supervision
    assert not any(".conv3." in k or ".norm3." in k for k in plain.state_dict())
    assert any(".conv3." in k for k in residual.state_dict())


def test_nnunet_plain_registered():
    """The nnunet-plain name resolves through the model registry."""
    import mist.models  # noqa: F401  pylint: disable=unused-import
    from mist.models.model_registry import list_registered_models
    assert "nnunet-plain" in list_registered_models()


@pytest.mark.parametrize("missing_key", [
    "in_channels",
    "out_channels",
    "patch_size",
    "target_spacing",
])
def test_create_nnunet_missing_keys(valid_config, missing_key):
    """Test create_nnunet raises ValueError if a required key is missing."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(
        ValueError, match=f"Missing required key '{missing_key}'"
    ):
        create_nnunet(**config)
