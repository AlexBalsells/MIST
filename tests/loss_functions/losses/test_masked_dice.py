"""Tests for the class-presence-aware Dice and Dice + CE losses."""

import pytest
import torch

from mist.loss_functions.loss_registry import get_loss
from mist.loss_functions.losses.dice import DiceLoss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.losses.masked_dice import (
    DiceCEMaskedLoss,
    MaskedDiceLoss,
    merge_missing_into_background,
)

N_CLASSES = 4


def _batch(seed=0, batch=2, size=6):
    gen = torch.Generator().manual_seed(seed)
    y_true = torch.randint(0, 3, (batch, 1, size, size, size), generator=gen)
    y_pred = torch.randn(batch, N_CLASSES, size, size, size, generator=gen)
    return y_true, y_pred


def test_registered():
    assert get_loss("dice_ce_masked") is DiceCEMaskedLoss
    assert get_loss("dice_masked") is MaskedDiceLoss


def test_all_present_matches_unmasked():
    y_true, y_pred = _batch()
    present = torch.ones(2, N_CLASSES)
    masked = DiceCEMaskedLoss()(y_true, y_pred, present=present)
    plain = DiceCELoss()(y_true, y_pred)
    assert torch.allclose(masked, plain, atol=1e-5)

    masked_dice = MaskedDiceLoss(exclude_background=True)(y_true, y_pred)
    plain_dice = DiceLoss(exclude_background=True)(y_true, y_pred)
    # Class 3 is absent from y_true, so derived presence differs. Compare with
    # an explicit all-present tensor instead.
    explicit = MaskedDiceLoss(exclude_background=True)(
        y_true, y_pred, present=present
    )
    assert torch.allclose(explicit, plain_dice, atol=1e-5)
    assert masked_dice.item() != pytest.approx(plain_dice.item())


def test_merge_moves_missing_mass_into_background():
    _, y_pred = _batch()
    probs = torch.softmax(y_pred, 1)
    present = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()
    merged = merge_missing_into_background(probs, present)
    assert torch.all(merged[0, 3] == 0)
    assert torch.allclose(merged[0, 0], probs[0, 0] + probs[0, 3])
    assert torch.allclose(merged[1], probs[1])
    assert torch.allclose(merged.sum(1), torch.ones_like(merged.sum(1)), atol=1e-5)


def test_predicting_missing_class_over_background_is_free():
    y_true = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    present = torch.tensor([[1, 0]], dtype=torch.float32)
    low = torch.zeros(1, 2, 4, 4, 4)
    low[:, 0] = 5.0
    high = torch.zeros(1, 2, 4, 4, 4)
    high[:, 1] = 5.0  # Confident fibula-like prediction on background voxels.
    loss = DiceCEMaskedLoss()
    # Mass on the missing class is merged into background, so both are cheap.
    l_low = loss(y_true, low, present=present)
    l_high = loss(y_true, high, present=present)
    assert l_low.item() < 0.1 and l_high.item() < 0.1
    # Without masking, the same prediction is heavily penalized.
    assert DiceCELoss()(y_true, high).item() > 1.0


def test_wrong_other_class_still_penalized():
    y_true = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    present = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    pred = torch.zeros(1, 3, 4, 4, 4)
    pred[:, 1] = 5.0  # Predicts class 1 where the label is background.
    assert DiceCEMaskedLoss()(y_true, pred, present=present).item() > 1.0


def test_requires_present():
    y_true, y_pred = _batch()
    with pytest.raises(ValueError, match="present"):
        DiceCEMaskedLoss()(y_true, y_pred)


def test_bad_present_shape():
    y_true, y_pred = _batch()
    with pytest.raises(ValueError, match="num_classes"):
        DiceCEMaskedLoss()(y_true, y_pred, present=torch.ones(2, 3))


def test_derived_presence_for_validation():
    y_true = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    y_true[..., :2] = 1
    y_pred = torch.randn(1, 3, 4, 4, 4)
    derived = MaskedDiceLoss(exclude_background=True)(y_true, y_pred)
    explicit = MaskedDiceLoss(exclude_background=True)(
        y_true, y_pred, present=torch.tensor([[1, 1, 0]])
    )
    assert torch.allclose(derived, explicit)


def test_backward_with_all_missing_foreground():
    y_true = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    y_pred = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
    loss = DiceCEMaskedLoss(exclude_background=True)(
        y_true, y_pred, present=torch.tensor([[1, 0, 0]])
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(y_pred.grad).all()
