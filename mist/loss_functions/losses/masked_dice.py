"""Dice and Dice + Cross Entropy losses that ignore classes without labels.

Some training cases do not have ground truth for every class (e.g., a dataset
that never annotated the fibula). In those cases the missing class is marked as
"not present" for the case, and the loss does not punish the model for
predicting it. For a case with a missing class c:
    - The predicted probability of class c is merged into the background
      probability. Predicting c where the label says background is therefore
      not penalized, but predicting any other class in the wrong place still is.
    - Class c is dropped from the per-class Dice average for that case.
"""

from typing import Any

import torch
from torch.nn import functional as F

from mist.loss_functions import loss_utils
from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice import DiceLoss

# Floor on the probability inside the log of the cross entropy.
_PROB_EPS = 1e-7


def _validate_present(present: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Check that the class-presence tensor is (batch, classes)."""
    if present.ndim != 2 or present.shape != y_pred.shape[:2]:
        raise ValueError(
            "present must have shape (batch_size, num_classes) = "
            f"{tuple(y_pred.shape[:2])}. Got {tuple(present.shape)}."
        )


def merge_missing_into_background(
    probs: torch.Tensor, present: torch.Tensor
) -> torch.Tensor:
    """Add the probability of each missing class to the background.

    Args:
        probs: Softmax probabilities of shape (B, C, H, W, D).
        present: Boolean tensor of shape (B, C). False marks a class with no
            ground truth for that sample. The background (channel 0) is never
            treated as missing.

    Returns:
        Probabilities of shape (B, C, H, W, D) with the missing channels zeroed
        and their mass added to channel 0.
    """
    missing = (~present).to(probs.dtype)[:, :, None, None, None]
    missing[:, 0] = 0.0
    extra_bg = torch.sum(probs * missing, dim=1, keepdim=True)
    probs = probs * (1.0 - missing)
    return torch.cat([probs[:, :1] + extra_bg, probs[:, 1:]], dim=1)


@register_loss(name="dice_masked")
class MaskedDiceLoss(DiceLoss):
    """Soft Dice loss that skips classes with no ground truth in a sample.

    If `present` is not given, a class is considered present in a sample when
    it has at least one voxel in y_true. That is only appropriate when y_true
    is the full scan (e.g., validation), not a patch.
    """

    def _resolve_present(
        self,
        y_onehot: torch.Tensor,
        present: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the (B, C) boolean class-presence tensor."""
        if present is None:
            present = torch.sum(y_onehot, dim=self.spatial_dims_3d) > 0
        present = present.to(device=y_onehot.device, dtype=torch.bool).clone()
        present[:, 0] = True
        return present

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        present: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the masked Dice loss.

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Raw model output tensor of shape (B, C, H, W, D).
            present: Optional boolean tensor of shape (B, C).

        Returns:
            Dice loss as a scalar tensor.
        """
        loss_utils.check_loss_inputs(y_true, y_pred)
        y_onehot = loss_utils.get_one_hot(y_true, y_pred.shape[1])
        present = self._resolve_present(y_onehot, present)
        _validate_present(present, y_pred)

        probs = merge_missing_into_background(
            F.softmax(y_pred.float(), dim=1), present
        )

        if self.exclude_background:
            y_onehot = y_onehot[:, 1:]
            probs = probs[:, 1:]
            present = present[:, 1:]

        numerator = torch.sum(
            torch.square(y_onehot - probs), dim=self.spatial_dims_3d
        )
        denominator = (
            torch.sum(torch.square(y_onehot), dim=self.spatial_dims_3d) +
            torch.sum(torch.square(probs), dim=self.spatial_dims_3d) +
            self.avoid_division_by_zero
        )
        loss = numerator / denominator  # Per sample, per class.

        # Mean over the classes that have ground truth, then over the batch.
        weights = present.to(loss.dtype)
        loss = torch.sum(loss * weights, dim=1) / torch.clamp(
            torch.sum(weights, dim=1), min=1.0
        )
        return torch.mean(loss)


@register_loss(name="dice_ce_masked")
class DiceCEMaskedLoss(MaskedDiceLoss):
    """Dice + cross entropy loss that ignores classes with no ground truth.

    The total loss is 0.5 * (CE + Dice), as in `dice_ce`. Both terms use the
    class-presence tensor `present` (B, C), which must be passed to forward.
    It has to describe the whole scan, not the patch, because a class that is
    simply outside the current patch should still be penalized.
    """

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        present: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the masked Dice + Cross Entropy loss.

        Args:
            y_true: Ground truth mask of shape (B, 1, H, W, D).
            y_pred: Network output logits of shape (B, C, H, W, D).
            present: Boolean tensor of shape (B, C) marking the classes that
                have ground truth in each sample's scan.

        Returns:
            Scalar loss value 0.5 * Dice + 0.5 * CE.

        Raises:
            ValueError: If `present` is not provided.
        """
        if present is None:
            raise ValueError(
                "dice_ce_masked requires per-scan class presence ('present')."
            )
        loss_dice = super().forward(y_true, y_pred, present=present, **kwargs)

        present = present.to(device=y_pred.device, dtype=torch.bool).clone()
        present[:, 0] = True
        probs = merge_missing_into_background(
            F.softmax(y_pred.float(), dim=1), present
        )
        target = y_true.long()  # (B, 1, H, W, D).
        p_true = torch.gather(probs, dim=1, index=target)
        loss_ce = -torch.log(torch.clamp(p_true, min=_PROB_EPS)).mean()

        return 0.5 * (loss_ce + loss_dice)
