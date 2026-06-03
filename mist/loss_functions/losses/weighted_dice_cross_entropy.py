"""Dice + Cross Entropy loss for segmentation tasks with weights inversely proportional to volume of class."""

from typing import Any

import torch
from torch import nn

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice import SegmentationLoss


@register_loss(name="wt_dice_ce")
class WTDiceCELoss(SegmentationLoss):
    """Dice loss combined with cross entropy loss but weighted inversely to volume of region.

    The total loss is the average of:
        - Soft Dice loss (weighted)
        - CrossEntropyLoss (weighted)

    Attributes:
        cross_entropy: PyTorch's cross entropy loss module.
    """

    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize the WTDiceCELoss.

        Args:
            exclude_background: If True, background class (channel 0) is
                excluded from the Dice computation ONLY. The Cross Entropy
                calculation will always include the background to ensure
                the model learns to suppress false positives.
            kwargs: Additional keyword arguments for future extensions.
        """
        # Initialize WTDiceLoss (parent handles exclude_background for Dice part).
        super().__init__(exclude_background=exclude_background, **kwargs)

        # Initialize CE.
        # We do NOT use ignore_index here. We always want CE to penalize
        # background misclassifications, even if Dice ignores them.
        #self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the (weighted) Dice + Cross Entropy loss.

        Args:
            y_true: Ground truth mask shaped (B, 1, H, W, D).
            y_pred: Network output logits (no softmax) shaped (B, C, H, W, D).
            **kwargs: Additional arguments.

        Returns:
            Scalar loss value 0.5 * Dice + 0.5 * CE.
        """
        # 1. Compute target for Cross Entropy Loss before one hot encoding.
        target = y_true.long().squeeze(1)
        y_pred_ce = y_pred

        # 1.5 Compute CE Loss.
        loss_ce = nn.functional.cross_entropy(y_pred_ce, target, reduction='none') # Shape (B,H,W,D)
        loss_ce = loss_ce.mean(dim=(1,2,3)) # Shape (B,)
        loss_ce = loss_ce.mean() #Scalar

        # 2. Compute weights
        #   weights: (B, C)
        #   total_volume: scalar
        y_true, y_pred = self.preprocess(y_true, y_pred)
        class_volume = torch.sum(y_true, dim=self.spatial_dims_3d)
        #class_volume += 1 # Avoid division by zero error if class not present in volume
        #total_volume = torch.sum(class_volume,dim=[0,1])
        #weights = total_volume/class_volume
        weights = 1.0 / (torch.square(class_volume) + self.avoid_division_by_zero)
        # Normalize the weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + self.avoid_division_by_zero)
        
        #self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
        # 3. Compute Dice Loss.
        numerator = torch.sum(
            torch.square(y_true - y_pred), dim=self.spatial_dims_3d
        )
        denominator = (
            torch.sum(torch.square(y_true), dim=self.spatial_dims_3d) +
            torch.sum(torch.square(y_pred), dim=self.spatial_dims_3d) +
            self.avoid_division_by_zero
        )

        loss = numerator / denominator # Per class. (B, C)
        loss = weights * loss # Weights applied to each class. (B, C)
        #loss = torch.mean(loss, dim=1) # Mean over classes. (B,)
        loss = torch.sum(loss, dim=1) # Sum over classes since sum of weights is one. (B, )
        loss_dice = torch.mean(loss) # Mean over batch.

        # 4. Compute CE Loss.
        #loss_ce = self.cross_entropy(y_pred_ce, target)

        return 0.5 * (loss_ce + loss_dice)
