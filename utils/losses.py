"""Loss functions used in the manuscript experiments."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Foreground Dice loss for binary segmentation."""

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        foreground = torch.softmax(inputs, dim=1)[:, 1]
        targets = targets.float()

        intersection = (foreground * targets).sum()
        dice = (
            2.0 * intersection + smooth
        ) / (foreground.sum() + targets.sum() + smooth)

        return 1.0 - dice


class ComboLoss(nn.Module):
    """Weighted Cross-Entropy + foreground Dice loss."""

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.alpha * self.cross_entropy(inputs, targets)
            + (1.0 - self.alpha) * self.dice(inputs, targets)
        )
