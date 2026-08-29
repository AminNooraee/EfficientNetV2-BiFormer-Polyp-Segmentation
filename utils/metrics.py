"""Segmentation metrics matching the experiment notebooks."""

from __future__ import annotations

import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import DiceScore


class BatchMetricSuite:
    """Compute the same batch-wise macro metrics used in the notebooks."""

    def __init__(self, device: torch.device, num_classes: int = 2) -> None:
        self.dice = DiceScore(
            num_classes=num_classes,
            average="macro",
            input_format="index",
        ).to(device)
        self.iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.precision = MulticlassPrecision(
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro",
        ).to(device)
        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        ).to(device)

    def training_dice(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        return float(self.dice(predictions, targets).item())

    def all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "dice": float(self.dice(predictions, targets).item()),
            "iou": float(self.iou(predictions, targets).item()),
            "precision": float(self.precision(predictions, targets).item()),
            "recall": float(self.recall(predictions, targets).item()),
            "accuracy": float(self.accuracy(predictions, targets).item()),
        }


def average_metric_dicts(
    totals: dict[str, float],
    n_batches: int,
) -> dict[str, float]:
    return {key: value / n_batches for key, value in totals.items()}
