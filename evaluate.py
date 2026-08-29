"""Evaluate a trained checkpoint on one held-out cross-validation fold."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from data.datasets import build_dataset, build_fold_subsets
from models.model import EfficientUNetWithBiFormerDecoder
from utils.losses import ComboLoss
from utils.metrics import BatchMetricSuite, average_metric_dicts
from utils.training import load_model_state_compat


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    dataset = build_dataset(
        dataset_name=config["dataset"],
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    _, _, test_ds = build_fold_subsets(
        dataset=dataset,
        fold=args.fold,
        n_splits=config["n_splits"],
        validation_ratio=config["validation_ratio"],
        seed=config["split_seed"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = EfficientUNetWithBiFormerDecoder(
        out_channels=2,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_model_state_compat(
        model,
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    criterion = ComboLoss(alpha=config["loss_alpha"]).to(device)
    metrics = BatchMetricSuite(device)
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            totals["loss"] += float(criterion(outputs, masks).item())
            predictions = torch.argmax(outputs, dim=1)
            batch_metrics = metrics.all_metrics(predictions, masks)
            for key, value in batch_metrics.items():
                totals[key] += value

    results = average_metric_dicts(totals, len(test_loader))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
