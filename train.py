"""Train the EfficientNetV2-S + BiFormer polyp segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.datasets import build_dataset, build_fold_subsets
from models.model import EfficientUNetWithBiFormerDecoder
from utils.losses import ComboLoss
from utils.metrics import BatchMetricSuite, average_metric_dicts
from utils.reproducibility import set_global_seed
from utils.training import EarlyStopping, MetricsLogger, load_model_state_compat


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_loaders(dataset, fold: int, config: dict, pin_memory: bool):
    train_ds, val_ds, test_ds = build_fold_subsets(
        dataset=dataset,
        fold=fold,
        n_splits=config["n_splits"],
        validation_ratio=config["validation_ratio"],
        seed=config["split_seed"],
    )
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": pin_memory,
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def evaluate_loader(model, loader, criterion, device):
    model.eval()
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
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            totals["loss"] += float(criterion(outputs, masks).item())
            predictions = torch.argmax(outputs, dim=1)
            batch_metrics = metrics.all_metrics(predictions, masks)
            for key, value in batch_metrics.items():
                totals[key] += value

    return average_metric_dicts(totals, len(loader))


def train_one_fold(dataset, fold, config, output_root, device):
    set_global_seed(config["training_seed"] + fold)

    run_dir = output_root / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        dataset,
        fold,
        config,
        pin_memory=device.type == "cuda",
    )

    model = EfficientUNetWithBiFormerDecoder(
        out_channels=2,
        pretrained=True,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = ComboLoss(alpha=config["loss_alpha"]).to(device)
    training_metrics = BatchMetricSuite(device)

    checkpoint_path = run_dir / "best_dice.ckpt"
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
        checkpoint_path=str(checkpoint_path),
    )
    logger = MetricsLogger(str(run_dir))

    for epoch in range(config["max_epochs"]):
        model.train()
        total_train_loss = 0.0
        total_train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            total_train_dice += training_metrics.training_dice(
                predictions,
                masks,
            )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_dice = total_train_dice / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        total_val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                total_val_loss += float(criterion(outputs, masks).item())
                predictions = torch.argmax(outputs, dim=1)
                total_val_dice += training_metrics.training_dice(
                    predictions,
                    masks,
                )

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_dice = total_val_dice / len(val_loader)

        logger.log_epoch(
            epoch,
            avg_train_loss,
            avg_val_loss,
            avg_train_dice,
            avg_val_dice,
        )
        logger.save()

        print(
            f"[Fold {fold} | Epoch {epoch + 1}/{config['max_epochs']}] "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        early_stopping(
            avg_val_dice,
            epoch,
            model,
            optimizer,
            extra_state={"fold": fold, "config": config},
        )
        if early_stopping.early_stop:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {early_stopping.best_epoch + 1}; "
                f"Best validation Dice: {early_stopping.best_score:.4f}"
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    load_model_state_compat(target_model, checkpoint["model_state"], strict=True)
    model.eval()

    test_metrics = evaluate_loader(
        model,
        test_loader,
        criterion,
        device,
    )
    summary = {
        "fold": fold,
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_val_dice": float(checkpoint["score"]),
        "test_metrics": test_metrics,
        "n_train": len(train_loader.dataset),
        "n_validation": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }

    with open(run_dir / "test_results.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Fold {fold} test metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
    )
    return summary


def aggregate_folds(summaries):
    metric_names = list(summaries[0]["test_metrics"].keys())
    mean_metrics = {
        metric: sum(item["test_metrics"][metric] for item in summaries)
        / len(summaries)
        for metric in metric_names
    }
    return {
        "n_folds": len(summaries),
        "mean_test_metrics": mean_metrics,
        "folds": summaries,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    dataset = build_dataset(
        config["dataset"],
        args.image_dir,
        args.mask_dir,
    )
    if len(dataset) == 0:
        raise RuntimeError("No images were found. Check dataset paths.")

    output_root = Path(args.output_dir) / config["run_name"]
    output_root.mkdir(parents=True, exist_ok=True)

    folds = range(config["n_splits"]) if args.all_folds else [args.fold]
    summaries = [
        train_one_fold(dataset, fold, config, output_root, device)
        for fold in folds
    ]

    if len(summaries) > 1:
        aggregate = aggregate_folds(summaries)
        with open(
            output_root / "cross_validation_summary.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(aggregate, file, indent=2)

        print(
            "Cross-validation mean metrics: "
            + ", ".join(
                f"{key}={value:.4f}"
                for key, value in aggregate["mean_test_metrics"].items()
            )
        )


if __name__ == "__main__":
    main()
