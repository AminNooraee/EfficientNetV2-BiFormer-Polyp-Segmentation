"""Training utilities."""

from __future__ import annotations

import json
import os

import pandas as pd
import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_for_saving(model: nn.Module) -> dict:
    """Save the underlying model without a DataParallel ``module.`` prefix."""
    return unwrap_model(model).state_dict()


def load_model_state_compat(
    model: nn.Module,
    state_dict: dict,
    strict: bool = True,
) -> None:
    """Load both clean and legacy DataParallel checkpoints."""
    clean_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        clean_state[key] = value

    model.load_state_dict(clean_state, strict=strict)


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        checkpoint_path: str = "checkpoints/best_dice.ckpt",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score: float | None = None
        self.best_epoch = -1
        self.counter = 0
        self.early_stop = False

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score > self.best_score + self.min_delta

    def __call__(
        self,
        score: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extra_state: dict | None = None,
    ) -> None:
        if self._improved(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            state = {
                "epoch": epoch,
                "score": score,
                "model_state": model_state_for_saving(model),
            }
            if optimizer is not None:
                state["optimizer_state"] = optimizer.state_dict()
            if extra_state is not None:
                state.update(extra_state)

            os.makedirs(
                os.path.dirname(self.checkpoint_path) or ".",
                exist_ok=True,
            )
            torch.save(state, self.checkpoint_path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


class MetricsLogger:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_dice: float,
        val_dice: float,
    ) -> None:
        self.history["epoch"].append(int(epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["train_dice"].append(float(train_dice))
        self.history["val_dice"].append(float(val_dice))

    def save(self) -> None:
        json_path = os.path.join(self.save_dir, "metrics.json")
        csv_path = os.path.join(self.save_dir, "metrics.csv")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

        pd.DataFrame(self.history).to_csv(csv_path, index=False)
