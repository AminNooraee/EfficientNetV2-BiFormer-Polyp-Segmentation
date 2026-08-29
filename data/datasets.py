"""Dataset and five-fold split utilities."""

from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, Subset


class KvasirSegDataset(Dataset):
    """Kvasir-SEG loader matching the experiment notebooks."""

    def __init__(self, image_dir: str, mask_dir: str) -> None:
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.jpg")))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self.image_paths)} vs {len(self.mask_paths)}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = (mask > 127).astype(np.uint8)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


class CVCClinicDBDataset(Dataset):
    """CVC-ClinicDB loader matching the experiment notebooks."""

    def __init__(self, image_dir: str, mask_dir: str) -> None:
        self.image_paths = sorted(
            glob(os.path.join(image_dir, "*.png")),
            key=lambda path: int(os.path.basename(path).split(".")[0]),
        )
        self.mask_paths = sorted(
            glob(os.path.join(mask_dir, "*.png")),
            key=lambda path: int(os.path.basename(path).split(".")[0]),
        )

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self.image_paths)} vs {len(self.mask_paths)}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = (mask > 127).astype(np.uint8)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )


def build_dataset(
    dataset_name: str,
    image_dir: str,
    mask_dir: str,
) -> Dataset:
    name = dataset_name.lower()

    if name in {"kvasir", "kvasir-seg", "kvasir_seg"}:
        return KvasirSegDataset(image_dir, mask_dir)

    if name in {"cvc", "cvc-clinicdb", "cvc_clinicdb", "clinicdb"}:
        return CVCClinicDBDataset(image_dir, mask_dir)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_fold_subsets(
    dataset: Dataset,
    fold: int,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Create train/validation/test subsets exactly as in the notebooks."""

    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}]")

    all_indices = np.arange(len(dataset))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = [test_idx for _, test_idx in kfold.split(all_indices)]

    test_idx = fold_indices[fold]
    remaining_idx = np.setdiff1d(all_indices, test_idx)

    train_idx, val_idx = train_test_split(
        remaining_idx,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
