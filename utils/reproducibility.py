"""Reproducibility helpers for the public release."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed common RNGs.

    The original notebooks explicitly used seed=42 for fold construction.
    The public implementation additionally seeds training RNGs to make reruns
    more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
