"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed non-DP randomness sources used by experiments."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
