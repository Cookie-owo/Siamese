"""Utilities for reproducible PyTorch experiments."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch random-number generators.

    Parameters
    ----------
    seed:
        Random seed used by all supported generators.
    deterministic:
        When True, configure cuDNN for deterministic execution.
        This may reduce training speed.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, received {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python random generators in a DataLoader worker.

    PyTorch assigns each worker an initial seed. The seed is converted
    to the range accepted by NumPy and reused for Python's random module.
    """
    del worker_id  # The worker-specific seed is obtained from PyTorch.

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for deterministic DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
