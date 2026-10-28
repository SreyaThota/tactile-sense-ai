"""Configuration loading, logging setup and seeding shared by all entry points."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Repository root, so that relative paths in config.yaml work from any working directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file (defaults to ``config.yaml`` in the repo root)."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
