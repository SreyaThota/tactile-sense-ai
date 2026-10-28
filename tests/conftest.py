"""SYNTHETIC test fixtures.

These images imitate the structure of a GelSight frame (smooth illumination gradient plus a
local "imprint") only so that the software can be tested without the real dataset. They are
NOT tactile data and no result obtained on them says anything about tactile sensing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SIZE = 64


def synthetic_background(size: int = SIZE) -> np.ndarray:
    """Smooth colour gradient standing in for the static LED illumination of a gel."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([60 + 60 * y, 90 + 30 * x, 80 + 50 * (1 - y)], axis=-1)
    return img.clip(0, 255).astype(np.uint8)


def synthetic_contact(cls: int, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Background plus a class-dependent imprint: class 0 = horizontal bar, class 1 = vertical bar."""
    img = synthetic_background(size).astype(np.int16)
    off = int(rng.integers(-6, 7))
    c = size // 2 + off
    if cls == 0:
        cv2.rectangle(img, (12, c - 3), (52, c + 3), (60, -40, 40), -1)
    else:
        cv2.rectangle(img, (c - 3, 12), (c + 3, 52), (60, -40, 40), -1)
    img = img + rng.normal(0, 1.5, img.shape)
    return img.clip(0, 255).astype(np.uint8)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def background() -> np.ndarray:
    return synthetic_background()


@pytest.fixture
def pre_cfg() -> dict:
    return {
        "image_size": SIZE,
        "blur_kernel": 5,
        "diff_threshold": 15,
        "morph_kernel": 3,
        "min_contact_fraction": 0.02,
        "drop_no_contact": True,
    }


@pytest.fixture
def folder_dataset(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Folder-per-class dataset with 6 recordings (groups) x 4 frames per class."""
    root = tmp_path / "dataset"
    for cls, name in enumerate(["horizontal", "vertical"]):
        (root / name).mkdir(parents=True)
        for rec in range(6):
            for frame in range(4):
                cv2.imwrite(str(root / name / f"rec{rec}__f{frame}.png"), synthetic_contact(cls, rng))
    return root
