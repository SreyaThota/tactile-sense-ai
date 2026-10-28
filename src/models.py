"""Model definitions: classical scikit-learn pipeline and a compact PyTorch CNN."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


def build_classical_model(cfg: dict, seed: int) -> Pipeline:
    """Standardise -> PCA -> RBF-SVM.

    PCA compresses the ~1.8k-dimensional HOG vector, which makes the kernel SVM much faster and
    less prone to overfitting. Sigmoid (Platt) calibration turns SVM decision values into class
    probabilities for the demo's confidence display.
    """
    svm = SVC(C=cfg["svm_c"], gamma=cfg["svm_gamma"], random_state=seed)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=cfg["pca_components"], random_state=seed)),
            ("svm", CalibratedClassifierCV(svm, method="sigmoid", cv=3, ensemble=False)),
        ]
    )


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TactileCNN(nn.Module):
    """Three VGG-style conv blocks + global average pooling (~290k parameters).

    Input: (N, 3, H, W) normalised difference images. Global average pooling keeps the model
    independent of the exact input size and adds some tolerance to the contact position.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(_conv_block(in_channels, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def to_cnn_input(images: np.ndarray, reference: np.ndarray, diff_std: np.ndarray) -> torch.Tensor:
    """uint8 BGR (N, H, W, 3) -> float tensor (N, 3, H, W) of standardised reference differences."""
    x = (images.astype(np.float32) - reference) / diff_std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
