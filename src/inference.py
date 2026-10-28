"""Loading trained models and running predictions on new tactile images."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from .feature_extraction import FeatureExtractor
from .models import TactileCNN, to_cnn_input
from .preprocessing import ContactRegion, detect_contact, resize, validate_image
from .train import CLASSICAL_FILE, CNN_FILE, STATE_FILE

log = logging.getLogger(__name__)

MODEL_NAMES = ("cnn", "classical")


class ModelNotAvailableError(FileNotFoundError):
    """Raised when a requested model or the preprocessing state has not been trained yet."""


@dataclass
class Prediction:
    model: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    contact: ContactRegion


class TactilePredictor:
    """Loads the saved preprocessing state and whichever trained models are present."""

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        state_path = self.models_dir / STATE_FILE
        if not state_path.is_file():
            raise ModelNotAvailableError(
                f"No preprocessing state at {state_path}. Train the models first: python -m src.train"
            )
        with np.load(state_path, allow_pickle=False) as s:
            self.reference = s["reference"]
            self.diff_std = s["diff_std"]
            self.class_names = [str(c) for c in s["class_names"]]
            self.pre_cfg = json.loads(str(s["preprocessing"]))
            feat_cfg = json.loads(str(s["features"]))
        self.image_size = self.pre_cfg["image_size"]
        self.extractor = FeatureExtractor(self.reference, self.pre_cfg, feat_cfg)

        self.classical = None
        if (self.models_dir / CLASSICAL_FILE).is_file():
            self.classical = joblib.load(self.models_dir / CLASSICAL_FILE)["model"]

        self.cnn = None
        if (self.models_dir / CNN_FILE).is_file():
            ckpt = torch.load(self.models_dir / CNN_FILE, map_location="cpu", weights_only=True)
            self.cnn = TactileCNN(ckpt["num_classes"], ckpt["dropout"])
            self.cnn.load_state_dict(ckpt["state_dict"])
            self.cnn.eval()

    @property
    def available_models(self) -> list[str]:
        return [m for m in MODEL_NAMES if getattr(self, m) is not None]

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Validate and resize a raw BGR image to the training resolution."""
        return resize(validate_image(image), self.image_size)

    def contact(self, image: np.ndarray) -> ContactRegion:
        p = self.pre_cfg
        return detect_contact(self.prepare(image), self.reference, p["diff_threshold"], p["blur_kernel"],
                              p["morph_kernel"], p["min_contact_fraction"])

    def predict_proba(self, images: np.ndarray, model: str = "cnn", batch_size: int = 512) -> np.ndarray:
        """Class probabilities for a batch of already prepared (resized) BGR images."""
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{model}', choose from {MODEL_NAMES}")
        if getattr(self, model) is None:
            raise ModelNotAvailableError(f"The {model} model has not been trained (no file in {self.models_dir})")
        if model == "classical":
            return self.classical.predict_proba(self.extractor.extract_batch(images))
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                x = to_cnn_input(images[start:start + batch_size], self.reference, self.diff_std)
                out.append(torch.softmax(self.cnn(x), dim=1).numpy())
        return np.concatenate(out)

    def predict(self, image: np.ndarray, model: str = "cnn") -> Prediction:
        """Predict the class of a single raw BGR image."""
        prepared = self.prepare(image)
        probs = self.predict_proba(prepared[None], model)[0]
        best = int(np.argmax(probs))
        return Prediction(
            model=model,
            label=self.class_names[best],
            confidence=float(probs[best]),
            probabilities={c: float(p) for c, p in zip(self.class_names, probs)},
            contact=self.contact(prepared),
        )
