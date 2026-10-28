"""Training of the classical baseline and the CNN.

Usage (from the repository root):
    python -m src.train                  # train both models
    python -m src.train --model cnn      # only the CNN
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from .config import load_config, resolve, set_seed, setup_logging
from .data_loader import TactileDataset, load_dataset, make_splits
from .feature_extraction import FeatureExtractor
from .models import TactileCNN, build_classical_model, count_parameters, to_cnn_input
from .preprocessing import contact_area_batch, estimate_reference

log = logging.getLogger(__name__)

STATE_FILE = "preprocess_state.npz"
CLASSICAL_FILE = "classical_svm.joblib"
CNN_FILE = "cnn.pt"
INFO_FILE = "training_info.json"


@dataclass
class ExperimentData:
    train: TactileDataset
    val: TactileDataset
    test: TactileDataset
    test_all: TactileDataset          # test split including no-contact frames (secondary evaluation)
    test_all_contact: np.ndarray      # contact flag for every frame in test_all
    reference: np.ndarray
    diff_std: np.ndarray
    summary: dict[str, Any]


def prepare_experiment(cfg: dict, reference: np.ndarray | None = None) -> ExperimentData:
    """Load data, split by physical object, estimate the reference frame on the training split
    only and remove frames without detectable contact (if configured)."""
    seed, pre = cfg["seed"], cfg["preprocessing"]
    ds = load_dataset(cfg)
    splits = make_splits(ds, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"], seed)

    if reference is None:
        rng = np.random.default_rng(seed)
        train_idx = splits["train"]
        sample = rng.choice(train_idx, size=min(5000, len(train_idx)), replace=False)
        reference = estimate_reference(ds.images[sample])

    parts, summary = {}, {"data_report": ds.report, "splits": {}}
    for name, idx in splits.items():
        area = contact_area_batch(ds.images[idx], reference, pre)
        contact = area >= pre["min_contact_fraction"]
        keep = idx[contact] if pre["drop_no_contact"] else idx
        parts[name] = ds.subset(keep)
        if name == "test":
            parts["test_all"], test_contact = ds.subset(idx), contact
        summary["splits"][name] = {
            "frames_total": int(len(idx)),
            "frames_with_contact": int(contact.sum()),
            "frames_used": int(len(keep)),
            "objects": int(len(np.unique(ds.groups[idx]))),
        }
        log.info("%-5s: %d objects, %d frames, %d with contact (%.0f%%)", name,
                 summary["splits"][name]["objects"], len(idx), contact.sum(), 100 * contact.mean())

    # Per-channel std of the reference difference on training frames, used to standardise CNN inputs.
    diff = parts["train"].images[:5000].astype(np.float32) - reference
    diff_std = diff.reshape(-1, 3).std(axis=0).astype(np.float32) + 1e-3
    return ExperimentData(parts["train"], parts["val"], parts["test"], parts["test_all"],
                          test_contact, reference, diff_std, summary)


def save_state(models_dir: Path, data: ExperimentData, cfg: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        models_dir / STATE_FILE,
        reference=data.reference,
        diff_std=data.diff_std,
        class_names=np.asarray(data.train.class_names),
        preprocessing=json.dumps(cfg["preprocessing"]),
        features=json.dumps(cfg["features"]),
    )


# --------------------------------------------------------------------------- classical


def train_classical(data: ExperimentData, cfg: dict, models_dir: Path) -> dict[str, Any]:
    seed, ccfg = cfg["seed"], cfg["classical"]
    rng = np.random.default_rng(seed)
    train = data.train
    if len(train) > ccfg["max_train_samples"]:
        # Per-class subsampling keeps the class balance of the full training set.
        idx = np.concatenate([
            rng.choice(np.flatnonzero(train.labels == c),
                       size=min((train.labels == c).sum(), ccfg["max_train_samples"] // len(train.class_names)),
                       replace=False)
            for c in np.unique(train.labels)
        ])
        train = train.subset(np.sort(idx))
    log.info("Classical: extracting features for %d training frames", len(train))

    extractor = FeatureExtractor(data.reference, cfg["preprocessing"], cfg["features"])
    t0 = time.perf_counter()
    x_train = extractor.extract_batch(train.images)
    x_val = extractor.extract_batch(data.val.images)
    feat_time = time.perf_counter() - t0

    model = build_classical_model(ccfg, seed)
    model.set_params(pca__n_components=min(ccfg["pca_components"], *x_train.shape))
    t0 = time.perf_counter()
    model.fit(x_train, train.labels)
    fit_time = time.perf_counter() - t0
    val_acc = float((model.predict(x_val) == data.val.labels).mean())
    log.info("Classical: feature dim %d, fit %.1fs, validation accuracy %.3f", x_train.shape[1], fit_time, val_acc)

    joblib.dump({"model": model, "class_names": train.class_names}, models_dir / CLASSICAL_FILE)
    return {
        "train_samples": int(len(train)),
        "feature_dim": int(x_train.shape[1]),
        "feature_extraction_s": round(feat_time, 2),
        "fit_time_s": round(fit_time, 2),
        "val_accuracy": round(val_acc, 4),
    }


# --------------------------------------------------------------------------- CNN


def _augment(x: torch.Tensor, max_shift: int, intensity: float, gen: torch.Generator) -> torch.Tensor:
    """Physically plausible augmentation for tactile difference images.

    * Random translation (zero padding = "no deformation"): the touch position on the object is
      random anyway, so shifting the imprint does not change its meaning.
    * Random contact-signal scaling: models variation in pressure and between gels.
    No flips or rotations: the LED illumination direction is fixed relative to the sensor, so
    rotating/flipping would create shading patterns the sensor cannot produce, and mirrored
    digit strokes would change the class semantics.
    """
    n, _, h, w = x.shape
    if max_shift > 0:
        padded = nn.functional.pad(x, (max_shift,) * 4)
        dx = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
        dy = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
        x = torch.stack([padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])
    if intensity > 0:
        scale = 1 + (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1) * intensity
        x = x * scale
    return x


def _run_epoch(model, x, y, batch_size, loss_fn, optimizer=None, aug=None, gen=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(y), generator=gen) if training else torch.arange(len(y))
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            if training and aug:
                xb = _augment(xb, aug["max_shift"], aug["intensity_scale"], gen)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(y), correct / len(y)


def train_cnn(data: ExperimentData, cfg: dict, models_dir: Path) -> dict[str, Any]:
    ccfg, seed = cfg["cnn"], cfg["seed"]
    set_seed(seed)
    torch.set_num_threads(ccfg["num_threads"])
    gen = torch.Generator().manual_seed(seed)

    x_train = to_cnn_input(data.train.images, data.reference, data.diff_std)
    x_val = to_cnn_input(data.val.images, data.reference, data.diff_std)
    y_train, y_val = torch.from_numpy(data.train.labels), torch.from_numpy(data.val.labels)

    num_classes = len(data.train.class_names)
    model = TactileCNN(num_classes, ccfg["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=ccfg["learning_rate"], weight_decay=ccfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()
    log.info("CNN: %d parameters, %d train / %d val frames", count_parameters(model), len(y_train), len(y_val))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, ccfg["max_epochs"] + 1):
        tl, ta = _run_epoch(model, x_train, y_train, ccfg["batch_size"], loss_fn, optimizer, ccfg["augment"], gen)
        vl, va = _run_epoch(model, x_val, y_val, ccfg["batch_size"], loss_fn)
        scheduler.step(vl)
        for k, v in zip(history, (tl, ta, vl, va, optimizer.param_groups[0]["lr"])):
            history[k].append(v)
        log.info("epoch %2d  train loss %.3f acc %.3f | val loss %.3f acc %.3f", epoch, tl, ta, vl, va)
        if vl < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad_epochs = vl, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= ccfg["early_stopping_patience"]:
                log.info("Early stopping after epoch %d (best epoch %d)", epoch, best_epoch)
                break
    train_time = time.perf_counter() - t0

    torch.save(
        {
            "state_dict": best_state,
            "num_classes": num_classes,
            "dropout": ccfg["dropout"],
            "class_names": data.train.class_names,
            "best_epoch": best_epoch,
            "history": history,
        },
        models_dir / CNN_FILE,
    )
    return {
        "parameters": count_parameters(model),
        "epochs_run": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "val_accuracy_best_epoch": round(history["val_acc"][best_epoch - 1], 4),
        "train_samples": int(len(y_train)),
    }


# --------------------------------------------------------------------------- entry point


def run_training(cfg: dict, which: str = "both") -> dict[str, Any]:
    set_seed(cfg["seed"])
    models_dir = resolve(cfg["paths"]["models_dir"])
    data = prepare_experiment(cfg)
    save_state(models_dir, data, cfg)

    info_path = models_dir / INFO_FILE
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    info["data"] = data.summary
    if which in ("both", "classical"):
        info["classical"] = train_classical(data, cfg, models_dir)
    if which in ("both", "cnn"):
        info["cnn"] = train_cnn(data, cfg, models_dir)
    info_path.write_text(json.dumps(info, indent=2))
    log.info("Saved models and training info to %s", models_dir)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--model", choices=["both", "classical", "cnn"], default="both")
    args = parser.parse_args()
    setup_logging()
    run_training(load_config(args.config), args.model)


if __name__ == "__main__":
    main()
