"""Diagnose the classical baseline on the VALIDATION split (the test set is not touched).

Questions answered:
* Is HOG + SVM under- or over-fitting? (training vs. validation accuracy)
* Which feature group carries the signal? (ablation: HOG only / contact statistics only / all)
* Is the hand-crafted representation the bottleneck? (control: SVM on raw down-sampled
  difference pixels with the same Scaler -> PCA -> SVM pipeline)

Usage:  python scripts/analyze_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve, set_seed, setup_logging  # noqa: E402
from src.feature_extraction import FeatureExtractor  # noqa: E402
from src.train import prepare_experiment  # noqa: E402

log = logging.getLogger("analyze_baseline")
TRAIN_SAMPLES = 6000  # per variant; smaller than the main run so the ablation stays fast


def raw_pixels(images: np.ndarray, reference: np.ndarray, size: int = 16) -> np.ndarray:
    diff = (images.astype(np.float32) - reference).mean(axis=3)
    return np.stack([cv2.resize(d, (size, size), interpolation=cv2.INTER_AREA).ravel() for d in diff])


def fit_eval(name, x_tr, y_tr, x_va, y_va, cfg, seed):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=min(cfg["pca_components"], x_tr.shape[1]), random_state=seed)),
        ("svm", SVC(C=cfg["svm_c"], gamma=cfg["svm_gamma"], random_state=seed)),
    ])
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    row = {
        "variant": name,
        "feature_dim": int(x_tr.shape[1]),
        "train_accuracy": round(float((model.predict(x_tr) == y_tr).mean()), 4),
        "val_accuracy": round(float((model.predict(x_va) == y_va).mean()), 4),
        "fit_s": round(time.perf_counter() - t0, 1),
    }
    log.info("%s", row)
    return row


def main() -> None:
    setup_logging()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    data = prepare_experiment(cfg)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data.train), size=min(TRAIN_SAMPLES, len(data.train)), replace=False)
    train = data.train.subset(np.sort(idx))

    extractor = FeatureExtractor(data.reference, cfg["preprocessing"], cfg["features"])
    f_tr, f_va = extractor.extract_batch(train.images), extractor.extract_batch(data.val.images)
    hog_dim = len(extractor.hog.compute(extractor.signed_gray_difference(train.images[0])).ravel())
    y_tr, y_va = train.labels, data.val.labels

    ccfg = cfg["classical"]
    rows = [
        fit_eval("All hand-crafted features (HOG + histograms + contact geometry)", f_tr, y_tr, f_va, y_va, ccfg, seed),
        fit_eval("HOG only", f_tr[:, :hog_dim], y_tr, f_va[:, :hog_dim], y_va, ccfg, seed),
        fit_eval("Contact statistics only (histogram, colour stats, geometry)", f_tr[:, hog_dim:], y_tr,
                 f_va[:, hog_dim:], y_va, ccfg, seed),
        fit_eval("Control: raw 16x16 grey difference pixels", raw_pixels(train.images, data.reference), y_tr,
                 raw_pixels(data.val.images, data.reference), y_va, ccfg, seed),
    ]

    # Leakage demonstration (NOT a valid result): split the same train+val pool randomly by
    # frame, so touches of one object can land on both sides. Same features and pipeline.
    pool_x = np.concatenate([f_tr, f_va])
    pool_y = np.concatenate([y_tr, y_va])
    perm = rng.permutation(len(pool_y))
    cut = len(f_tr)
    rows.append(fit_eval("Leaky control: same features, random frame-level split (invalid protocol)",
                         pool_x[perm[:cut]], pool_y[perm[:cut]], pool_x[perm[cut:]], pool_y[perm[cut:]], ccfg, seed))
    out = {
        "split_used": "validation (test set untouched)",
        "train_samples_per_variant": int(len(train)),
        "val_samples": int(len(data.val)),
        "chance_accuracy": round(1 / len(train.class_names), 3),
        "variants": rows,
    }
    reports = resolve(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "baseline_analysis.json").write_text(json.dumps(out, indent=2))
    table = ["| Variant | Dim | Train acc. | Val acc. |", "|---|---|---|---|"]
    table += [f"| {r['variant']} | {r['feature_dim']} | {r['train_accuracy']:.3f} | {r['val_accuracy']:.3f} |" for r in rows]
    (reports / "baseline_analysis.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
