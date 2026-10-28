"""Final evaluation of all trained models on the held-out test set.

Loads the saved checkpoints (not the in-memory training objects) so that this also verifies
checkpoint loading. Writes metrics.json, comparison.md and figures to the reports directory.

Usage:  python -m src.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import load_config, resolve, set_seed, setup_logging
from .inference import TactilePredictor
from .preprocessing import contact_area_batch, detect_contact
from .train import CNN_FILE, INFO_FILE, prepare_experiment
from . import visualization as viz

log = logging.getLogger(__name__)

DEMO_FILE = "demo_examples.npz"
DEMO_PER_CLASS = 50


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    y_pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report_text": classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0),
    }


def measure_latency(predictor: TactilePredictor, model: str, images: np.ndarray, runs: int) -> dict[str, float]:
    """Single-image latency (resize + preprocessing/features + model) on CPU, in milliseconds."""
    torch.set_num_threads(1)  # a realistic single-request setting, independent of training threads
    for img in images[:5]:
        predictor.predict(img, model)  # warm-up
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        predictor.predict(images[i % len(images)], model)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def multi_touch_fusion(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       ks: list[int], trials: int, seed: int) -> dict[int, float]:
    """Object-level accuracy when the class probabilities of k random contact touches of the
    same object are averaged (objects with fewer than k contact touches use all they have)."""
    rng = np.random.default_rng(seed)
    objects = np.unique(groups)
    result = {}
    for k in ks:
        correct = 0
        for _ in range(trials):
            for obj in objects:
                idx = np.flatnonzero(groups == obj)
                pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
                correct += int(probs[pick].mean(0).argmax() == labels[idx[0]])
        result[k] = correct / (trials * len(objects))
    return result


def run_evaluation(cfg: dict) -> dict[str, Any]:
    set_seed(cfg["seed"])
    models_dir, reports = resolve(cfg["paths"]["models_dir"]), resolve(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    predictor = TactilePredictor(models_dir)
    if not predictor.available_models:
        raise FileNotFoundError(f"No trained models in {models_dir}. Run python -m src.train first.")

    data = prepare_experiment(cfg, reference=predictor.reference)
    test, names = data.test, predictor.class_names
    info_path = models_dir / INFO_FILE
    train_info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    ev = cfg["evaluation"]

    results: dict[str, Any] = {
        "test_set": {
            "description": "Official Real Tactile MNIST test split (objects never seen in training/validation)",
            "objects": int(len(np.unique(data.test_all.groups))),
            "frames_total": int(len(data.test_all)),
            "frames_with_contact": int(len(test)),
            "objects_with_contact_frames": int(len(np.unique(test.groups))),
            "contact_frames_per_class": {c: int((test.labels == i).sum()) for i, c in enumerate(names)},
        },
        "models": {},
    }
    probs_by_model, fusion = {}, {}
    for model in predictor.available_models:
        log.info("Evaluating %s on %d test frames", model, len(test))
        t0 = time.perf_counter()
        probs = predictor.predict_proba(test.images, model)
        batch_time = time.perf_counter() - t0
        probs_by_model[model] = probs
        m = classification_metrics(test.labels, probs, names)
        all_probs = predictor.predict_proba(data.test_all.images, model)
        m["accuracy_all_frames_incl_no_contact"] = float((all_probs.argmax(1) == data.test_all.labels).mean())
        m["batch_throughput_frames_per_s"] = float(len(test) / batch_time)
        m["latency_single_image"] = measure_latency(predictor, model, test.images, ev["latency_runs"])
        fusion[model] = multi_touch_fusion(probs, test.labels, test.groups, ev["fusion_touches"],
                                           ev["fusion_trials"], cfg["seed"])
        m["object_level_accuracy_by_touches"] = {str(k): v for k, v in fusion[model].items()}
        m["training"] = train_info.get(model, {})
        results["models"][model] = m
        log.info("%s: accuracy %.3f, macro-F1 %.3f\n%s", model, m["accuracy"], m["macro_f1"], m["report_text"])
        viz.plot_confusion(np.array(m["confusion_matrix"]), names,
                           f"{viz.LABELS[model]} - test confusion matrix", reports / f"confusion_{model}.png")

    # ---- figures
    viz.plot_per_class_f1({m: [results["models"][m]["per_class"][c]["f1"] for c in names] for m in probs_by_model},
                          names, reports / "per_class_f1.png")
    viz.plot_fusion(fusion, reports / "multi_touch_fusion.png")
    if (models_dir / CNN_FILE).is_file():
        ckpt = torch.load(models_dir / CNN_FILE, map_location="cpu", weights_only=True)
        viz.plot_history(ckpt["history"], ckpt["best_epoch"], reports / "cnn_training_curves.png")
    if data.test_all.gel_z is not None:
        area = contact_area_batch(data.test_all.images, predictor.reference, predictor.pre_cfg)
        viz.plot_contact_vs_height(area, data.test_all.gel_z, reports / "contact_vs_gel_height.png")
        results["contact_detector_vs_gel_height_pearson_r"] = float(np.corrcoef(area, data.test_all.gel_z)[0, 1])

    p = predictor.pre_cfg
    example = test.images[int(np.argmax(contact_area_batch(test.images[:200], predictor.reference, p)))]
    region = detect_contact(example, predictor.reference, p["diff_threshold"], p["blur_kernel"],
                            p["morph_kernel"], p["min_contact_fraction"])
    viz.plot_preprocessing_steps(example, predictor.reference, region, p["blur_kernel"], reports / "preprocessing_pipeline.png")

    rng = np.random.default_rng(cfg["seed"])
    # Small set of held-out contact frames for the demo app (avoids loading the full cache there).
    demo_idx = np.concatenate([rng.permutation(np.flatnonzero(test.labels == c))[:DEMO_PER_CLASS]
                               for c in range(len(names))])
    np.savez_compressed(models_dir / DEMO_FILE, images=test.images[demo_idx], labels=test.labels[demo_idx])

    for model, probs in probs_by_model.items():
        pred, conf = probs.argmax(1), probs.max(1)
        for kind, mask in (("correct", pred == test.labels), ("failures", pred != test.labels)):
            idx = np.flatnonzero(mask)
            if len(idx):
                idx = rng.choice(idx, size=min(16, len(idx)), replace=False)
                viz.plot_examples(test.images[idx], test.labels[idx], pred[idx], conf[idx], names,
                                  f"{viz.LABELS[model]}: {'correct predictions' if kind == 'correct' else 'failure cases'}",
                                  reports / f"examples_{model}_{kind}.png")

    with open(reports / "classification_report.txt", "w", encoding="utf-8") as f:
        for model, m in results["models"].items():
            f.write(f"=== {viz.LABELS[model]} - held-out test set, contact frames (n={len(test)}) ===\n")
            f.write(m.pop("report_text") + "\n")
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    (reports / "comparison.md").write_text(comparison_table(results))
    log.info("Wrote metrics and figures to %s\n%s", reports, comparison_table(results))
    return results


def comparison_table(results: dict[str, Any]) -> str:
    rows = [
        "| Metric (held-out test set) | " + " | ".join(viz.LABELS[m] for m in results["models"]) + " |",
        "|---|" + "---|" * len(results["models"]),
    ]

    def row(name, fn):
        rows.append(f"| {name} | " + " | ".join(fn(m) for m in results["models"].values()) + " |")

    row("Accuracy (contact frames)", lambda m: f"{m['accuracy']:.3f}")
    row("Macro precision", lambda m: f"{m['macro_precision']:.3f}")
    row("Macro recall", lambda m: f"{m['macro_recall']:.3f}")
    row("Macro F1", lambda m: f"{m['macro_f1']:.3f}")
    row("Accuracy (all frames incl. no contact)", lambda m: f"{m['accuracy_all_frames_incl_no_contact']:.3f}")
    for k in ("1", "8", "32"):
        row(f"Object accuracy, {k} touch(es) fused", lambda m, k=k: f"{m['object_level_accuracy_by_touches'].get(k, float('nan')):.3f}")
    row("Training samples", lambda m: str(m["training"].get("train_samples", "-")))
    row("Training time [s]", lambda m: str(m["training"].get("train_time_s", m["training"].get("fit_time_s", "-"))))
    row("Latency, 1 image, 1 CPU thread [ms, median]", lambda m: f"{m['latency_single_image']['median_ms']:.2f}")
    row("Batch throughput [frames/s]", lambda m: f"{m['batch_throughput_frames_per_s']:.0f}")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    run_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
