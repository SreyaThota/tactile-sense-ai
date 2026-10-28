"""Assemble the shareable results/ package from the saved experiment outputs.

Every number and figure is taken from models/ and reports/ (written by src.train / src.evaluate)
or recomputed with the saved checkpoints; nothing is entered by hand.

Usage (after scripts/run_experiments.py):  python scripts/build_results.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import visualization as viz  # noqa: E402
from src.config import ROOT, load_config, resolve, setup_logging  # noqa: E402
from src.inference import TactilePredictor  # noqa: E402
from src.preprocessing import contact_area_batch, draw_contact_overlay, to_rgb  # noqa: E402

OUT = ROOT / "results"


def model_comparison_plot(metrics: dict, path: Path) -> None:
    items = [("accuracy", "Accuracy\n(single frame)"), ("macro_f1", "Macro-F1\n(single frame)")]
    fused = ["8", "32"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    groups = [label for _, label in items] + [f"Object accuracy\n({k} touches fused)" for k in fused]
    x = np.arange(len(groups))
    width = 0.38
    for i, (model, m) in enumerate(metrics["models"].items()):
        values = [m[key] for key, _ in items] + [m["object_level_accuracy_by_touches"][k] for k in fused]
        bars = ax.bar(x + (i - 0.5) * width, values, width - 0.03, color=viz.COLORS[model], label=viz.LABELS[model])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color=viz.INK)
    ax.axhline(0.1, color=viz.MUTED, lw=1, ls="--")
    ax.annotate("chance (10 classes)", (x[0] - 0.45, 0.1), xytext=(0, -11), textcoords="offset points",
                ha="left", fontsize=8, color=viz.MUTED)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.grid(axis="x", visible=False)
    ax.set_title("Held-out test set: CNN vs. HOG + SVM baseline")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def prediction_figure(image, true_label, predictor, title, path: Path) -> dict:
    """Input frame, contact overlay and class probabilities of both models."""
    region = predictor.contact(image)
    preds = {m: predictor.predict(image, m) for m in predictor.available_models}
    fig, axes = plt.subplots(1, 2 + len(preds), figsize=(3.1 * (2 + len(preds)), 3.0),
                             gridspec_kw={"width_ratios": [1, 1] + [1.6] * len(preds)})
    axes[0].imshow(to_rgb(image), interpolation="nearest")
    axes[0].set_title(f"Input (true digit {true_label})", fontsize=9)
    axes[1].imshow(to_rgb(draw_contact_overlay(image, region)), interpolation="nearest")
    axes[1].set_title(f"Contact: {region.area_fraction:.0%} of image", fontsize=9)
    for ax in axes[:2]:
        ax.axis("off")
    for ax, (model, p) in zip(axes[2:], preds.items()):
        names = list(p.probabilities)
        ax.bar(names, list(p.probabilities.values()), color=viz.COLORS[model], width=0.7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("digit")
        verdict = "correct" if p.label == true_label else "wrong"
        ax.set_title(f"{viz.LABELS[model]}: {p.label} ({p.confidence:.0%}, {verdict})", fontsize=9)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {m: {"predicted": p.label, "confidence": round(p.confidence, 4)} for m, p in preds.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="build from the smoke-test outputs (software check only)")
    parser.add_argument("--out", default=str(OUT), help="output directory")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config()
    if args.smoke:
        from scripts.run_experiments import smoke_config

        cfg = smoke_config(cfg)
    out = Path(args.out)
    reports, models_dir = resolve(cfg["paths"]["reports_dir"]), resolve(cfg["paths"]["models_dir"])
    metrics = json.loads((reports / "metrics.json").read_text())
    info = json.loads((models_dir / "training_info.json").read_text())
    predictor = TactilePredictor(models_dir)
    names = predictor.class_names

    for sub in ("metrics", "plots", "images"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- metrics
    shutil.copy(reports / "metrics.json", out / "metrics" / "evaluation_metrics.json")
    shutil.copy(reports / "classification_report.txt", out / "metrics" / "classification_report.txt")
    for name in ("baseline_analysis.json", "baseline_analysis.md"):
        if (reports / name).is_file():
            shutil.copy(reports / name, out / "metrics" / name)
    with open(out / "metrics" / "model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "accuracy_all_frames", "object_acc_1_touch", "object_acc_8_touches", "object_acc_32_touches",
                    "train_samples", "train_time_s", "latency_median_ms", "latency_p95_ms"])
        for model, m in metrics["models"].items():
            t = m["training"]
            fusion = m["object_level_accuracy_by_touches"]
            w.writerow([viz.LABELS[model], *(round(m[k], 4) for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")),
                        round(m["accuracy_all_frames_incl_no_contact"], 4),
                        *(round(fusion[k], 4) for k in ("1", "8", "32")),
                        t.get("train_samples"), t.get("train_time_s", t.get("fit_time_s")),
                        round(m["latency_single_image"]["median_ms"], 2), round(m["latency_single_image"]["p95_ms"], 2)])

    # ---- plots
    for src, dst in (("cnn_training_curves.png", "training_curves.png"), ("confusion_cnn.png", "confusion_matrix_cnn.png"),
                     ("confusion_classical.png", "confusion_matrix_classical.png"), ("per_class_f1.png", "per_class_f1.png"),
                     ("multi_touch_fusion.png", "multi_touch_fusion.png"), ("contact_vs_gel_height.png", "contact_vs_gel_height.png")):
        shutil.copy(reports / src, out / "plots" / dst)
    model_comparison_plot(metrics, out / "plots" / "model_comparison.png")

    # ---- images (real test-split frames)
    with np.load(resolve(cfg["data"]["cache_path"]), allow_pickle=False) as z:
        test = z["official_split"] == "test"
        images, labels, gel_z, test_groups = z["images"][test], z["labels"][test], z["gel_z"][test], z["groups"][test]
    area = contact_area_batch(images, predictor.reference, predictor.pre_cfg)
    contact = np.flatnonzero(area >= predictor.pre_cfg["min_contact_fraction"])
    images, labels, area, gel_z = images[contact], labels[contact], area[contact], gel_z[contact]

    probs = predictor.predict_proba(images, "cnn")
    pred, conf = probs.argmax(1), probs.max(1)
    # Representative example: a correct prediction with a clear contact (median area among correct ones).
    correct = np.flatnonzero(pred == labels)
    ok = correct[np.argsort(area[correct])[len(correct) // 2]]
    # Failure: the most confident wrong CNN prediction (the most instructive kind of error).
    wrong = np.flatnonzero(pred != labels)
    fail = wrong[np.argmax(conf[wrong])]

    example = images[ok]
    shutil.copy(reports / "preprocessing_pipeline.png", out / "images" / "opencv_preprocessing.png")
    for name, img in (("original_tactile_image.png", to_rgb(example)),):
        plt.imsave(out / "images" / name, np.kron(img, np.ones((4, 4, 1), np.uint8)))  # 4x nearest upscale
    region = predictor.contact(example)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for ax, (title, img, cmap) in zip(axes, (("Input frame", to_rgb(example), None),
                                             ("Binary contact mask", region.mask, "gray"),
                                             ("Contours, bounding box, centroid", to_rgb(draw_contact_overlay(example, region)), None))):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "images" / "contact_mask.png")
    plt.close(fig)

    ok_pred = prediction_figure(example, names[labels[ok]], predictor, "Correct prediction (held-out test frame)",
                                out / "images" / "prediction_example.png")
    fail_pred = prediction_figure(images[fail], names[labels[fail]], predictor,
                                  "Failure case: most confident wrong CNN prediction (held-out test frame)",
                                  out / "images" / "prediction_failure_example.png")
    for model in ("cnn", "classical"):
        for kind in ("correct", "failures"):
            src = reports / f"examples_{model}_{kind}.png"
            if src.is_file():
                shutil.copy(src, out / "images" / f"examples_{model}_{kind}.png")

    # Confidence vs. accuracy: is a high predicted probability trustworthy?
    calibration = {}
    for model in predictor.available_models:
        pr = probs if model == "cnn" else predictor.predict_proba(images, model)
        pm, cm = pr.argmax(1), pr.max(1)
        calibration[model] = [
            {"confidence_bin": f"[{lo:.1f}, {hi:.1f})", "frames": int(m.sum()),
             "accuracy": round(float((pm[m] == labels[m]).mean()), 4) if m.any() else None}
            for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01))
            for m in [(cm >= lo) & (cm < hi)]
        ]

    # Object-level bootstrap: frames of one object are correlated, so resample whole test objects
    # to get honest 95% intervals for each model and for the CNN - SVM difference.
    groups = test_groups[contact]
    preds = {m: (probs if m == "cnn" else predictor.predict_proba(images, m)).argmax(1) for m in predictor.available_models}
    frames_of = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(cfg["seed"])
    boot = {m: {"accuracy": [], "macro_f1": []} for m in preds}
    for _ in range(2000):
        # sample objects with replacement; a repeated object contributes its frames repeatedly
        pick = np.concatenate([frames_of[i] for i in rng.integers(0, len(frames_of), len(frames_of))])
        for m, p in preds.items():
            boot[m]["accuracy"].append(float((p[pick] == labels[pick]).mean()))
            boot[m]["macro_f1"].append(float(f1_score(labels[pick], p[pick], average="macro")))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    bootstrap = {m: {k: ci(v) for k, v in d.items()} for m, d in boot.items()}
    if {"cnn", "classical"} <= preds.keys():
        bootstrap["cnn_minus_classical"] = {
            k: ci(np.array(boot["cnn"][k]) - np.array(boot["classical"][k])) for k in ("accuracy", "macro_f1")
        }

    summary = {
        "dataset": "Real Tactile MNIST (TimSchneider42/tactile-mnist-touch-real-single-t256-64x64), CC-BY-2.0",
        "task": "10-class digit classification of the touched 3D-printed object from one GelSight Mini frame",
        "data": info["data"]["splits"],
        "test_set": metrics["test_set"],
        "training": {m: info.get(m) for m in ("classical", "cnn")},
        "results": {m: {k: metrics["models"][m][k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                                                            "accuracy_all_frames_incl_no_contact",
                                                            "object_level_accuracy_by_touches", "latency_single_image")}
                    for m in metrics["models"]},
        "contact_detector_vs_gel_height_pearson_r": metrics.get("contact_detector_vs_gel_height_pearson_r"),
        "examples": {
            "prediction_example": {"true": names[labels[ok]], **ok_pred},
            "prediction_failure_example": {
                "true": names[labels[fail]],
                "contact_area": round(float(area[fail]), 4),
                "gel_height_mm": round(float(gel_z[fail] * 1000), 2),
                "gel_height_mm_percentiles_10_50_90_of_test_contact_frames":
                    [round(float(v), 2) for v in np.percentile(gel_z * 1000, [10, 50, 90])],
                **fail_pred,
            },
        },
        "confidence_vs_accuracy_test_contact_frames": calibration,
        "bootstrap_95ci_over_test_objects": bootstrap,
    }
    (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("examples", "bootstrap_95ci_over_test_objects")}, indent=2))


if __name__ == "__main__":
    main()
