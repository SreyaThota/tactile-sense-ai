"""Matplotlib figures for reports and the demo. All functions return the Figure (and save it if a
path is given) so they can be reused in Streamlit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works without a display (CI, servers, Streamlit)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .preprocessing import ContactRegion, contact_magnitude, draw_contact_overlay, to_gray, to_hsv, to_rgb

# Fixed series colours (categorical slots 1 and 2): colour follows the model, never its rank.
COLORS = {"cnn": "#2a78d6", "classical": "#eb6834", "train": "#2a78d6", "val": "#eb6834"}
LABELS = {"cnn": "CNN", "classical": "HOG + SVM"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
})


def _save(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_preprocessing_steps(image: np.ndarray, reference: np.ndarray, region: ContactRegion,
                             blur_kernel: int = 5, path: str | Path | None = None) -> Figure:
    """Original | reference | grey | HSV value | |difference| | contact overlay."""
    magnitude = contact_magnitude(image, reference, blur_kernel)
    panels = [
        ("Raw tactile frame", to_rgb(image), None),
        ("No-contact reference", to_rgb(reference.clip(0, 255).astype(np.uint8)), None),
        ("Grey-scale", to_gray(image), "gray"),
        ("HSV hue", to_hsv(image)[..., 0], "twilight"),
        ("|frame - reference| (blurred)", magnitude, "magma"),
        ("Contact region", to_rgb(draw_contact_overlay(image, region)), None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.7))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    return _save(fig, path)


def plot_history(history: dict[str, list[float]], best_epoch: int, path: str | Path | None = None) -> Figure:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")):
        for split in ("train", "val"):
            ax.plot(epochs, history[f"{split}_{metric}"], color=COLORS[split], lw=2,
                    label="training" if split == "train" else "validation")
        ax.axvline(best_epoch, color=MUTED, lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[0].legend()
    axes[0].annotate("best checkpoint", (best_epoch, axes[0].get_ylim()[1]), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, path: str | Path | None = None) -> Figure:
    """Row-normalised confusion matrix (recall per true class) with counts annotated."""
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names)
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_f1(f1: dict[str, list[float]], class_names: list[str], path: str | Path | None = None) -> Figure:
    x = np.arange(len(class_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, model in enumerate(f1):
        ax.bar(x + (k - 0.5) * width, f1[model], width - 0.03, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(x, class_names)
    ax.set_xlabel("digit class")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-class F1 on the held-out test set (contact frames)")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def plot_fusion(results: dict[str, dict[int, float]], path: str | Path | None = None) -> Figure:
    """Object-level accuracy when averaging class probabilities over k touches of one object."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for model, curve in results.items():
        ks = sorted(curve)
        ax.plot(ks, [curve[k] for k in ks], marker="o", ms=6, lw=2, color=COLORS[model], label=LABELS[model])
        ax.annotate(f"{curve[ks[-1]]:.2f}", (ks[-1], curve[ks[-1]]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("touches per object (k)")
    ax.set_ylabel("object-level accuracy")
    ax.set_title("Multi-touch fusion on test objects")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_contact_vs_height(area: np.ndarray, gel_z: np.ndarray, path: str | Path | None = None) -> Figure:
    """Sanity check of the contact detector against an independent physical measurement."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(gel_z * 1000, area * 100, s=4, alpha=0.25, color=COLORS["cnn"], linewidths=0)
    r = np.corrcoef(gel_z, area)[0, 1]
    ax.set_xlabel("gel height at touch [mm] (robot kinematics)")
    ax.set_ylabel("detected contact area [% of image]")
    ax.set_title(f"Image-based contact vs. gel height (Pearson r = {r:.2f})", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_examples(images: np.ndarray, true: np.ndarray, pred: np.ndarray, conf: np.ndarray,
                  class_names: list[str], title: str, path: str | Path | None = None, cols: int = 8) -> Figure:
    n = len(images)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.75 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t, p, c in zip(np.ravel(axes), images, true, pred, conf):
        ax.imshow(to_rgb(img), interpolation="nearest")
        ax.set_title(f"true {class_names[t]} / pred {class_names[p]}\n{c:.0%}", fontsize=7,
                     color=INK if t == p else "#b3261e")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_probabilities(probs: dict[str, float], color: str = COLORS["cnn"]) -> Figure:
    names = list(probs)
    values = [probs[n] for n in names]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(names, values, color=color, width=0.7)
    best = int(np.argmax(values))
    ax.annotate(f"{values[best]:.0%}", (best, values[best]), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("class")
    ax.set_ylabel("probability")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
