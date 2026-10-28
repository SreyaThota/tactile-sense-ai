"""Run the full pipeline: train both models, then evaluate them on the held-out test set.

Usage (from the repository root):
    python scripts/run_experiments.py           # full experiment (config.yaml)
    python scripts/run_experiments.py --smoke   # fast end-to-end check on a small subset;
                                                # writes to models/smoke and reports/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, setup_logging  # noqa: E402
from src.evaluate import run_evaluation  # noqa: E402
from src.train import run_training  # noqa: E402


def smoke_config(cfg: dict) -> dict:
    """Tiny subset and few epochs. Results are only a software check, not a scientific result."""
    cfg["data"].update(touch_stride=16, max_objects=100, cache_path="data/processed/smoke.npz")
    cfg["classical"]["max_train_samples"] = 1000
    cfg["cnn"].update(max_epochs=2, early_stopping_patience=2)
    cfg["evaluation"].update(latency_runs=20, fusion_trials=2)
    cfg["paths"].update(models_dir="models/smoke", reports_dir="reports/smoke")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="only evaluate existing models")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    if not args.skip_training:
        run_training(cfg)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
