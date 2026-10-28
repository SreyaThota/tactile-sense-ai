"""Download Real Tactile MNIST (64x64, single frame per touch) and build the local cache.

Usage (from the repository root):
    python scripts/prepare_data.py              # download (~133 MB) + validate + cache
    python scripts/prepare_data.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve, setup_logging  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402

log = logging.getLogger("prepare_data")

HF_REPO = "TimSchneider42/tactile-mnist-touch-real-single-t256-64x64"
FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]


def download(raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for remote in FILES:
        target = raw_dir / Path(remote).name
        if target.is_file() and not force:
            log.info("Already present: %s", target)
            continue
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{remote}"
        log.info("Downloading %s", url)
        tmp = target.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as err:
            raise SystemExit(
                f"Download failed ({err}). Download the files manually from "
                f"https://huggingface.co/datasets/{HF_REPO} into {raw_dir}"
            ) from err
        tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download and rebuild the cache")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if cfg["data"]["source"] == "tactile_mnist" and not args.skip_download:
        download(resolve(cfg["data"]["raw_dir"]), args.force)
    ds = load_dataset(cfg, rebuild=args.force)
    log.info("Dataset validation report:\n%s", json.dumps(ds.report, indent=2))


if __name__ == "__main__":
    main()
