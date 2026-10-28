"""Dataset loading, validation and leakage-free splitting.

Two sources are supported:

* ``tactile_mnist`` - the Real Tactile MNIST parquet files (Schneider et al., 2025). Each parquet
  row is one physical 3D-printed digit (``object_id``) with 256 touches recorded by a GelSight
  Mini; ``label`` is the digit class. All touches of an object always stay in the same split.
* ``folder`` - a generic folder-per-class layout ``<root>/<class_name>/<file>``. If file names
  follow ``<group>__<anything>.png``, ``<group>`` is used as the recording/trial id so that
  frames of one recording stay together; otherwise every file is its own group.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .config import resolve
from .preprocessing import SUPPORTED_EXTENSIONS, decode_image, load_image, resize

log = logging.getLogger(__name__)


@dataclass
class TactileDataset:
    images: np.ndarray                # (N, H, W, 3) uint8, BGR
    labels: np.ndarray                # (N,) int64
    groups: np.ndarray                # (N,) str - physical object / recording id
    class_names: list[str]
    official_split: np.ndarray | None = None   # (N,) "train"/"test" if the dataset defines one
    gel_z: np.ndarray | None = None   # (N,) gel height at touch [m]; analysis only, never a model input
    report: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx: np.ndarray) -> "TactileDataset":
        return TactileDataset(
            images=self.images[idx],
            labels=self.labels[idx],
            groups=self.groups[idx],
            class_names=self.class_names,
            official_split=None if self.official_split is None else self.official_split[idx],
            gel_z=None if self.gel_z is None else self.gel_z[idx],
        )


# --------------------------------------------------------------------------- Real Tactile MNIST


def _parquet_class_names(parquet_file) -> list[str]:
    """Read class names from the Hugging Face ``ClassLabel`` metadata stored in the parquet."""
    meta = parquet_file.schema_arrow.metadata or {}
    try:
        info = json.loads(meta[b"huggingface"])
        return [str(n) for n in info["info"]["features"]["label"]["names"]]
    except (KeyError, ValueError, TypeError):
        return [str(i) for i in range(10)]


def load_tactile_mnist(
    raw_dir: str | Path, image_size: int, touch_stride: int = 1, max_objects: int | None = None
) -> TactileDataset:
    """Load all ``train-*.parquet`` / ``test-*.parquet`` files of Real Tactile MNIST."""
    import pyarrow.parquet as pq

    raw_dir = resolve(raw_dir)
    files = {s: sorted(raw_dir.glob(f"{s}-*.parquet")) for s in ("train", "test")}
    if not files["train"] or not files["test"]:
        raise FileNotFoundError(
            f"Real Tactile MNIST parquet files not found in {raw_dir}. "
            "Run `python scripts/prepare_data.py` first (see data/README.md)."
        )

    columns = ["label", "object_id", "gel_pose_cell_frame.position", "sensor_image"]
    images, labels, groups, splits, gel_z = [], [], [], [], []
    invalid, hashes, duplicates = 0, set(), 0
    class_names: list[str] | None = None

    for split, paths in files.items():
        n_objects = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            class_names = class_names or _parquet_class_names(pf)
            for batch in pf.iter_batches(batch_size=16, columns=columns):
                rows = batch.to_pydict()
                for label, obj, positions, frames in zip(
                    rows["label"], rows["object_id"], rows["gel_pose_cell_frame.position"], rows["sensor_image"]
                ):
                    if max_objects is not None and n_objects >= max_objects:
                        break
                    n_objects += 1
                    for t in range(0, len(frames), touch_stride):
                        data = frames[t]["bytes"]
                        digest = hashlib.md5(data).digest()
                        if digest in hashes:
                            duplicates += 1
                            continue
                        hashes.add(digest)
                        try:
                            img = resize(decode_image(data), image_size)
                        except ValueError:
                            invalid += 1
                            continue
                        images.append(img)
                        labels.append(int(label))
                        groups.append(f"obj{obj}")
                        splits.append(split)
                        gel_z.append(float(positions[t][2]))
        log.info("Loaded %d %s objects", n_objects, split)

    ds = TactileDataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        class_names=class_names or [str(i) for i in range(10)],
        official_split=np.asarray(splits),
        gel_z=np.asarray(gel_z, dtype=np.float32),
    )
    ds.report = validation_report(ds, invalid=invalid, duplicates=duplicates)
    return ds


# --------------------------------------------------------------------------- folder-per-class


def load_image_folder(root: str | Path, image_size: int) -> TactileDataset:
    """Load ``<root>/<class_name>/<image>`` files. Unreadable files are skipped and counted."""
    root = resolve(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two class sub-folders in {root}, found {len(class_dirs)}")

    images, labels, groups = [], [], []
    invalid, duplicates, hashes = 0, 0, set()
    for label, class_dir in enumerate(class_dirs):
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            log.warning("Class folder %s contains no images", class_dir)
        for path in files:
            digest = hashlib.md5(path.read_bytes()).digest()
            if digest in hashes:
                duplicates += 1
                log.warning("Skipping duplicate image %s", path)
                continue
            hashes.add(digest)
            try:
                img = resize(load_image(path), image_size)
            except ValueError as err:
                invalid += 1
                log.warning("Skipping invalid image: %s", err)
                continue
            images.append(img)
            labels.append(label)
            group = path.stem.split("__")[0] if "__" in path.stem else path.stem
            groups.append(f"{class_dir.name}/{group}")

    if not images:
        raise ValueError(f"No valid images found under {root}")
    ds = TactileDataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        class_names=[d.name for d in class_dirs],
    )
    ds.report = validation_report(ds, invalid=invalid, duplicates=duplicates)
    return ds


# --------------------------------------------------------------------------- validation


def validation_report(ds: TactileDataset, invalid: int = 0, duplicates: int = 0) -> dict[str, Any]:
    """Summarise data quality: invalid/duplicate files, class balance and label consistency."""
    counts = Counter(ds.labels.tolist())
    per_class = {ds.class_names[k]: counts.get(k, 0) for k in range(len(ds.class_names))}
    nonzero = [c for c in per_class.values() if c > 0]
    # Every frame of one physical object/recording must carry the same label.
    group_labels: dict[str, set[int]] = {}
    for g, y in zip(ds.groups, ds.labels):
        group_labels.setdefault(g, set()).add(int(y))
    inconsistent = sorted(g for g, ys in group_labels.items() if len(ys) > 1)

    report = {
        "num_samples": int(len(ds)),
        "num_groups": len(group_labels),
        "invalid_images_skipped": invalid,
        "duplicate_images_skipped": duplicates,
        "samples_per_class": per_class,
        "imbalance_ratio": float(max(nonzero) / min(nonzero)) if nonzero else float("nan"),
        "groups_with_inconsistent_labels": inconsistent,
    }
    if invalid:
        log.warning("%d invalid images were skipped", invalid)
    if duplicates:
        log.warning("%d duplicate images were skipped", duplicates)
    if report["imbalance_ratio"] > 1.5:
        log.warning("Class imbalance detected (max/min = %.2f)", report["imbalance_ratio"])
    if inconsistent:
        raise ValueError(f"Groups with more than one label (annotation error): {inconsistent[:5]}")
    return report


# --------------------------------------------------------------------------- splitting


def group_split(
    labels: np.ndarray, groups: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices so that all samples of a group land on the same side.

    The split is stratified by the (single) label of each group when every class has at least
    two groups; otherwise it falls back to an unstratified group split.
    """
    unique_groups, first = np.unique(groups, return_index=True)
    group_labels = labels[first]
    classes = np.unique(group_labels)
    n_held = int(np.ceil(fraction * len(unique_groups)))
    can_stratify = np.min(np.bincount(group_labels)[classes]) >= 2 and n_held >= len(classes)
    stratify = group_labels if can_stratify else None
    if stratify is None:
        log.warning("Too few groups per class for a stratified split; using a random group split")
    _, held_groups = train_test_split(
        unique_groups, test_size=fraction, random_state=seed, stratify=stratify
    )
    held = np.isin(groups, held_groups)
    return np.flatnonzero(~held), np.flatnonzero(held)


def make_splits(ds: TactileDataset, val_fraction: float, test_fraction: float, seed: int) -> dict[str, np.ndarray]:
    """Return index arrays for train / val / test.

    If the dataset ships an official test split it is used unchanged as the final test set;
    the validation set is then carved out of the official training objects.
    """
    if ds.official_split is not None:
        test = np.flatnonzero(ds.official_split == "test")
        pool = np.flatnonzero(ds.official_split == "train")
    else:
        pool, test = group_split(ds.labels, ds.groups, test_fraction, seed)
    tr, va = group_split(ds.labels[pool], ds.groups[pool], val_fraction, seed)
    splits = {"train": pool[tr], "val": pool[va], "test": test}
    assert_no_leakage(ds.groups, splits)
    return splits


def assert_no_leakage(groups: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(groups[splits[a]]) & set(groups[splits[b]])
            if shared:
                raise RuntimeError(f"Data leakage: {len(shared)} groups appear in both {a} and {b}")


# --------------------------------------------------------------------------- caching / entry point


def load_dataset(cfg: dict[str, Any], rebuild: bool = False) -> TactileDataset:
    """Load the configured dataset, using a compressed ``.npz`` cache when available."""
    dcfg, size = cfg["data"], cfg["preprocessing"]["image_size"]
    if dcfg["source"] == "folder":
        return load_image_folder(dcfg["folder_root"], size)
    if dcfg["source"] != "tactile_mnist":
        raise ValueError(f"Unknown data source '{dcfg['source']}'")

    settings = {"image_size": size, "touch_stride": dcfg["touch_stride"], "max_objects": dcfg["max_objects"]}
    cache = resolve(dcfg["cache_path"])
    if cache.is_file() and not rebuild:
        with np.load(cache, allow_pickle=False) as z:
            if json.loads(str(z["settings"])) == settings:
                log.info("Loading cached dataset %s", cache)
                return TactileDataset(
                    images=z["images"],
                    labels=z["labels"],
                    groups=z["groups"],
                    class_names=[str(c) for c in z["class_names"]],
                    official_split=z["official_split"],
                    gel_z=z["gel_z"],
                    report=json.loads(str(z["report"])),
                )
            log.info("Cache settings differ from config; rebuilding %s", cache)

    ds = load_tactile_mnist(dcfg["raw_dir"], size, dcfg["touch_stride"], dcfg["max_objects"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        images=ds.images,
        labels=ds.labels,
        groups=ds.groups,
        class_names=np.asarray(ds.class_names),
        official_split=ds.official_split,
        gel_z=ds.gel_z,
        report=json.dumps(ds.report),
        settings=json.dumps(settings),
    )
    log.info("Cached %d frames to %s", len(ds), cache)
    return ds
