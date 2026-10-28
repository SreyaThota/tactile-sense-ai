"""Tests for dataset loading, validation and leakage-free splitting (synthetic data)."""

import json
import shutil

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import synthetic_contact
from src.data_loader import (
    TactileDataset,
    assert_no_leakage,
    group_split,
    load_image_folder,
    load_tactile_mnist,
    make_splits,
    validation_report,
)


def test_folder_loader_classes_and_groups(folder_dataset):
    ds = load_image_folder(folder_dataset, 64)
    assert ds.class_names == ["horizontal", "vertical"]
    assert len(ds) == 48 and ds.images.shape == (48, 64, 64, 3)
    assert len(np.unique(ds.groups)) == 12  # 6 recordings per class
    assert ds.report["samples_per_class"] == {"horizontal": 24, "vertical": 24}


def test_folder_loader_skips_invalid_and_duplicates(folder_dataset):
    (folder_dataset / "horizontal" / "broken.png").write_bytes(b"garbage")
    shutil.copy(folder_dataset / "vertical" / "rec0__f0.png", folder_dataset / "vertical" / "copy.png")
    ds = load_image_folder(folder_dataset, 64)
    assert ds.report["invalid_images_skipped"] == 1
    assert ds.report["duplicate_images_skipped"] == 1
    assert len(ds) == 48


def test_folder_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_folder(tmp_path / "nope", 64)
    (tmp_path / "only_one_class").mkdir()
    with pytest.raises(ValueError):
        load_image_folder(tmp_path, 64)


def test_inconsistent_group_labels_rejected():
    ds = TactileDataset(np.zeros((2, 4, 4, 3), np.uint8), np.array([0, 1]), np.array(["g", "g"]), ["a", "b"])
    with pytest.raises(ValueError):
        validation_report(ds)


def test_group_split_keeps_groups_together():
    labels = np.repeat(np.arange(4), 25)
    groups = np.array([f"g{i // 5}" for i in range(100)])  # 20 groups of 5 frames
    tr, te = group_split(labels, groups, 0.4, seed=0)
    assert not set(groups[tr]) & set(groups[te])
    assert len(tr) + len(te) == 100
    # stratified at group level: every class is in the held-out part
    assert set(labels[te]) == {0, 1, 2, 3}


def test_make_splits_folder_no_leakage(folder_dataset):
    ds = load_image_folder(folder_dataset, 64)
    splits = make_splits(ds, val_fraction=0.25, test_fraction=0.34, seed=1)
    assert sum(len(v) for v in splits.values()) == len(ds)
    assert_no_leakage(ds.groups, splits)
    assert all(len(v) for v in splits.values())


def test_assert_no_leakage_detects_overlap():
    groups = np.array(["a", "a", "b"])
    with pytest.raises(RuntimeError):
        assert_no_leakage(groups, {"train": np.array([0]), "test": np.array([1, 2])})


def _write_fake_parquet(path, labels, rng, touches=4):
    """Parquet file with the same columns/encoding as Real Tactile MNIST (synthetic images)."""
    rows = {"label": [], "object_id": [], "gel_pose_cell_frame.position": [], "sensor_image": []}
    for i, label in enumerate(labels):
        frames = [{"bytes": cv2.imencode(".png", synthetic_contact(label % 2, rng))[1].tobytes(), "path": None}
                  for _ in range(touches)]
        rows["label"].append(label)
        rows["object_id"].append(1000 + i + (0 if "train" in path.name else 500))
        rows["gel_pose_cell_frame.position"].append([[0.0, 0.0, 0.008]] * touches)
        rows["sensor_image"].append(frames)
    meta = {b"huggingface": json.dumps({"info": {"features": {"label": {"names": ["zero", "one"]}}}}).encode()}
    pq.write_table(pa.table(rows).replace_schema_metadata(meta), path)


def test_tactile_mnist_parquet_loader(tmp_path, rng):
    _write_fake_parquet(tmp_path / "train-00000-of-00001.parquet", [0, 1, 0, 1], rng)
    _write_fake_parquet(tmp_path / "test-00000-of-00001.parquet", [0, 1], rng)
    ds = load_tactile_mnist(tmp_path, 64, touch_stride=2)
    assert ds.class_names == ["zero", "one"]
    assert len(ds) == (4 + 2) * 2  # stride 2 over 4 touches
    assert set(ds.official_split) == {"train", "test"}
    assert ds.gel_z is not None and np.allclose(ds.gel_z, 0.008)
    # all touches of one object share its group id
    assert len(np.unique(ds.groups)) == 6


def test_tactile_mnist_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tactile_mnist(tmp_path, 64)
