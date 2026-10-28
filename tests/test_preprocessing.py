"""Tests for the OpenCV preprocessing functions (synthetic images, see conftest.py)."""

import cv2
import numpy as np
import pytest

from conftest import synthetic_background, synthetic_contact
from src import preprocessing as pp


def test_load_image_roundtrip(tmp_path, background):
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), background)
    loaded = pp.load_image(path)
    assert loaded.shape == (64, 64, 3) and loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, background)


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.load_image(tmp_path / "missing.png")
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        pp.load_image(bad)
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(ValueError):
        pp.load_image(txt)


def test_validate_image_expands_gray_and_rejects_float():
    gray = np.zeros((10, 10), np.uint8)
    assert pp.validate_image(gray).shape == (10, 10, 3)
    with pytest.raises(ValueError):
        pp.validate_image(np.zeros((10, 10, 3), np.float32))


def test_resize_and_conversions(background):
    big = cv2.resize(background, (240, 320))
    assert pp.resize(big, 64).shape == (64, 64, 3)
    assert pp.to_gray(background).shape == (64, 64)
    np.testing.assert_array_equal(pp.to_rgb(background)[..., 0], background[..., 2])
    assert pp.to_hsv(background).shape == (64, 64, 3)


def test_denoise_requires_odd_kernel(background):
    assert pp.denoise(background, 1) is background
    with pytest.raises(ValueError):
        pp.denoise(background, 4)


def test_normalize_range(background):
    n = pp.normalize(background)
    assert n.dtype == np.float32 and n.min() == 0.0 and n.max() == pytest.approx(1.0)
    assert not pp.normalize(np.full((4, 4, 3), 7, np.uint8)).any()


def test_image_statistics_keys(background):
    stats = pp.image_statistics(background)
    assert {"gray_mean", "gray_std", "mean_b", "mean_g", "mean_r"} <= stats.keys()
    assert 0 <= stats["gray_min"] <= stats["gray_mean"] <= stats["gray_max"] <= 255


def test_reference_is_median_background(rng):
    frames = np.stack([synthetic_contact(i % 2, rng) for i in range(21)])
    ref = pp.estimate_reference(frames)
    # Imprints sit at different places/orientations, so the median recovers the background.
    assert np.abs(ref - synthetic_background()).mean() < 3


def test_detect_contact_finds_imprint(rng, background, pre_cfg):
    img = synthetic_contact(0, rng)
    region = pp.detect_contact(img, background.astype(np.float32), pre_cfg["diff_threshold"],
                               pre_cfg["blur_kernel"], pre_cfg["morph_kernel"], pre_cfg["min_contact_fraction"])
    assert region.has_contact
    x, y, w, h = region.bbox
    assert w > h  # horizontal bar
    cx, cy = region.centroid
    assert 20 < cx < 44 and 20 < cy < 44
    assert 0.03 < region.area_fraction < 0.3


def test_detect_contact_none_on_background(background, rng, pre_cfg):
    noisy = (background + rng.normal(0, 2, background.shape)).clip(0, 255).astype(np.uint8)
    region = pp.detect_contact(noisy, background.astype(np.float32), pre_cfg["diff_threshold"])
    assert not region.has_contact and region.bbox is None and region.centroid is None


def test_difference_shape_mismatch(background):
    with pytest.raises(ValueError):
        pp.difference_image(background, np.zeros((32, 32, 3), np.float32))


def test_overlay_does_not_modify_input(rng, background):
    img = synthetic_contact(1, rng)
    copy = img.copy()
    region = pp.detect_contact(img, background.astype(np.float32))
    out = pp.draw_contact_overlay(img, region)
    np.testing.assert_array_equal(img, copy)
    assert out.shape == img.shape and not np.array_equal(out, img)
