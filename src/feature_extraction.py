"""Hand-crafted features for the classical baseline.

Features are computed on the *difference to the no-contact reference frame*, so that the static
illumination pattern of the gel does not dominate them:

* HOG of the signed grey-level difference - captures the orientation of the edges of the
  imprint (the stroke direction of the digit is the main class cue).
* Histogram of the contact magnitude - how strong / how widespread the deformation is.
* Per-channel difference mean/std - which LED colour responds (encodes slope direction).
* Contact-region geometry - area, number of blobs, bounding box size and centroid.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import contact_magnitude, detect_contact, difference_image


class FeatureExtractor:
    def __init__(self, reference: np.ndarray, pre_cfg: dict, feat_cfg: dict):
        self.reference = reference
        self.pre = pre_cfg
        size = pre_cfg["image_size"]
        cell, block = feat_cfg["hog_cell"], feat_cfg["hog_block"]
        if size % cell or block % cell:
            raise ValueError("image_size and hog_block must be multiples of hog_cell")
        # winSize, blockSize, blockStride, cellSize, nbins
        self.hog = cv2.HOGDescriptor((size, size), (block, block), (cell, cell), (cell, cell), feat_cfg["hog_bins"])
        self.hist_bins = feat_cfg["hist_bins"]

    def signed_gray_difference(self, image: np.ndarray) -> np.ndarray:
        """Grey-level difference to the reference, mapped to uint8 around 128 for HOG."""
        diff = difference_image(image, self.reference).mean(axis=2)
        return np.clip(128 + 2 * diff, 0, 255).astype(np.uint8)

    def extract(self, image: np.ndarray) -> np.ndarray:
        size = self.pre["image_size"]
        hog = self.hog.compute(self.signed_gray_difference(image)).ravel()

        magnitude = contact_magnitude(image, self.reference, self.pre["blur_kernel"])
        hist, _ = np.histogram(magnitude, bins=self.hist_bins, range=(0, 80))
        hist = hist.astype(np.float32) / magnitude.size

        diff = difference_image(image, self.reference)
        channel_stats = np.concatenate([diff.mean(axis=(0, 1)), diff.std(axis=(0, 1))]) / 50.0

        region = detect_contact(
            image,
            self.reference,
            self.pre["diff_threshold"],
            self.pre["blur_kernel"],
            self.pre["morph_kernel"],
            self.pre["min_contact_fraction"],
        )
        if region.has_contact:
            x, y, w, h = region.bbox
            cx, cy = region.centroid
            geometry = [region.area_fraction, region.num_regions / 5, w / size, h / size, cx / size, cy / size]
        else:
            geometry = [region.area_fraction, 0, 0, 0, 0.5, 0.5]

        return np.concatenate([hog, hist, channel_stats, np.asarray(geometry, np.float32)]).astype(np.float32)

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self.extract(img) for img in images])
