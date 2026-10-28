"""OpenCV preprocessing for vision-based tactile images (GelSight-style sensors).

A GelSight sensor images an elastomer gel lit by coloured LEDs from different sides. When an
object presses into the gel, the local surface normals change and so does the colour/shading of
those pixels. The most robust contact cue is therefore the *change relative to a no-contact
reference frame*, not the absolute colour: the raw frames contain a strong, static illumination
gradient (vignetting, LED colours) that is identical for every frame of the same sensor.

All functions work on uint8 BGR images (OpenCV's native channel order) unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ContactRegion:
    """Geometric description of the detected contact area (pixel units)."""

    mask: np.ndarray            # uint8 {0, 255}
    area_fraction: float        # contact pixels / all pixels
    num_regions: int            # connected contact blobs
    bbox: tuple[int, int, int, int] | None   # x, y, w, h of the union of all blobs
    centroid: tuple[float, float] | None     # x, y of the contact mask
    contours: list[np.ndarray]

    @property
    def has_contact(self) -> bool:
        return self.bbox is not None


# --------------------------------------------------------------------------- loading


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as a 3-channel uint8 BGR array, raising on unreadable files."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{path.suffix}' for {path}")
    # imdecode + fromfile instead of imread so that non-ASCII Windows paths also work.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image (corrupt or not an image): {path}")
    return validate_image(image)


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (PNG/JPEG) into a BGR array."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return validate_image(image)


def validate_image(image: np.ndarray) -> np.ndarray:
    """Check dtype/shape and return a 3-channel uint8 image (grey images are expanded)."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image is empty or not a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"Unsupported image shape {image.shape}")
    return image


# --------------------------------------------------------------------------- basic ops


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. INTER_AREA averages pixels when shrinking, which avoids aliasing
    of the fine gel texture; INTER_LINEAR is used when enlarging."""
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image
    interp = cv2.INTER_AREA if size < min(h, w) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR -> RGB for display with Matplotlib / Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """HSV separates the LED hue from brightness; useful to inspect which illumination
    channel responds to a contact."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def denoise(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Gaussian blur suppresses sensor noise and the fine gel texture before thresholding,
    so that the contact mask is not dominated by single noisy pixels."""
    if kernel <= 1:
        return image
    if kernel % 2 == 0:
        raise ValueError("Blur kernel size must be odd")
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def normalize(image: np.ndarray) -> np.ndarray:
    """Min-max normalise to float32 in [0, 1] (contrast stretch for visualisation/statistics)."""
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


def image_statistics(image: np.ndarray) -> dict[str, float]:
    """Basic intensity statistics of the grey-scale image and per-channel means."""
    gray = to_gray(image).astype(np.float32)
    b, g, r = (image[..., i].astype(np.float32).mean() for i in range(3))
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_min": float(gray.min()),
        "gray_max": float(gray.max()),
        "mean_b": float(b),
        "mean_g": float(g),
        "mean_r": float(r),
    }


# --------------------------------------------------------------------------- contact


def estimate_reference(images: np.ndarray) -> np.ndarray:
    """Estimate the no-contact reference frame as the pixel-wise median over many frames.

    Contacts appear at different positions in different frames, so for every pixel most frames
    show the undeformed gel and the median recovers the background. Should be computed on
    training data only.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3) images, got {images.shape}")
    return np.median(images, axis=0).astype(np.float32)


def difference_image(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed per-channel difference to the reference (float32). Removes the static
    illumination pattern so that only gel deformation remains."""
    if image.shape != reference.shape:
        raise ValueError(f"Image shape {image.shape} does not match reference {reference.shape}")
    return image.astype(np.float32) - reference


def contact_magnitude(image: np.ndarray, reference: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """Per-pixel deformation strength: max absolute channel difference after denoising.
    The max over channels is used because, depending on the slope direction, the contact
    shows up in a different LED colour channel."""
    diff = np.abs(difference_image(image, reference)).max(axis=2)
    return denoise(diff, blur_kernel)


def detect_contact(
    image: np.ndarray,
    reference: np.ndarray,
    diff_threshold: float = 15,
    blur_kernel: int = 5,
    morph_kernel: int = 3,
    min_contact_fraction: float = 0.02,
) -> ContactRegion:
    """Segment the contact region by thresholding the difference to the reference frame.

    A fixed threshold (rather than Otsu) is used on purpose: Otsu always splits the histogram in
    two, so it would "find" a contact even in frames where the gel is untouched.
    """
    magnitude = contact_magnitude(image, reference, blur_kernel)
    mask = (magnitude > diff_threshold).astype(np.uint8) * 255
    if morph_kernel > 1:
        # Morphological opening removes isolated false-positive pixels from noise.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    area_fraction = float((mask > 0).mean())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if area_fraction < min_contact_fraction or not contours:
        return ContactRegion(mask, area_fraction, 0, None, None, list(contours))

    bbox = cv2.boundingRect(np.vstack(contours))
    m = cv2.moments(mask, binaryImage=True)
    centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return ContactRegion(
        mask=mask,
        area_fraction=area_fraction,
        num_regions=len(contours),
        bbox=tuple(int(v) for v in bbox),
        centroid=(float(centroid[0]), float(centroid[1])),
        contours=list(contours),
    )


def contact_area_batch(images: np.ndarray, reference: np.ndarray, cfg: dict) -> np.ndarray:
    """Contact area fraction for a batch of images (used for filtering no-contact frames)."""
    return np.array(
        [
            detect_contact(
                img,
                reference,
                cfg["diff_threshold"],
                cfg["blur_kernel"],
                cfg["morph_kernel"],
                cfg["min_contact_fraction"],
            ).area_fraction
            for img in images
        ],
        dtype=np.float32,
    )


def draw_contact_overlay(image: np.ndarray, region: ContactRegion) -> np.ndarray:
    """Draw the contact contours, bounding box and centroid on a copy of the image (BGR)."""
    out = image.copy()
    cv2.drawContours(out, region.contours, -1, (0, 255, 255), 1)
    if region.bbox is not None:
        x, y, w, h = region.bbox
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
    if region.centroid is not None:
        cx, cy = (int(round(v)) for v in region.centroid)
        cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 6, 1)
    return out
