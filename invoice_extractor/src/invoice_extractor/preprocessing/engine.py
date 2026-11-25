"""OpenCV preprocessing engine — pure functions on BGR arrays.

Pipeline (fixed, in this order):
  1. downscale — cap the long edge at 2048 px (photos in, model budget out)
  2. shadow removal — divide by a heavy Gaussian (same scale, so the page
     becomes locally neutral white; dark corners on phone shots stop eating
     contrast)
  3. CLAHE — bounded contrast enhancement on the LAB lightness channel
     (clipLimit/tileGridSize are the agent's two main knobs)
  4. unsharp-mask sharpen — optional; amplifies sensor noise, so it's off by
     default and capped at 0.8
  5. deskew — rotate the dominant sheet back to axis-aligned for small
     angles only; a large rotation means the photo is sideways, which the
     model handles fine on its own

The engine never touches the disk and never raises on "weird but decodable"
images; it raises ``ValueError`` on undecodable input. Callers (tools layer)
own all file I/O.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .config import PreprocessConfig

#: Long edge above which the image is downscaled before anything else.
MAX_LONG_EDGE = 2048

#: Deskew only corrects small tilts; anything larger is an orientation
#: problem the OCR model is better off solving by itself.
_DESKEW_MAX_DEG = 10.0


def load_image(path) -> np.ndarray:
    """Decode a file to a BGR ndarray. Raises ValueError when it can't be read."""
    if not os.path.isfile(str(path)):
        raise ValueError(f"not a readable image file: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def encode_image(img: np.ndarray, path) -> bytes:
    """Encode to the same container format the source file used (by extension).

    Used for both the model payloads (base64 of these bytes) and the attempt
    history BLOBs, so every recorded attempt is a self-contained re-runnable
    image.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:  # .jpg/.jpeg/.webp and friends default to JPEG-quality-ish
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    ok, buf = cv2.imencode(ext or ".jpg", img, params)
    if not ok:
        # Fall back to a guaranteed format the model can read, rather than
        # failing an extraction because of a container choice.
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError(f"could not encode enhanced image for {path}")
    return np.asarray(buf).tobytes()


# --- individual stages --------------------------------------------------------


def _downscale(img: np.ndarray, max_edge: int = MAX_LONG_EDGE) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return cv2.resize(img, (max(1, int(w * scale + 0.5)), max(1, int(h * scale + 0.5))), interpolation=cv2.INTER_AREA)


def _remove_shadow(img: np.ndarray) -> np.ndarray:
    """Normalize uneven illumination by dividing out a heavy low-frequency estimate.

    A large-sigma Gaussian captures the page's illumination field (corners
    darker than the center on phone shots) without touching text edges.
    ``cv2.divide(…)`` scales the result so a neutral 255 background stays 255.
    """
    blur = cv2.GaussianBlur(img, (0, 0), 50)
    return cv2.divide(img, blur, scale=255)


def _clahe(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Bounded contrast enhancement on the lightness channel (color-safe)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp-mask; strength 0.0 is a no-op."""
    if strength <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Rotate the dominant sheet back to axis-aligned, for small angles only."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 200).astype(np.uint8)  # "not paper" = dark pixels
    if mask.sum() < max(64, mask.size // 500):
        return img  # nothing with a shape to measure
    # Largest connected cluster of dark pixels ≈ the sheet/region of interest,
    # robust to a few scattered dark corners.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels.size <= 1:
        return img
    # stats rows: [x, y, w, h, area]; index 0 is background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x, y, rw, rh = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], \
        stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    if rw * rh < 0.02 * w * h:
        return img  # too small a cluster to trust as "the sheet"
    # minAreaRect wants a float32 array of (x, y) points, not a mask image.
    ys, xs = np.where(_crop_mask(mask, x, y, rw, rh) > 0)
    rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    angle = _rect_angle(rect)
    if not angle:
        return img
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _crop_mask(full: np.ndarray, x: int, y: int, rw: int, rh: int) -> np.ndarray:
    return full[y : y + rh, x : x + rw]


def _rect_angle(rect) -> float | None:
    """Rotation to re-align the box, or None when it's already aligned / sideways.

    OpenCV reports the angle of the *width* edge; the other edge is a quarter
    turn away, so any rotation deg + 90k re-aligns the rectangle. Take the
    smallest such value — it undoes the tilt without swapping which way the
    sheet's long axis points.
    """
    (cx, cy), (rw, rh), deg = rect
    if min(rw, rh) < 8:
        return None
    tilt = deg % 90.0
    if tilt > 45.0:
        tilt -= 90.0
    if abs(tilt) < 0.25:
        return None  # below measurement noise; rotating would only add artifacts
    if abs(tilt) > _DESKEW_MAX_DEG:
        return None  # sideways photo; the OCR model handles that better than us
    return tilt


# --- public pipeline ----------------------------------------------------------


def process(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Run the full pipeline for one attempt. Pure: returns a new array."""
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
        raise ValueError("process() expects a 3-channel BGR ndarray")
    out = _downscale(img)
    out = _remove_shadow(out)
    out = _clahe(out, cfg.clahe_clip, int(cfg.clahe_tile))
    out = _sharpen(out, float(cfg.sharpen_strength))
    out = _deskew(out)
    return out
