"""Image IO helpers (RGB uint8 everywhere)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as RGB uint8. Raises on unreadable files."""
    # np.fromfile + imdecode instead of cv2.imread: works with non-ASCII paths on Windows too
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, img: np.ndarray) -> None:
    """Write an RGB uint8 image (format from extension)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(Path(path).suffix or ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Could not write {path}")
    buf.tofile(str(path))


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    """Write a boolean mask as an 8-bit PNG (0/255)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise IOError(f"Could not write {path}")
    buf.tofile(str(path))


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask PNG as boolean."""
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Could not read mask {path}")
    return m > 127
