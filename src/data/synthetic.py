"""Synthetic PlantVillage-like images for smoke tests: a green leaf on a light-grey background.

Class ``healthy`` has a clean leaf; class ``spotted`` has brown lesions. A few exact and
near-duplicates are planted so the dedup step has something to find.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.io import write_rgb


def synth_leaf(rng: np.random.Generator, size: int = 128, lesions: int = 0,
               bg: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (RGB image, true leaf mask)."""
    bg = bg or tuple(int(v) for v in rng.integers(175, 215, 3))
    img = np.empty((size, size, 3), np.uint8)
    img[:] = bg
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    c = (int(size / 2 + rng.integers(-size // 10, size // 10)), int(size / 2 + rng.integers(-size // 10, size // 10)))
    axes = (int(size * rng.uniform(0.25, 0.38)), int(size * rng.uniform(0.14, 0.22)))
    cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    green = np.array([rng.integers(40, 90), rng.integers(110, 170), rng.integers(20, 60)], np.uint8)
    leaf = np.clip(green.astype(np.int16) + rng.integers(-10, 11, img.shape), 0, 255).astype(np.uint8)
    img[mask > 0] = leaf[mask > 0]
    ys, xs = np.nonzero(mask)
    for _ in range(lesions):
        k = int(rng.integers(0, len(ys)))
        cv2.circle(img, (int(xs[k]), int(ys[k])), int(rng.integers(2, max(3, size // 20))),
                   (int(rng.integers(90, 130)), int(rng.integers(55, 80)), int(rng.integers(20, 40))), -1)
    return img, mask.astype(bool)


def make_synthetic_dataset(root: str | Path, n_per_class: int = 25, size: int = 128, seed: int = 0) -> Path:
    """Write ``root/{healthy,spotted}/*.png`` (``2 * n_per_class`` files, including planted duplicates)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for cname, les in (("healthy", 0), ("spotted", 6)):
        d = root / cname
        d.mkdir(parents=True, exist_ok=True)
        n_unique = n_per_class - 2
        last = None
        for i in range(n_unique):
            img, _ = synth_leaf(rng, size, les)
            write_rgb(d / f"{cname}_{i:03d}.png", img)
            last = img
        assert last is not None
        write_rgb(d / f"{cname}_dup_exact.png", last)                       # byte-identical duplicate
        write_rgb(d / f"{cname}_dup_near.png", np.clip(last.astype(np.int16) + 3, 0, 255).astype(np.uint8))
    return root
