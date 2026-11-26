"""Held-out background textures for the background-shift test.

If ``root`` contains ``soil/``, ``grass/`` and ``clutter/`` folders of photos (e.g. picked
from DTD, see scripts/download_data.py), a random photo crop is used. Otherwise textures are
generated procedurally (seeded), which is still disjoint from anything seen in training.
None of these images is ever used for training.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.classes import IMG_EXTS
from src.data.io import read_rgb


def _fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = rng.random((s, s)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def procedural_soil(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array([rng.uniform(90, 140), rng.uniform(60, 95), rng.uniform(35, 60)], np.float32)
    n = _fractal_noise(h, w, rng)[..., None]
    img = base * (0.55 + 0.9 * n)
    for _ in range(int(rng.integers(30, 80))):  # pebbles
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(1, max(2, min(h, w) // 40)))
        g = float(rng.uniform(80, 200))
        cv2.circle(img, c, r, (g, g * 0.9, g * 0.8), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def procedural_grass(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (rng.uniform(30, 60), rng.uniform(70, 110), rng.uniform(20, 45))
    img *= (0.7 + 0.6 * _fractal_noise(h, w, rng))[..., None]
    for _ in range(int(rng.integers(250, 500))):  # blades
        x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        ang = float(rng.uniform(-0.6, 0.6)) - np.pi / 2
        ln = float(rng.uniform(0.05, 0.25)) * h
        col = (rng.uniform(40, 130), rng.uniform(100, 200), rng.uniform(20, 80))
        p2 = (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang)))
        cv2.line(img, (int(x), int(y)), p2, col, int(rng.integers(1, 3)), cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def procedural_clutter(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    g0, g1 = rng.uniform(40, 220, 3), rng.uniform(40, 220, 3)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    img = np.repeat(g0 * (1 - t) + g1 * t, h, axis=0).astype(np.float32)
    for _ in range(int(rng.integers(25, 60))):
        col = tuple(float(v) for v in rng.uniform(0, 255, 3))
        kind = rng.integers(0, 3)
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if kind == 0:
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.rectangle(img, p1, p2, col, -1)
        elif kind == 1:
            cv2.circle(img, p1, int(rng.integers(3, max(4, min(h, w) // 5))), col, -1)
        else:
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p1, p2, col, int(rng.integers(1, 6)))
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)


PROCEDURAL = {"soil": procedural_soil, "grass": procedural_grass, "clutter": procedural_clutter}


class TextureBank:
    """Serve textures by kind; photo folders when available, else procedural."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.photos: dict[str, list[Path]] = {}
        if root and Path(root).exists():
            for kind in PROCEDURAL:
                d = Path(root) / kind
                if d.exists():
                    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
                    if files:
                        self.photos[kind] = files

    def source(self, kind: str) -> str:
        return "photos" if kind in self.photos else "procedural"

    def get(self, kind: str, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        h, w = shape
        if kind in self.photos:
            img = read_rgb(self.photos[kind][int(rng.integers(0, len(self.photos[kind])))])
            ih, iw = img.shape[:2]
            s = max(h / ih, w / iw)
            if s > 1:
                img = cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))))
                ih, iw = img.shape[:2]
            y0 = int(rng.integers(0, ih - h + 1))
            x0 = int(rng.integers(0, iw - w + 1))
            return img[y0:y0 + h, x0:x0 + w].copy()
        if kind not in PROCEDURAL:
            raise KeyError(f"Unknown texture kind {kind!r}")
        return PROCEDURAL[kind](h, w, rng)
