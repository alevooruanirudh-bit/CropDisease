"""Deterministic test-time perturbations, applied to the RAW image (before any segmentation).

Every perturbed image is a pure function of (base_seed, perturbation, severity, image path),
so Exp 1 and Exp 2 models see bit-identical inputs, and reruns reproduce exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from src.robustness.textures import TextureBank
from src.utils.common import stable_seed

PerturbFn = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return _clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape))


def gaussian_blur(img: np.ndarray, k: float, rng: np.random.Generator) -> np.ndarray:
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def brightness(img: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    return _clip(img.astype(np.float32) * factor)


def contrast(img: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    mean = f.mean(axis=(0, 1), keepdims=True)
    return _clip((f - mean) * factor + mean)


def jpeg(img: np.ndarray, quality: float, rng: np.random.Generator) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def gamma(img: np.ndarray, g: float, rng: np.random.Generator) -> np.ndarray:
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return lut[img]


def rotation(img: np.ndarray, deg: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    ang = float(deg) * (1 if rng.random() < 0.5 else -1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def scale(img: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, float(s))
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def occlusion(img: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Random erasing: one rectangle covering ``frac`` of the image, filled with uniform noise."""
    h, w = img.shape[:2]
    area = frac * h * w
    ar = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    rh = int(min(h, round(np.sqrt(area * ar))))
    rw = int(min(w, round(area / max(rh, 1))))
    y0 = int(rng.integers(0, h - rh + 1))
    x0 = int(rng.integers(0, w - rw + 1))
    out = img.copy()
    out[y0:y0 + rh, x0:x0 + rw] = rng.integers(0, 256, (rh, rw, 3), dtype=np.uint8)
    return out


SIMPLE: dict[str, PerturbFn] = {
    "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur, "brightness": brightness,
    "contrast": contrast, "jpeg": jpeg, "gamma": gamma, "rotation": rotation, "scale": scale,
    "occlusion": occlusion,
}

DEFAULT_SEVERITIES: dict[str, list] = {
    "gaussian_noise": [5, 15, 25],
    "gaussian_blur": [3, 7, 11],
    "brightness": [0.5, 0.7, 1.3, 1.6],
    "contrast": [0.5, 0.7, 1.5],
    "jpeg": [50, 20],
    "gamma": [0.5, 2.0],
    "rotation": [15, 30, 45],
    "scale": [0.7, 0.85, 1.2],
    "background": ["soil", "grass", "clutter"],
    "occlusion": [0.1, 0.2, 0.3],
}


@dataclass
class Perturbation:
    """Callable ``(img, idx) -> img`` for one (family, severity) over a fixed list of image paths.

    For ``background`` the leaf is cut out with ``leaf_extractor`` (the BorB+ mask on the clean
    image) and composited onto a held-out texture of kind ``severity``.
    """
    name: str
    severity: float | str
    paths: list[str]
    base_seed: int = 1234
    leaf_extractor: Callable[[np.ndarray], np.ndarray] | None = None
    textures: TextureBank | None = None

    def rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(stable_seed(self.base_seed, self.name, self.severity, self.paths[idx]))

    def __call__(self, img: np.ndarray, idx: int) -> np.ndarray:
        rng = self.rng(idx)
        if self.name == "clean":
            return img
        if self.name == "background":
            if self.leaf_extractor is None or self.textures is None:
                raise ValueError("background perturbation needs leaf_extractor and textures")
            mask = self.leaf_extractor(img)
            tex = self.textures.get(str(self.severity), img.shape[:2], rng)
            return np.where(mask[..., None], img, tex).astype(np.uint8)
        return SIMPLE[self.name](img, float(self.severity), rng)
