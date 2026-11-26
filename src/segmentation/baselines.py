"""Otsu and K-means segmentation baselines (paper Table 11)."""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.borb import apply_mask, border_and_center


def _orient(mask: np.ndarray) -> np.ndarray:
    """Flip the mask if it covers most of the border ring (the leaf rarely does)."""
    bm, _ = border_and_center(*mask.shape)
    return ~mask if mask[bm].mean() > 0.5 else mask


def otsu_segment(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Global Otsu on grayscale; the side that dominates the border is background."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _orient(th > 0)
    return apply_mask(img, mask), mask


def kmeans_segment(img: np.ndarray, k: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-means (k=2) on Lab colour; the cluster that dominates the border is background."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(img.shape[:2])
    bm, _ = border_and_center(*labels.shape)
    border_share = [np.mean(labels[bm] == c) for c in range(k)]
    mask = labels != int(np.argmax(border_share))
    return apply_mask(img, mask), mask
