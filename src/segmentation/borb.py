"""BorB segmentation (Ozcan & Polat, IEEE Access 2025) and the BorB+ extension.

All functions take and return **RGB uint8** arrays of shape (H, W, 3); masks are
boolean (H, W).

BorB = (mask from RGB B channel) OR (mask from Lab b channel). With OpenCV 8-bit Lab,
b is offset by 128, so neutral grey/white backgrounds sit near 128 while
green/yellow leaf tissue is > 128. The leaf is low in RGB blue. The *polarity*
(leaf = below or above the threshold) is therefore ``B: low``, ``b: high`` for
PlantVillage; :func:`estimate_polarity` checks this from data instead of assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import cv2
import numpy as np

# Paper thresholds (read off the histogram valley of PlantVillage's mean histogram).
PAPER_T_B = 130
PAPER_T_LAB_B = 135


@dataclass
class BorbParams:
    """Thresholds and polarities for the two BorB channel masks."""
    t_B: float = PAPER_T_B
    t_b: float = PAPER_T_LAB_B
    pol_B: str = "low"    # leaf where RGB-B <  t_B
    pol_b: str = "high"   # leaf where Lab-b >  t_b

    @classmethod
    def from_dict(cls, d: dict) -> "BorbParams":
        return cls(**{k: d[k] for k in ("t_B", "t_b", "pol_B", "pol_b") if k in d})


@dataclass
class BorbPlusOptions:
    """Options for :func:`borb_plus` (see configs ``segmentation.*``)."""
    morphology: bool = True
    kernel_frac: float = 0.02          # elliptical kernel size as a fraction of min(H, W)
    fill_holes: bool = True
    min_component_frac: float = 0.02   # keep extra components larger than this fraction of the image
    lesion_protect: bool = True
    bg_chroma_protect: bool = True     # also keep in-hull pixels whose Lab (a,b) differs from this image's background
    bg_chroma_dist: float = 10.0
    grabcut: bool = False
    grabcut_iters: int = 3
    output: str = "black"              # "black" | "crop"
    crop_pad: float = 0.05
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- channels
def channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (RGB blue channel, OpenCV 8-bit Lab b channel)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img[..., 2], lab[..., 2]


def _thresh(ch: np.ndarray, t: float, pol: str) -> np.ndarray:
    return ch < t if pol == "low" else ch > t


def raw_masks(img: np.ndarray, p: BorbParams) -> tuple[np.ndarray, np.ndarray]:
    """The two per-channel masks (B mask, b mask)."""
    B, b = channels(img)
    return _thresh(B, p.t_B, p.pol_B), _thresh(b, p.t_b, p.pol_b)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep RGB where mask is True, black elsewhere."""
    return img * mask[..., None].astype(img.dtype)


# ----------------------------------------------------------------------------- polarity
def border_and_center(h: int, w: int, border: float = 0.06, center: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for an outer border ring and a central box."""
    bm = np.zeros((h, w), bool)
    bh, bw = max(1, int(h * border)), max(1, int(w * border))
    bm[:bh], bm[-bh:], bm[:, :bw], bm[:, -bw:] = True, True, True, True
    cm = np.zeros((h, w), bool)
    ch, cw = int(h * (1 - center) / 2), int(w * (1 - center) / 2)
    cm[ch:h - ch, cw:w - cw] = True
    return bm, cm


def estimate_polarity(images: list[np.ndarray]) -> dict:
    """Decide mask polarity from data: compare channel means in the image centre (leaf)
    vs the border ring (background), voting over ``images``.

    Returns ``{"pol_B": "low"|"high", "pol_b": ..., "vote_B_low": frac, "vote_b_high": frac}``.
    """
    votes_B_low, votes_b_high = [], []
    for img in images:
        B, b = channels(img)
        bm, cm = border_and_center(*B.shape)
        votes_B_low.append(B[cm].mean() < B[bm].mean())
        votes_b_high.append(b[cm].mean() > b[bm].mean())
    fB, fb = float(np.mean(votes_B_low)), float(np.mean(votes_b_high))
    return {"pol_B": "low" if fB >= 0.5 else "high", "pol_b": "high" if fb >= 0.5 else "low",
            "vote_B_low": fB, "vote_b_high": fb, "n": len(images)}


# ----------------------------------------------------------------------------- BorB original
def borb_original(img: np.ndarray, p: BorbParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Faithful BorB: OR of the two fixed-threshold masks, background set to black.

    Returns (segmented RGB image, boolean mask).
    """
    p = p or BorbParams()
    mB, mb = raw_masks(img, p)
    mask = mB | mb
    return apply_mask(img, mask), mask


# ----------------------------------------------------------------------------- cleanup helpers
def _ellipse(k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions not connected to the image border."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = ff[1:-1, 1:-1] == 0
    return mask | holes


def keep_components(mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Keep the largest connected component and any component >= ``min_frac`` of the image."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = {int(np.argmax(areas)) + 1}
    keep |= {i + 1 for i, a in enumerate(areas) if a >= min_frac * mask.size}
    return np.isin(lab, list(keep))


def lesion_pixels(img: np.ndarray) -> np.ndarray:
    """Dark or brown/necrotic pixels (HSV heuristics, OpenCV hue 0-180)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    brown = (h >= 5) & (h <= 30) & (s >= 50) & (v >= 20) & (v <= 210)
    dark = v < 70
    return brown | dark


def background_chroma_distance(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel Lab (a, b) distance to the median background colour of the border ring
    (border pixels outside ``mask``). Uses only the image itself."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32)
    bm, _ = border_and_center(*mask.shape)
    ref = bm & ~mask
    if ref.sum() < 20:
        ref = bm
    bg = np.median(lab[ref], axis=0)
    return np.linalg.norm(lab - bg, axis=-1)


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Filled convex hull of all foreground pixels."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is None or len(pts) < 3:
        return out.astype(bool)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def grabcut_refine(img: np.ndarray, mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """GrabCut seeded from ``mask``: eroded mask = sure FG, dilated complement = sure BG."""
    if mask.sum() < 50 or (~mask).sum() < 50:
        return mask
    k = _ellipse(int(0.03 * min(mask.shape)))
    sure_fg = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    sure_bg = ~cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    gc = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
    gc[mask] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def square_crop_box(mask: np.ndarray, pad: float) -> tuple[int, int, int, int]:
    """Square bounding box (y0, y1, x0, x1) around the mask with relative padding, clipped to the image."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, h, 0, w
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
    side = min(max(side, 8), h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, y0 + side, x0, x0 + side


# ----------------------------------------------------------------------------- BorB+
def borb_plus_mask(img: np.ndarray, p: BorbParams, o: BorbPlusOptions) -> dict[str, np.ndarray]:
    """Compute all intermediate BorB+ masks (opening-by-reconstruction, closing, hole filling,
    component filter, lesion/chroma protection inside the hull, optional GrabCut). Keys: ``B``, ``b``, ``or``, ``clean``."""
    mB, mb = raw_masks(img, p)
    m = mB | mb
    out = {"B": mB, "b": mb, "or": m.copy()}
    if o.morphology:
        k = _ellipse(int(o.kernel_frac * min(m.shape)))
        opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
        # Opening by reconstruction: keep whole original components that survive the opening, so
        # specks vanish but thin petioles/stalks attached to the leaf are not cut off.
        _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        survivors = np.unique(lab[opened > 0])
        m8 = np.isin(lab, survivors[survivors > 0]).astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
        m = m8.astype(bool)
        if o.fill_holes:
            m = fill_holes(m)
        m = keep_components(m, o.min_component_frac)
    if o.lesion_protect and m.any():
        hull = convex_hull_mask(m)
        protect = lesion_pixels(img)
        if o.bg_chroma_protect:
            protect |= background_chroma_distance(img, m) > o.bg_chroma_dist
        m = m | (hull & protect)
        if o.fill_holes:
            m = fill_holes(m)
    if o.grabcut:
        m = grabcut_refine(img, m, o.grabcut_iters)
        if o.morphology:
            m = keep_components(m, o.min_component_frac)
    out["clean"] = m
    return out


def borb_plus(img: np.ndarray, p: BorbParams, o: BorbPlusOptions | None = None) -> tuple[np.ndarray, np.ndarray]:
    """BorB+ segmentation. Returns (segmented image, final mask in the *original* frame)."""
    o = o or BorbPlusOptions()
    m = borb_plus_mask(img, p, o)["clean"]
    seg = apply_mask(img, m)
    if o.output == "crop":
        y0, y1, x0, x1 = square_crop_box(m, o.crop_pad)
        seg = seg[y0:y1, x0:x1]
    return seg, m


# ----------------------------------------------------------------------------- factory
Segmenter = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def build_segmenter(seg_cfg: dict, thresholds: dict | None = None) -> Segmenter | None:
    """Create a segmenter from the ``segmentation`` config block.

    Args:
        seg_cfg: config with ``method`` in {none, borb_original, borb_plus, otsu, kmeans}.
        thresholds: dict with t_B/t_b/pol_B/pol_b (from the thresholds JSON). For
            ``borb_original`` only the polarities are used; thresholds stay at the paper's 130/135.
    """
    from src.segmentation.baselines import kmeans_segment, otsu_segment

    method = seg_cfg.get("method", "none")
    if method in (None, "none"):
        return None
    if method == "borb_original":
        p = BorbParams()
        if thresholds:
            p.pol_B, p.pol_b = thresholds.get("pol_B", "low"), thresholds.get("pol_b", "high")
        return partial(borb_original, p=p)
    if method == "borb_plus":
        if not thresholds:
            raise ValueError("borb_plus needs auto thresholds; run threshold estimation first")
        p = BorbParams.from_dict(thresholds)
        fields = BorbPlusOptions.__dataclass_fields__
        o = BorbPlusOptions(**{k: v for k, v in seg_cfg.items() if k in fields})
        return partial(borb_plus, p=p, o=o)
    if method == "otsu":
        return otsu_segment
    if method == "kmeans":
        return partial(kmeans_segment, seed=seg_cfg.get("kmeans_seed", 0))
    raise ValueError(f"Unknown segmentation method {method!r}")
