"""Automatic BorB thresholds from the TRAINING split's mean channel histograms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.segmentation.borb import PAPER_T_B, PAPER_T_LAB_B, channels, estimate_polarity
from src.utils.common import get_logger, save_json

log = get_logger(__name__)


def mean_histograms(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean normalised 256-bin histograms of the RGB-B and Lab-b channels."""
    hB = np.zeros(256)
    hb = np.zeros(256)
    for img in images:
        B, b = channels(img)
        hB += np.bincount(B.ravel(), minlength=256) / B.size
        hb += np.bincount(b.ravel(), minlength=256) / b.size
    return hB / len(images), hb / len(images)


def otsu_on_hist(hist: np.ndarray) -> int:
    """Otsu threshold for a 256-bin histogram (maximises between-class variance)."""
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sb = np.nan_to_num(sb)
    return int(np.argmax(sb))


def find_valley(hist: np.ndarray, sigma: float = 3.0, min_prominence: float = 0.05) -> tuple[int, str]:
    """Threshold at the valley between the two most prominent modes of a smoothed histogram.

    Falls back to Otsu when fewer than two modes with prominence >= ``min_prominence`` x max
    are found. Returns (threshold, method) where method is ``valley`` or ``otsu``.
    """
    s = gaussian_filter1d(hist.astype(float), sigma)
    peaks, props = find_peaks(s, prominence=min_prominence * s.max())
    if len(peaks) < 2:
        return otsu_on_hist(hist), "otsu"
    top2 = np.sort(peaks[np.argsort(props["prominences"])[-2:]])
    lo, hi = int(top2[0]), int(top2[1])
    return lo + int(np.argmin(s[lo:hi + 1])), "valley"


def estimate_thresholds(images: list[np.ndarray], sigma: float = 3.0, min_prominence: float = 0.05) -> dict:
    """Estimate polarity and BorB+ thresholds from training images."""
    pol = estimate_polarity(images)
    hB, hb = mean_histograms(images)
    tB, mB = find_valley(hB, sigma, min_prominence)
    tb, mb = find_valley(hb, sigma, min_prominence)
    return {
        "t_B": tB, "t_b": tb, "method_B": mB, "method_b": mb,
        "pol_B": pol["pol_B"], "pol_b": pol["pol_b"], "polarity_votes": pol,
        "paper_t_B": PAPER_T_B, "paper_t_b": PAPER_T_LAB_B,
        "hist_B": hB.tolist(), "hist_b": hb.tolist(), "n_images": len(images),
    }


def save_threshold_plot(th: dict, path: str | Path) -> None:
    """Plot the mean histograms with the auto and paper thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, key, t, tp, name in [(axes[0], "hist_B", th["t_B"], th["paper_t_B"], "RGB B channel"),
                                 (axes[1], "hist_b", th["t_b"], th["paper_t_b"], "Lab b channel (8-bit)")]:
        h = np.array(th[key])
        ax.plot(h, color="#444", lw=1, label="mean hist")
        ax.plot(gaussian_filter1d(h, 3), color="#1f77b4", lw=1.5, label="smoothed")
        ax.axvline(t, color="#d62728", ls="--", label=f"auto={t}")
        ax.axvline(tp, color="#2ca02c", ls=":", label=f"paper={tp}")
        ax.set_title(name)
        ax.set_xlim(0, 255)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compute_and_save(images: list[np.ndarray], out_json: str | Path, sigma: float = 3.0,
                     min_prominence: float = 0.05) -> dict:
    """Estimate thresholds, save JSON + plot, log the result."""
    th = estimate_thresholds(images, sigma, min_prominence)
    save_json(th, out_json)
    save_threshold_plot(th, Path(out_json).with_suffix(".png"))
    log.info("Thresholds: t_B=%s (%s, pol=%s) t_b=%s (%s, pol=%s); paper 130/135; polarity votes B-low=%.3f b-high=%.3f",
             th["t_B"], th["method_B"], th["pol_B"], th["t_b"], th["method_b"], th["pol_b"],
             th["polarity_votes"]["vote_B_low"], th["polarity_votes"]["vote_b_high"])
    return th
