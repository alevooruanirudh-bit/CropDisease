"""Matplotlib figures: confusion matrices, training curves, image grids, robustness curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Fixed colours so BorB / no-BorB read the same in every figure.
COLORS = {"exp1_borb": "#1f6fb4", "exp2_noborb": "#d1495b"}


def short_names(classes: list[str]) -> list[str]:
    """Compact class labels for axes."""
    out = []
    for c in classes:
        crop, _, dis = c.partition("___")
        crop = crop.replace(",_bell", "").replace("_(maize)", "").replace("_(including_sour)", "")
        out.append(f"{crop[:6]}:{dis.replace('_', ' ')[:18]}" if dis else c[:22])
    return out


def save_confusion(cm: np.ndarray, classes: list[str], path: str | Path, normalize: bool) -> None:
    """Absolute or row-normalised confusion matrix heatmap."""
    cm = np.asarray(cm, dtype=float)
    data = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    n = len(classes)
    side = min(0.55 * n + 2, 24)  # large label sets: fixed-size heatmap without tick labels
    ticks = short_names(classes) if n <= 60 else False
    fig, ax = plt.subplots(figsize=(max(5, side), max(4, min(0.5 * n + 1.5, 22))))
    sns.heatmap(data, annot=n <= 20, fmt=".2f" if normalize else ".0f", cmap="Blues", cbar=True, ax=ax,
                xticklabels=ticks, yticklabels=ticks, annot_kws={"size": 7},
                vmin=0, vmax=1 if normalize else None, linewidths=0.3 if n <= 60 else 0, linecolor="white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else " (counts)"))
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_curves(hist: pd.DataFrame, path: str | Path) -> None:
    """Loss and accuracy curves (train vs val) with the phase boundary marked."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(hist["epoch"], hist[f"train_{key}"], label="train", color="#555")
        ax.plot(hist["epoch"], hist[f"val_{key}"], label="val", color="#1f6fb4")
        frozen = hist[hist["phase"] == "frozen"]
        if len(frozen) and len(frozen) < len(hist):
            ax.axvline(frozen["epoch"].max() + 0.5, color="#999", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_image_grid(rows: list[list[np.ndarray]], col_titles: list[str], path: str | Path,
                    row_titles: list[str] | None = None, cell: float = 1.8) -> None:
    """Grid of RGB/boolean images; masks are shown in grayscale."""
    nr, nc = len(rows), len(col_titles)
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr), squeeze=False)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax = axes[i][j]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1 if im.dtype == bool else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8)
            if j == 0 and row_titles:
                ax.set_ylabel(row_titles[i], fontsize=6)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_robustness_plot(df: pd.DataFrame, path: str | Path) -> None:
    """Accuracy vs severity per perturbation family, BorB vs no-BorB (mean ± std over seeds).

    ``df`` columns: exp, perturbation, severity_idx, severity, accuracy (one row per seed).
    """
    fams = [f for f in df["perturbation"].unique() if f not in ("clean", "plantdoc")]
    nc = 4
    nr = int(np.ceil(len(fams) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
    clean = df[df["perturbation"] == "clean"].groupby("exp")["accuracy"].mean()
    for ax, fam in zip(axes.ravel(), fams):
        sub = df[df["perturbation"] == fam]
        for exp, g in sub.groupby("exp"):
            agg = g.groupby(["severity_idx", "severity"])["accuracy"].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values("severity_idx")
            x = np.arange(len(agg))
            c = COLORS.get(exp, None)
            ax.errorbar(x, agg["mean"], yerr=agg["std"].fillna(0), marker="o", ms=3, lw=1.5, capsize=2,
                        color=c, label=exp)
            if exp in clean:
                ax.axhline(clean[exp], color=c, ls=":", lw=0.8)
            ax.set_xticks(x, [str(s) for s in agg["severity"]], fontsize=7)
        ax.set_title(fam, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)
