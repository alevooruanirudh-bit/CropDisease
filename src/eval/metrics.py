"""Prediction and classification metrics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)
from torch import nn
from torch.utils.data import DataLoader

from src.training.trainer import amp_context


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False,
            amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities and labels. TTA averages identity + h-flip + v-flip."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with amp_context(device, amp):
            p = model(x).float().softmax(1)
            if tta:
                p = p + model(x.flip(3)).float().softmax(1) + model(x.flip(2)).float().softmax(1)
                p = p / 3
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy given a 0/1 correctness vector."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    accs = correct[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def compute_metrics(y: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro/weighted P/R/F1, kappa, top-1/top-5, bootstrap CI, per-class report."""
    pred = probs.argmax(1)
    k = len(class_names)
    labels = np.arange(k)
    out: dict = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        out.update({f"precision_{avg}": float(p), f"recall_{avg}": float(r), f"f1_{avg}": float(f)})
    out["cohen_kappa"] = float(cohen_kappa_score(y, pred, labels=labels))
    out["top1"] = out["accuracy"]
    out["top5"] = float(top_k_accuracy_score(y, probs, k=min(5, k), labels=labels)) if k > 2 else 1.0
    lo, hi = bootstrap_ci((pred == y).astype(float))
    out["accuracy_ci95"] = [lo, hi]
    out["report"] = classification_report(y, pred, labels=labels, target_names=class_names, output_dict=True,
                                          zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["per_class_accuracy"] = dict(zip(class_names, np.nan_to_num(cm.diagonal() / cm.sum(1)).round(6).tolist()))
    return out


def text_report(y: np.ndarray, probs: np.ndarray, class_names: list[str]) -> str:
    """sklearn text classification report."""
    return classification_report(y, probs.argmax(1), labels=np.arange(len(class_names)), target_names=class_names,
                                 digits=4, zero_division=0)
