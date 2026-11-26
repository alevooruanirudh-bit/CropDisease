"""Grad-CAM for any timm model via ``forward_features`` / ``forward_head``."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def gradcam(model: nn.Module, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM heatmap in [0, 1] at input resolution for one image ``x`` (1, C, H, W).

    Works for CNNs (NCHW features) and Swin-style NHWC features.
    Returns (heatmap HxW, class index explained).
    """
    model.eval()
    x = x.requires_grad_(False)
    with torch.enable_grad():
        feats = model.forward_features(x)
        nhwc = feats.ndim == 4 and feats.shape[-1] == getattr(model, "num_features", -1) and feats.shape[1] != feats.shape[-1]
        if nhwc:
            feats = feats.permute(0, 3, 1, 2)
        feats.retain_grad()
        logits = model.forward_head(feats.permute(0, 2, 3, 1) if nhwc else feats)
        cls = int(logits.argmax(1)) if target is None else int(target)
        model.zero_grad(set_to_none=True)
        logits[0, cls].backward()
    g = feats.grad[0]
    a = feats.detach()[0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((w * a).sum(0)).float().cpu().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, cls


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap onto an RGB uint8 image of the same size."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def background_energy(cam: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Share of Grad-CAM energy that falls outside the leaf mask (0 = all on the leaf)."""
    if leaf_mask.shape != cam.shape:
        leaf_mask = cv2.resize(leaf_mask.astype(np.uint8), cam.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    tot = cam.sum()
    return float(cam[~leaf_mask].sum() / tot) if tot > 0 else 0.0
