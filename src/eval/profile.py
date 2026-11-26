"""Parameter count, FLOPs and inference latency."""
from __future__ import annotations

import copy
import time

import torch
from torch import nn

from src.models.factory import count_params


def flops_g(model: nn.Module, img_size: int) -> float | None:
    """GFLOPs (multiply-adds x2) for one image, via thop on a CPU copy. None if unsupported."""
    try:
        from thop import profile
        m = copy.deepcopy(model).cpu().eval()
        macs, _ = profile(m, inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)
        return float(2 * macs / 1e9)
    except Exception:  # thop does not handle every op (e.g. some attention blocks)
        return None


@torch.no_grad()
def latency_ms(model: nn.Module, img_size: int, device: torch.device, n: int = 30, warmup: int = 5) -> float:
    """Mean single-image forward latency in milliseconds on ``device``."""
    model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model(x)
    sync()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    sync()
    return 1000 * (time.perf_counter() - t) / n


def profile_model(model: nn.Module, img_size: int, device: torch.device) -> dict:
    """Params (M), GFLOPs and latency (ms/img, batch 1)."""
    return {"params_m": count_params(model) / 1e6, "gflops": flops_g(model, img_size),
            "latency_ms_per_img": latency_ms(model, img_size, device), "latency_device": device.type}
