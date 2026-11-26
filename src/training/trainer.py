"""Two-phase fine-tuning loop (frozen backbone -> full fine-tune) with AMP and early stopping."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.factory import count_params, set_backbone_trainable
from src.utils.common import get_logger

log = get_logger(__name__)


@dataclass
class TrainResult:
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    history: pd.DataFrame
    ckpt_path: Path
    train_seconds: float


def warmup_cosine(total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    """LambdaLR factor: linear warmup then cosine decay to ``min_ratio``."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def amp_context(device: torch.device, enabled: bool):
    """Autocast context when enabled: fp16 on CUDA (with GradScaler), bf16 on MPS (same range as fp32, so no
    loss scaling is needed); no-op otherwise."""
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "mps" else torch.float16)


def resolve_amp(setting: str | bool, device: torch.device) -> bool:
    """``auto`` -> AMP on CUDA only (MPS fp16 autocast is opt-in)."""
    if isinstance(setting, bool):
        return setting and device.type != "cpu"
    return device.type == "cuda" if setting == "auto" else bool(setting)


@torch.no_grad()
def evaluate_loss_acc(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[float, float]:
    """Mean CE loss (no smoothing) and accuracy over a loader."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_context(device, amp):
            out = model(x)
        out = out.float()
        tot_loss += ce(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / max(n, 1), correct / max(n, 1)


class Trainer:
    """Config-driven trainer. See ``train`` block in configs/base.yaml."""

    def __init__(self, model: nn.Module, cfg: dict, device: torch.device, out_dir: Path,
                 class_weights: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.t = cfg["train"]
        self.channels_last = bool(self.t.get("channels_last", False))
        self.model = model.to(device, memory_format=torch.channels_last) if self.channels_last else model.to(device)
        self.device = device
        self.out_dir = Path(out_dir)
        self.amp = resolve_amp(self.t.get("amp", "auto"), device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and device.type == "cuda")
        w = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=self.t.get("label_smoothing", 0.1))
        self.mixup = None
        mx = cfg.get("augment", {}).get("mixup", {})
        if mx.get("enabled", False):
            from timm.data import Mixup
            from timm.loss import SoftTargetCrossEntropy
            self.mixup = Mixup(mixup_alpha=mx.get("mixup_alpha", 0.2), cutmix_alpha=mx.get("cutmix_alpha", 0.2),
                               prob=mx.get("prob", 0.5), label_smoothing=self.t.get("label_smoothing", 0.1),
                               num_classes=cfg["data"]["n_classes"])
            self.soft_criterion = SoftTargetCrossEntropy()

    # ------------------------------------------------------------------ helpers
    def _optimizer(self, lr: float) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": self.t.get("weight_decay", 0.05)},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    def _train_epoch(self, loader: DataLoader, opt: torch.optim.Optimizer, sched) -> tuple[float, float]:
        self.model.train()
        # Frozen BatchNorm layers must not update running statistics, or the frozen phase drifts the
        # backbone's normalisation away from what its (frozen) weights expect.
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        # Running sums stay on the device: a .item() per step forces a GPU sync, which is costly on MPS.
        tot = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        n = 0
        clip = self.t.get("grad_clip", 1.0)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y_hard = y
            if self.mixup is not None and len(x) % 2 == 0:
                x, y_soft = self.mixup(x, y)
            else:
                y_soft = None
            with amp_context(self.device, self.amp):
                out = self.model(x)
                loss = self.soft_criterion(out, y_soft) if y_soft is not None else self.criterion(out, y)
            opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if clip:
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(opt)
            self.scaler.update()
            sched.step()
            tot += loss.detach().float() * len(y_hard)
            correct += (out.detach().argmax(1) == y_hard).sum()
            n += len(y_hard)
            if step % 50 == 0 and not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
        if not torch.isfinite(tot):
            raise FloatingPointError("Non-finite training loss")
        return tot.item() / max(n, 1), correct.item() / max(n, 1)

    # ------------------------------------------------------------------ main loop
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainResult:
        """Run both phases; early stop on val loss; keep best checkpoint by val acc."""
        t = self.t
        freeze_epochs = int(t.get("freeze_epochs", 3))
        max_epochs = int(t.get("epochs", 30))
        patience = int(t.get("early_stopping_patience", 8))
        ckpt = self.out_dir / "best.pt"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        best_acc, best_loss_for_acc, best_epoch = -1.0, float("inf"), -1
        best_val_loss, since_improve = float("inf"), 0
        hist = []
        t0 = time.time()
        steps_per_epoch = max(1, len(train_loader))

        phases = []
        if freeze_epochs > 0:
            phases.append(("frozen", min(freeze_epochs, max_epochs), t.get("lr_head", 1e-3)))
        if max_epochs - freeze_epochs > 0:
            phases.append(("finetune", max_epochs - freeze_epochs, t.get("lr", 1e-4)))

        epoch = 0
        stop = False
        for phase, n_ep, lr in phases:
            set_backbone_trainable(self.model, phase == "finetune")
            opt = self._optimizer(lr)
            total = n_ep * steps_per_epoch
            warm = int(t.get("warmup_epochs", 1) * steps_per_epoch) if phase == "finetune" else 0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, warm, t.get("min_lr_ratio", 0.01)))
            log.info("Phase %s: %d epochs, lr=%.2e, trainable=%.2fM", phase, n_ep, lr,
                     count_params(self.model, trainable_only=True) / 1e6)
            # Patience restarts when the backbone is unfrozen, against this phase's own best loss: the first
            # fine-tune epochs (high lr, warmup) rarely beat the frozen phase, which would stop training early.
            since_improve, best_val_loss = 0, float("inf")
            for _ in range(n_ep):
                te = time.time()
                tr_loss, tr_acc = self._train_epoch(train_loader, opt, sched)
                va_loss, va_acc = evaluate_loss_acc(self.model, val_loader, self.device, self.amp)
                epoch += 1
                hist.append({"epoch": epoch, "phase": phase, "lr": opt.param_groups[0]["lr"],
                             "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                             "seconds": time.time() - te})
                improved_acc = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss_for_acc)
                if improved_acc:
                    best_acc, best_loss_for_acc, best_epoch = va_acc, va_loss, epoch
                    torch.save({"model": self.model.state_dict(),
                                "meta": {"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss}}, ckpt)
                if va_loss < best_val_loss - 1e-4:
                    best_val_loss, since_improve = va_loss, 0
                else:
                    since_improve += 1
                log.info("ep %3d [%s] train %.4f/%.4f  val %.4f/%.4f  %s(%.0fs)", epoch, phase, tr_loss, tr_acc,
                         va_loss, va_acc, "*" if improved_acc else " ", time.time() - te)
                pd.DataFrame(hist).to_csv(self.out_dir / "history.csv", index=False)
                if phase == "finetune" and since_improve >= patience:
                    log.info("Early stopping at epoch %d (val loss did not improve for %d epochs)", epoch, patience)
                    stop = True
                    break
            if stop:
                break

        return TrainResult(best_acc, best_loss_for_acc, best_epoch, epoch, pd.DataFrame(hist), ckpt, time.time() - t0)
