"""Backbone factory (timm, ImageNet weights) and freezing helpers."""
from __future__ import annotations

import timm
import torch
from torch import nn

from src.utils.common import get_logger

log = get_logger(__name__)

# Short names used in configs -> timm model names with ImageNet-1k weights.
BACKBONES: dict[str, str] = {
    # paper backbones
    "vgg16": "vgg16.tv_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    # stronger backbones
    "efficientnetv2_s": "tf_efficientnetv2_s.in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
}


def timm_name(name: str) -> str:
    """Resolve a short backbone name (or pass a raw timm name through)."""
    return BACKBONES.get(name, name)


def build_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2,
                img_size: int | None = None) -> nn.Module:
    """Create a timm classifier with a fresh ``num_classes`` head."""
    kwargs: dict = {"pretrained": pretrained, "num_classes": num_classes, "drop_rate": drop_rate}
    if img_size and "swin" in name:
        kwargs["img_size"] = img_size
    model = timm.create_model(timm_name(name), **kwargs)
    log.info("Built %s (%s), %.2fM params, pretrained=%s", name, timm_name(name), count_params(model) / 1e6, pretrained)
    return model


def head_param_names(model: nn.Module) -> list[str]:
    """Parameter-name prefixes that belong to the classification head."""
    cls = model.pretrained_cfg.get("classifier", "fc")
    prefixes = [cls] if isinstance(cls, str) else list(cls)
    if hasattr(model, "pre_logits") and any(p.requires_grad for p in model.pre_logits.parameters()):
        prefixes.append("pre_logits")  # VGG's fc6/fc7 equivalent: train it together with the head
    return prefixes


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze every parameter except the head."""
    heads = head_param_names(model)
    for n, p in model.named_parameters():
        p.requires_grad = trainable or any(n.startswith(h) for h in heads)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def init_from_checkpoint(model: nn.Module, path: str) -> int:
    """Warm-start: copy every tensor whose name and shape match (the backbone; a resized head is skipped)."""
    src = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    log.info("Warm-started from %s: %d/%d tensors copied", path, len(keep), len(own))
    return len(keep)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """Load a checkpoint saved by the trainer; returns its metadata."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return ck.get("meta", {})
