"""YAML config loading with inheritance (`_base_`) and dotted CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge ``upd`` into a copy of ``base`` and return it."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config. A ``_base_`` key (path relative to the file) is merged first."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_update(load_config(path.parent / base), cfg)
    return cfg


def _parse_value(raw: str) -> Any:
    """Parse a CLI override value with YAML semantics (ints, floats, bools, lists)."""
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides to a config dict."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a dotted key from a nested dict."""
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def save_config(cfg: dict, path: str | Path) -> None:
    """Write a resolved config to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
