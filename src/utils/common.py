"""Seeding, device selection, logging and small IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "plant-borb", log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger. Adds a file handler when ``log_file`` is given.

    Module loggers become children of ``plant-borb`` so a run's file handler (attached to the
    ``plant-borb`` logger) captures every module's messages.
    """
    if name != "plant-borb":
        get_logger("plant-borb")
        child = logging.getLogger(f"plant-borb.{name}")
        if log_file is not None:
            get_logger("plant-borb", log_file)
        return child
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")  # Windows consoles may lack "—", "°", ...
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(h)
        logger.propagate = False
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(fh)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. Optionally request deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive numpy/random seeds from the torch seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device: ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    """Write JSON, converting numpy scalars/arrays."""
    def conv(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    import hashlib
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)
