"""PyTorch dataset reading raw or cached-segmented images."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.io import read_rgb
from src.segmentation.borb import Segmenter
from src.segmentation.cache import cached_image_path
from src.utils.common import seed_worker

Perturb = Callable[[np.ndarray, int], np.ndarray]


class LeafDataset(Dataset):
    """Images listed in a split dataframe.

    Pipeline per item: read (cached segmented image if available and no perturbation,
    else raw) -> perturb(img, idx) -> segment -> transform.

    Args:
        df: split dataframe with ``path`` and ``label``.
        root: raw image root.
        transform: albumentations transform returning ``{"image": tensor}``.
        cache_dir: segmentation cache (used only when ``perturb`` is None).
        segmenter: on-the-fly segmenter (used when no cache, or with perturbations).
        perturb: callable ``(img, idx) -> img`` applied *before* segmentation.
    """

    def __init__(self, df: pd.DataFrame, root: str | Path, transform: Callable,
                 cache_dir: Path | None = None, segmenter: Segmenter | None = None,
                 perturb: Perturb | None = None) -> None:
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.root = Path(root)
        self.transform = transform
        self.cache_dir = cache_dir
        self.segmenter = segmenter
        self.perturb = perturb

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, i: int) -> np.ndarray:
        """Return the pre-transform RGB image for index ``i``."""
        rel = self.paths[i]
        if self.perturb is None and self.cache_dir is not None:
            return read_rgb(cached_image_path(self.cache_dir, rel))
        img = read_rgb(self.root / rel)
        if self.perturb is not None:
            img = self.perturb(img, i)
        if self.segmenter is not None:
            img = self.segmenter(img)[0]
        return img

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(image=self.load(i))["image"], int(self.labels[i])


def make_loader(ds: Dataset, batch_size: int, train: bool, num_workers: int, seed: int,
                balanced: bool = False) -> DataLoader:
    """DataLoader with seeded shuffling; optional class-balanced sampling for training."""
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = None
    if train and balanced:
        labels = ds.labels  # type: ignore[attr-defined]
        w = 1.0 / np.bincount(labels)[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels), generator=g)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) > batch_size,  # avoid size-1 BatchNorm batches
                      worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0)
