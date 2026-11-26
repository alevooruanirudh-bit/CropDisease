"""Perceptual-hash deduplication and near-duplicate grouping.

* Exact duplicates (byte-identical files) are removed, keeping the first file in sorted
  order. Images with identical pHash but different bytes are counted and *grouped*, not
  removed: distinct leaves on PlantVillage's uniform background can collide in pHash.
* Near-duplicates (pHash Hamming distance <= ``max_dist``) are linked with
  union-find. Each connected component becomes one *group*; splits are made by
  group so near-duplicates never cross train/val/test.

Grouping is done within each class: a near-duplicate across classes would be a
label conflict, which we count and report separately instead of grouping.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _hash_one(path: Path) -> tuple[str, int]:
    with open(path, "rb") as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()
    with Image.open(path) as im:
        ph = imagehash.phash(im.convert("RGB"))
    return md5, int(str(ph), 16)


def compute_hashes(paths: list[Path], workers: int = 8,
                   cache: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Return (md5 list, uint64 pHash array) for ``paths``.

    With ``cache`` (a CSV of path, size, mtime, md5, phash), files whose size and mtime are unchanged
    reuse their stored hashes, so re-running after a label fix does not re-hash every image."""
    stat = [p.stat() for p in paths]
    keys = [(p.as_posix(), s.st_size, int(s.st_mtime)) for p, s in zip(paths, stat)]
    known: dict[tuple, tuple[str, int]] = {}
    if cache is not None and Path(cache).exists():
        c = pd.read_csv(cache, dtype={"phash": str})
        known = {(p, int(sz), int(mt)): (m, int(h, 16))
                 for p, sz, mt, m, h in zip(c["path"], c["size"], c["mtime"], c["md5"], c["phash"])}
    todo = [i for i, k in enumerate(keys) if k not in known]
    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(_hash_one, [paths[i] for i in todo]), total=len(todo), desc="pHash", leave=False))
    known.update({keys[i]: r for i, r in zip(todo, res)})
    md5s = [known[k][0] for k in keys]
    ph = np.array([known[k][1] for k in keys], dtype=np.uint64)
    if cache is not None and todo:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": [k[0] for k in known], "size": [k[1] for k in known], "mtime": [k[2] for k in known],
                      "md5": [v[0] for v in known.values()], "phash": [f"{v[1]:016x}" for v in known.values()]}
                     ).to_csv(cache, index=False)
    return md5s, ph


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    # Fallback for numpy < 2.0: byte-wise lookup table.
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(ph: np.ndarray, max_dist: int, block: int = 2048) -> np.ndarray:
    """All index pairs (i<j) with Hamming distance <= ``max_dist``. Blocked all-pairs."""
    n = len(ph)
    pairs = []
    for s in range(0, n, block):
        a = ph[s:s + block, None]
        d = _popcount64(a ^ ph[None, :])
        ii, jj = np.nonzero(d <= max_dist)
        ii = ii + s
        keep = jj > ii
        pairs.append(np.stack([ii[keep], jj[keep]], 1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)


def dedup_and_group(df: pd.DataFrame, max_dist: int = 4, workers: int = 8,
                    hash_cache: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Deduplicate and group images.

    Args:
        df: columns ``path`` (absolute or root-relative resolvable), ``label``, ``class_name``,
            and ``abs_path`` (absolute path used for reading).
        max_dist: Hamming threshold for near-duplicates.
        hash_cache: optional CSV reused / updated by :func:`compute_hashes`.

    Returns:
        (kept dataframe with ``md5``, ``phash`` (hex), ``group`` columns, stats dict).
    """
    df = df.sort_values("path").reset_index(drop=True)
    md5s, ph = compute_hashes([Path(p) for p in df["abs_path"]], workers, hash_cache)
    df["md5"] = md5s
    df["phash_int"] = ph

    stats: dict = {"n_input": int(len(df))}

    # 1) byte-identical duplicates (any class)
    dup_md5 = df.duplicated("md5", keep="first")
    md5_label_conflicts = int(df[df.duplicated("md5", keep=False)].groupby("md5")["label"].nunique().gt(1).sum())
    stats["exact_byte_duplicates_removed"] = int(dup_md5.sum())
    stats["exact_byte_duplicate_label_conflicts"] = md5_label_conflicts
    df = df[~dup_md5].reset_index(drop=True)

    # 2) identical pHash within class (grouped below, not removed)
    stats["identical_phash_within_class"] = int(df.duplicated(["label", "phash_int"], keep="first").sum())

    # 3) near-duplicate grouping within class
    groups = np.empty(len(df), dtype=object)
    n_pairs = 0
    for label, sub in df.groupby("label", sort=True):
        idx = sub.index.to_numpy()
        pairs = near_duplicate_pairs(sub["phash_int"].to_numpy(np.uint64), max_dist)
        n_pairs += len(pairs)
        uf = _UnionFind(len(idx))
        for i, j in pairs:
            uf.union(int(i), int(j))
        for k, gi in enumerate(idx):
            groups[gi] = f"{label}_{uf.find(k)}"
    df["group"] = groups
    sizes = df["group"].value_counts()
    stats.update({
        "near_duplicate_pairs": int(n_pairs),
        "n_groups": int(len(sizes)),
        "n_images_in_multi_groups": int(sizes[sizes > 1].sum()),
        "n_multi_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
    })

    # 4) cross-class near-duplicates (reported, not grouped)
    cross = near_duplicate_pairs(df["phash_int"].to_numpy(np.uint64), max_dist)
    if len(cross):
        lab = df["label"].to_numpy()
        stats["cross_class_near_duplicate_pairs"] = int((lab[cross[:, 0]] != lab[cross[:, 1]]).sum())
    else:
        stats["cross_class_near_duplicate_pairs"] = 0
    stats["n_kept"] = int(len(df))

    df["phash"] = [f"{int(h):016x}" for h in df["phash_int"]]
    df = df.drop(columns=["phash_int"])
    return df, stats
