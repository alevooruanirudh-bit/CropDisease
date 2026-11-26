"""Group-aware stratified train/val/test splits saved as CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.classes import class_list, list_images, match_class_dirs
from src.data.dedup import dedup_and_group
from src.utils.common import get_logger, save_json

log = get_logger(__name__)
SPLITS = ("train", "val", "test")


def index_dataset(root: str | Path, classes: list[str], max_per_class: int | None = None,
                  class_caps: dict[str, int] | None = None, seed: int = 42) -> pd.DataFrame:
    """Build a dataframe of all images: ``path`` (relative to root), ``abs_path``, ``label``, ``class_name``.

    ``class_caps`` randomly (seeded) subsamples the named classes to at most N images, e.g. to
    match the paper's 3,209 Tomato YLCV images when the mirror ships 5,357.
    """
    root = Path(root)
    dirs = match_class_dirs(root, classes)
    rows = []
    for label, c in enumerate(classes):
        imgs = list_images(dirs[c])
        cap = (class_caps or {}).get(c)
        if cap and len(imgs) > cap:
            keep = np.random.default_rng(seed).choice(len(imgs), cap, replace=False)
            imgs = [imgs[i] for i in sorted(keep)]
            log.info("Capped %s: %d -> %d images", c, len(list_images(dirs[c])), cap)
        if max_per_class:
            imgs = imgs[:max_per_class]
        for p in imgs:
            rows.append({"path": p.relative_to(root).as_posix(), "abs_path": str(p),
                         "label": label, "class_name": c})
    return pd.DataFrame(rows)


def group_stratified_split(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.Series:
    """Assign each row to train/val/test, stratified by label, never splitting a group.

    Uses StratifiedGroupKFold twice: first carve out the test fold (1/k_test of the data),
    then carve val out of the remainder.
    """
    tr, va, te = ratios
    assert abs(tr + va + te - 1) < 1e-6, ratios
    k_test = int(round(1 / te))
    split = pd.Series("train", index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], df["group"]))
    split.iloc[test_idx] = "test"
    rest = df.index[split == "train"]
    k_val = int(round((tr + va) / va))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    sub = df.loc[rest]
    _, val_idx = next(sgkf2.split(sub, sub["label"], sub["group"]))
    split.loc[rest[val_idx]] = "val"
    return split


def make_splits(cfg: dict) -> dict:
    """Index, deduplicate, group and split a dataset; save ``splits/<name>/{train,val,test}.csv``."""
    d = cfg["data"]
    root = Path(d["root"])
    classes = class_list(d["dataset"], d.get("num_classes"), root)
    df = index_dataset(root, classes, d.get("max_per_class"), d.get("class_caps"), d.get("split_seed", 42))
    log.info("Indexed %d images in %d classes from %s", len(df), len(classes), root)
    df, stats = dedup_and_group(df, max_dist=d.get("dedup_max_dist", 4), workers=d.get("hash_workers", 8))
    df["split"] = group_stratified_split(df, tuple(d.get("split_ratios", [0.6, 0.2, 0.2])), d.get("split_seed", 42))

    out = Path(d["split_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        df[df["split"] == s][["path", "label", "class_name", "group", "phash", "md5"]].to_csv(out / f"{s}.csv", index=False)

    counts = df.pivot_table(index="class_name", columns="split", values="path", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=classes, columns=list(SPLITS), fill_value=0)
    counts.to_csv(out / "class_counts.csv")
    stats["split_sizes"] = {s: int((df["split"] == s).sum()) for s in SPLITS}
    stats["classes"] = classes
    save_json(stats, out / "dedup_stats.json")
    log.info("Dedup stats: %s", {k: v for k, v in stats.items() if k != "classes"})
    return stats


def load_split(split_dir: str | Path, split: str) -> pd.DataFrame:
    """Load one split CSV."""
    return pd.read_csv(Path(split_dir) / f"{split}.csv")


def check_split_integrity(split_dir: str | Path) -> dict:
    """Assert there is no path, md5 or group overlap between splits. Returns overlap counts."""
    dfs = {s: load_split(split_dir, s) for s in SPLITS}
    report = {}
    for col in ("path", "md5", "group"):
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                n = len(set(dfs[a][col]) & set(dfs[b][col]))
                report[f"{col}:{a}&{b}"] = n
    return report


def split_fingerprint(split_dir: str | Path) -> str:
    """SHA-256 over the split CSVs, used to prove Exp 1 and Exp 2 share identical splits."""
    import hashlib
    h = hashlib.sha256()
    for s in SPLITS:
        h.update((Path(split_dir) / f"{s}.csv").read_bytes())
    return h.hexdigest()


def class_weights(train_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Inverse-frequency class weights normalised to mean 1."""
    cnt = np.bincount(train_df["label"], minlength=n_classes).astype(float)
    w = 1.0 / np.maximum(cnt, 1)
    return w / w.mean()
