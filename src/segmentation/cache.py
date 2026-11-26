"""Segment every image once, cache results to disk, and report mask quality."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.io import read_rgb, write_mask, write_rgb
from src.data.splits import SPLITS, load_split
from src.segmentation.borb import build_segmenter
from src.utils.common import get_logger, load_json, save_json

log = get_logger(__name__)

LOW_COVER, HIGH_COVER = 0.05, 0.95


def seg_tag(seg_cfg: dict, thresholds: dict | None) -> str:
    """Stable cache key for a segmentation config (+ thresholds actually used)."""
    th = {k: thresholds[k] for k in ("t_B", "t_b", "pol_B", "pol_b")} if thresholds else {}
    blob = json.dumps({"seg": seg_cfg, "th": th}, sort_keys=True)
    return f"{seg_cfg.get('method', 'none')}_{seg_cfg.get('output', 'black')}_{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def thresholds_path(cfg: dict) -> Path:
    """Where the auto-threshold JSON for this dataset/split lives."""
    return Path(cfg["data"]["split_dir"]) / "borb_thresholds.json"


def ensure_thresholds(cfg: dict) -> dict | None:
    """Load (or compute on the TRAINING split only) the BorB thresholds for this dataset."""
    method = cfg["segmentation"].get("method", "none")
    if method not in ("borb_original", "borb_plus"):
        return None
    p = thresholds_path(cfg)
    if p.exists():
        return load_json(p)
    from src.segmentation.thresholds import compute_and_save
    train = load_split(cfg["data"]["split_dir"], "train")
    n = min(len(train), cfg["segmentation"].get("threshold_sample", 2000))
    sample = train.sample(n=n, random_state=0) if n < len(train) else train
    root = Path(cfg["data"]["root"])
    imgs = [read_rgb(root / pth) for pth in tqdm(sample["path"], desc="thresholds", leave=False)]
    th = compute_and_save(imgs, p, cfg["segmentation"].get("hist_sigma", 3.0),
                          cfg["segmentation"].get("hist_min_prominence", 0.05))
    return th


def cache_dir(cfg: dict, thresholds: dict | None) -> Path:
    """Cache folder for the current dataset + segmentation config."""
    return Path(cfg["paths"]["cache_dir"]) / cfg["data"]["name"] / seg_tag(cfg["segmentation"], thresholds)


def cached_image_path(cdir: Path, rel: str) -> Path:
    return cdir / "img" / (str(Path(rel).with_suffix("")) + ".png")


def cached_mask_path(cdir: Path, rel: str) -> Path:
    return cdir / "mask" / (str(Path(rel).with_suffix("")) + ".png")


def build_cache(cfg: dict, splits: tuple[str, ...] = SPLITS, workers: int = 8) -> Path | None:
    """Segment all images in ``splits`` into the cache and write ``mask_quality.csv/json``.

    Returns the cache dir, or None when segmentation is disabled.
    """
    th = ensure_thresholds(cfg)
    segmenter = build_segmenter(cfg["segmentation"], th)
    if segmenter is None:
        return None
    cdir = cache_dir(cfg, th)
    done_flag = cdir / "DONE.json"
    if done_flag.exists() and set(load_json(done_flag)["splits"]) >= set(splits):
        return cdir
    root = Path(cfg["data"]["root"])
    rows = pd.concat([load_split(cfg["data"]["split_dir"], s).assign(split=s) for s in splits], ignore_index=True)

    def work(rel: str) -> float:
        img = read_rgb(root / rel)
        seg, mask = segmenter(img)
        write_rgb(cached_image_path(cdir, rel), seg)
        write_mask(cached_mask_path(cdir, rel), mask)
        return float(mask.mean())

    with ThreadPoolExecutor(workers) as ex:
        cover = list(tqdm(ex.map(work, rows["path"]), total=len(rows), desc=f"segment[{cdir.name}]", leave=False))
    rows["coverage"] = cover
    q = mask_quality(rows)
    rows[["path", "split", "class_name", "coverage"]].to_csv(cdir / "mask_quality.csv", index=False)
    save_json(q, cdir / "mask_quality.json")
    save_json({"splits": list(splits), "segmentation": cfg["segmentation"],
               "thresholds": {k: th[k] for k in ("t_B", "t_b", "pol_B", "pol_b")} if th else None},
              done_flag)
    log.info("Segmentation cache %s: failure rate %.2f%% (<%d%%: %d, >%d%%: %d)", cdir.name,
             100 * q["failure_rate"], int(LOW_COVER * 100), q["n_low"], int(HIGH_COVER * 100), q["n_high"])
    return cdir


def mask_quality(rows: pd.DataFrame) -> dict:
    """Sanity metric: share of masks covering < 5% or > 95% of the image."""
    c = rows["coverage"].to_numpy()
    low, high = c < LOW_COVER, c > HIGH_COVER
    fails = rows[low | high]
    per_class = rows.assign(fail=low | high).groupby("class_name")["fail"].mean().round(4).to_dict()
    return {
        "n": int(len(c)), "n_low": int(low.sum()), "n_high": int(high.sum()),
        "failure_rate": float((low | high).mean()) if len(c) else 0.0,
        "coverage_mean": float(c.mean()) if len(c) else 0.0,
        "coverage_p05": float(np.percentile(c, 5)) if len(c) else 0.0,
        "coverage_p95": float(np.percentile(c, 95)) if len(c) else 0.0,
        "per_class_failure_rate": per_class,
        "failures": fails[["path", "coverage"]].head(500).to_dict("records"),
    }
