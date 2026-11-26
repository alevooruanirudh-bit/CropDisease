"""Entry point: one command per experiment.

    python run.py --config configs/exp1_borb.yaml
    python run.py --config configs/exp2_noborb.yaml
    python run.py --config configs/exp3_robustness.yaml
    python run.py --smoke                      # synthetic data, all three experiments, < 3 min

Stages (``--stage``): prepare (dedup + splits) -> segment (cache + mask QC) -> train -> summary.
Robustness configs run: robustness -> summary. Finished runs are skipped unless ``--force``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.classes import class_list  # noqa: E402
from src.data.splits import check_split_integrity, make_splits  # noqa: E402
from src.utils.common import get_logger, load_json  # noqa: E402
from src.utils.config import apply_overrides, deep_update, load_config  # noqa: E402

log = get_logger("plant-borb")

SMOKE_CONFIGS = ["configs/exp1_borb.yaml", "configs/exp2_noborb.yaml", "configs/exp3_robustness.yaml"]


def prepare(cfg: dict) -> dict:
    """Create (or reuse) dedup'd group-aware splits; attach class names to the config."""
    d = cfg["data"]
    split_dir = Path(d["split_dir"])
    if not (split_dir / "test.csv").exists():
        if d["dataset"] == "manifest":
            raise FileNotFoundError(f"{split_dir} missing: run scripts/ingest_catalog.py and scripts/prepare_web_splits.py")
        if d["dataset"] == "synthetic" and not Path(d["root"]).exists():
            from src.data.synthetic import make_synthetic_dataset
            make_synthetic_dataset(d["root"], n_per_class=d.get("synthetic_per_class", 25))
        make_splits(cfg)
    integ = check_split_integrity(split_dir)
    bad = {k: v for k, v in integ.items() if v}
    if bad:
        raise RuntimeError(f"Split leakage detected in {split_dir}: {bad}")
    stats = load_json(split_dir / "dedup_stats.json")
    if d["dataset"] == "manifest":  # unified web dataset: classes come from scripts/prepare_web_splits.py
        expected = stats["classes"]
    else:
        expected = class_list(d["dataset"], d.get("num_classes"), d["root"] if d["dataset"] == "synthetic" else None)
    if stats["classes"] != expected:
        raise RuntimeError(f"{split_dir} was built for different classes; delete it or change data.split_dir")
    cfg["data"]["class_names"] = stats["classes"]
    cfg["data"]["n_classes"] = len(stats["classes"])
    log.info("Splits %s: %s (no overlap in path/md5/group)", split_dir, stats["split_sizes"])
    return cfg


def segment(cfg: dict):
    """Build the segmentation cache (no-op for method=none)."""
    from src.segmentation.cache import build_cache
    return build_cache(cfg, workers=cfg["data"].get("hash_workers", 8))


def train_all(cfg: dict, cache_dir, force: bool) -> None:
    """Train every backbone x seed (optionally after a val-only HPO search)."""
    import copy

    from src.training.hpo import random_search
    from src.training.pipeline import run_single
    from src.utils.config import apply_overrides as ov

    exp_dir = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]["name"]
    for bb in cfg["backbones"]:
        bcfg = copy.deepcopy(cfg)
        if cfg.get("hpo", {}).get("enabled"):
            best = random_search(bcfg, bb, exp_dir / bb / "hpo", cache_dir)
            bcfg = ov(bcfg, [f"{k}={v}" for k, v in best["best"].items()])
        for seed in cfg["seeds"]:
            out = exp_dir / bb / f"seed{seed}"
            if (out / "metrics.json").exists() and not force:
                log.info("Skip finished run %s", out)
                continue
            if out.exists() and force:
                shutil.rmtree(out)
            run_single(bcfg, bb, seed, out, cache_dir)


def run_config(cfg: dict, stage: str, force: bool) -> None:
    """Run one experiment config through the requested stages."""
    t0 = time.time()
    etype = cfg["experiment"].get("type", "train")
    log.info("=== %s (%s) ===", cfg["experiment"]["name"], etype)
    if etype == "robustness":
        if stage in ("all", "robustness"):
            from src.robustness.runner import run_robustness
            run_robustness(cfg)
    else:
        cfg = prepare(cfg)
        cache = segment(cfg) if stage in ("all", "segment", "train") else None
        if stage in ("all", "train"):
            train_all(cfg, cache, force)
    if stage in ("all", "summary"):
        from src.eval.summary import write_summary
        write_summary(cfg["paths"]["results_dir"], cfg)
    log.info("=== %s done in %.1f min ===", cfg["experiment"]["name"], (time.time() - t0) / 60)


def build_cfg(path: str, args: argparse.Namespace) -> dict:
    cfg = load_config(path)
    if args.smoke:
        cfg = deep_update(cfg, load_config(ROOT / "configs" / "smoke.yaml"))
    if args.backbones:
        cfg["backbones"] = args.backbones.split(",")
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    return apply_overrides(cfg, args.override)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", help="experiment YAML(s); run in order")
    ap.add_argument("--smoke", action="store_true", help="synthetic 2-class/50-image/1-epoch end-to-end run")
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "segment", "train", "robustness", "summary"])
    ap.add_argument("--backbones", help="comma-separated override of `backbones`")
    ap.add_argument("--seeds", help="comma-separated override of `seeds`")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=10")
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    args = ap.parse_args()
    configs = args.config or (SMOKE_CONFIGS if args.smoke else None)
    if not configs:
        ap.error("--config is required (or use --smoke)")
    for c in configs:
        run_config(build_cfg(c, args), args.stage, args.force)


if __name__ == "__main__":
    main()
