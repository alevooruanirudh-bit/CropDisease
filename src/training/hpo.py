"""Random-search hyperparameter tuning on the VALIDATION split only.

Candidates are drawn from a seeded RNG, so Exp 1 and Exp 2 try exactly the same
configurations with the same budget. The test split is never loaded here.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_json, save_json
from src.utils.config import apply_overrides

log = get_logger(__name__)


def sample_candidates(space: dict, n: int, seed: int) -> list[dict]:
    """Draw ``n`` candidates. Space entries: ``{type: loguniform|uniform|choice, low, high | values}``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = {}
        for key, spec in space.items():
            t = spec["type"]
            if t == "loguniform":
                c[key] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
            elif t == "uniform":
                c[key] = float(rng.uniform(spec["low"], spec["high"]))
            elif t == "choice":
                c[key] = spec["values"][int(rng.integers(0, len(spec["values"])))]
            else:
                raise ValueError(f"Unknown search type {t}")
        out.append(c)
    return out


def random_search(cfg: dict, backbone: str, out_dir: Path, cache_dir, segmenter=None) -> dict:
    """Run the search; returns and saves the best overrides (``hpo.json``). Resumable."""
    from src.training.pipeline import run_single

    h = cfg["hpo"]
    out_dir = Path(out_dir)
    best_file = out_dir / "hpo.json"
    if best_file.exists():
        return load_json(best_file)
    cands = sample_candidates(h["space"], h["n_trials"], h.get("seed", 0))
    rows = []
    for i, cand in enumerate(cands):
        tdir = out_dir / f"trial{i:02d}"
        trial_cfg = apply_overrides(copy.deepcopy(cfg), [f"{k}={v}" for k, v in cand.items()])
        trial_cfg["train"]["epochs"] = h.get("epochs", trial_cfg["train"]["epochs"])
        trial_cfg["paths"]["keep_checkpoints"] = False
        m = run_single(trial_cfg, backbone, h.get("trial_seed", 0), tdir, cache_dir, segmenter, evaluate_test=False)
        rows.append({**cand, "val_acc": m["val"]["accuracy"], "val_loss": m["train"]["best_val_loss"]})
        pd.DataFrame(rows).to_csv(out_dir / "hpo_trials.csv", index=False)
    df = pd.DataFrame(rows).sort_values(["val_acc", "val_loss"], ascending=[False, True])
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[0].items() if k in h["space"]}
    result = {"best": best, "best_val_acc": float(df.iloc[0]["val_acc"]), "n_trials": len(rows)}
    save_json(result, best_file)
    log.info("HPO best for %s: %s (val acc %.4f)", backbone, best, result["best_val_acc"])
    return result
