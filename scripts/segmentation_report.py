"""Segmentation QC before any training.

* Estimates BorB thresholds + polarity on the TRAINING split and asserts the expected polarity
  (leaf = low RGB-B, high Lab-b) for PlantVillage.
* Fig. 8/9-style grid: original | B mask | b mask | OR mask | cleaned mask | segmented.
* Method comparison grid: original | BorB-original | BorB+ | Otsu | K-means.
* Mask sanity metric per method (share of masks covering <5% or >95% of the image), with the
  failing files logged to CSV.

Usage: python scripts/segmentation_report.py --config configs/exp1_borb.yaml [--max-images N]
Outputs: results/segmentation/<dataset>/
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from run import prepare  # noqa: E402
from src.data.io import read_rgb  # noqa: E402
from src.data.splits import SPLITS, load_split  # noqa: E402
from src.segmentation.baselines import kmeans_segment, otsu_segment  # noqa: E402
from src.segmentation.borb import (BorbParams, BorbPlusOptions, apply_mask, borb_original,  # noqa: E402
                                   borb_plus_mask)
from src.segmentation.cache import HIGH_COVER, LOW_COVER, ensure_thresholds  # noqa: E402
from src.utils.common import get_logger, save_json  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402
from src.viz.plots import save_image_grid  # noqa: E402

log = get_logger("seg-report")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp1_borb.yaml")
    ap.add_argument("--max-images", type=int, default=None, help="subsample for the failure-rate scan")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), help="splits to scan (QC is label-free)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["segmentation"]["method"] = "borb_plus"
    cfg = prepare(cfg)
    th = ensure_thresholds(cfg)
    out = Path(cfg["paths"]["results_dir"]) / "segmentation" / (cfg["data"]["name"] + args.out_suffix)
    out.mkdir(parents=True, exist_ok=True)

    pol = th["polarity_votes"]
    log.info("Polarity from data (train split, n=%d): leaf has lower B than border in %.1f%%, higher b in %.1f%%",
             pol["n"], 100 * pol["vote_B_low"], 100 * pol["vote_b_high"])
    if cfg["data"]["dataset"] == "plantvillage":
        assert th["pol_B"] == "low" and th["pol_b"] == "high", f"Unexpected BorB polarity on PlantVillage: {pol}"
        log.info("Asserted: PlantVillage leaf is low-blue (RGB B) and high-b (Lab b).")

    p_orig = BorbParams(pol_B=th["pol_B"], pol_b=th["pol_b"])
    p_plus = BorbParams.from_dict(th)
    seg = cfg["segmentation"]
    o = BorbPlusOptions(**{k: v for k, v in seg.items() if k in BorbPlusOptions.__dataclass_fields__})

    # ---- figures on TRAINING images (one per class)
    train = load_split(cfg["data"]["split_dir"], "train")
    root = Path(cfg["data"]["root"])
    pick = train.sort_values("path").groupby("label").head(1)
    rows_steps, rows_methods, titles = [], [], []
    for _, r in pick.iterrows():
        img = read_rgb(root / r["path"])
        m = borb_plus_mask(img, p_plus, o)
        rows_steps.append([img, m["B"], m["b"], m["or"], m["clean"], apply_mask(img, m["clean"])])
        rows_methods.append([img, borb_original(img, p_orig)[0], apply_mask(img, m["clean"]),
                             otsu_segment(img)[0], kmeans_segment(img)[0]])
        titles.append(r["class_name"].replace("___", "\n")[:40])
    save_image_grid(rows_steps, ["original", f"B<{p_plus.t_B}", f"b>{p_plus.t_b}", "B OR b", "cleaned (BorB+)",
                                 "segmented"], out / "fig_borb_steps.png", titles)
    save_image_grid(rows_methods, ["original", "BorB-original (130/135)", "BorB+", "Otsu", "K-means"],
                    out / "fig_methods.png", titles)

    # ---- failure-rate scan on all splits (label-free QC, nothing is tuned on it)
    df = pd.concat([load_split(cfg["data"]["split_dir"], s).assign(split=s) for s in args.splits], ignore_index=True)
    if args.max_images and len(df) > args.max_images:
        df = df.sample(n=args.max_images, random_state=0)
    methods = {
        "borb_original": lambda im: borb_original(im, p_orig)[1],
        "borb_plus": lambda im: borb_plus_mask(im, p_plus, o)["clean"],
        "borb_plus_no_chroma": lambda im: borb_plus_mask(im, p_plus, BorbPlusOptions(bg_chroma_protect=False))["clean"],
        "borb_plus_nomorph": lambda im: borb_plus_mask(im, p_plus, BorbPlusOptions(morphology=False, fill_holes=False,
                                                                                    lesion_protect=False))["clean"],
        "otsu": lambda im: otsu_segment(im)[1],
        "kmeans": lambda im: kmeans_segment(im)[1],
    }

    def work(rel: str) -> dict:
        im = read_rgb(root / rel)
        return {k: float(f(im).mean()) for k, f in methods.items()}

    with ThreadPoolExecutor(cfg["data"].get("hash_workers", 8)) as ex:
        cov = list(tqdm(ex.map(work, df["path"]), total=len(df), desc="mask QC"))
    cov = pd.DataFrame(cov, index=df.index)
    full = pd.concat([df[["path", "split", "class_name"]], cov], axis=1)
    full.to_csv(out / "mask_coverage.csv", index=False)
    summary, fails = [], []
    for k in methods:
        c = full[k]
        bad = (c < LOW_COVER) | (c > HIGH_COVER)
        summary.append({"method": k, "n": len(c), "fail_<5%": int((c < LOW_COVER).sum()),
                        "fail_>95%": int((c > HIGH_COVER).sum()), "failure_rate_%": round(100 * bad.mean(), 3),
                        "coverage_mean_%": round(100 * c.mean(), 2), "coverage_median_%": round(100 * c.median(), 2)})
        fails.append(full[bad][["path", "split", "class_name", k]].rename(columns={k: "coverage"}).assign(method=k))
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "mask_quality_by_method.csv", index=False)
    pd.concat(fails).to_csv(out / "mask_failures.csv", index=False)
    per_class = full.groupby("class_name")[list(methods)].agg(
        lambda c: round(100 * ((c < LOW_COVER) | (c > HIGH_COVER)).mean(), 2))
    per_class.to_csv(out / "mask_failure_rate_per_class.csv")
    save_json({"thresholds": {k: th[k] for k in ("t_B", "t_b", "method_B", "method_b", "pol_B", "pol_b")},
               "polarity_votes": pol, "summary": summary}, out / "segmentation_report.json")
    print("\nThresholds (train split): t_B=%s (%s), t_b=%s (%s); paper 130/135" %
          (th["t_B"], th["method_B"], th["t_b"], th["method_b"]))
    print(sm.to_markdown(index=False))
    print("\nPer-class failure rate (%):\n" + per_class.to_markdown())
    print(f"\nFigures: {out}/fig_borb_steps.png, {out}/fig_methods.png")


if __name__ == "__main__":
    main()
