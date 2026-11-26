"""Deduplicate the unified web dataset and make group-aware stratified splits.

Reads ``data/web/manifest.csv`` (from ingest_catalog.py), drops labels with too few images,
removes byte-identical files, groups pHash near-duplicates (also across source datasets,
since several mirrors share images), and writes ``splits/web/{train,val,test}.csv``,
``labels.json`` (label -> crop/part/condition/kind) and ``dedup_stats.json``.

Usage: python scripts/prepare_web_splits.py [--min-per-class 30] [--ratios 0.8 0.1 0.1]
       [--keep-test old_splits/test.csv]   # earlier test images (and their pHash groups) stay in test
"""
from __future__ import annotations

import argparse
import sys

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.ingest_catalog import resolve  # noqa: E402
from src.data.dedup import dedup_and_group  # noqa: E402
from src.data.splits import SPLITS, check_split_integrity, group_stratified_split  # noqa: E402
from src.utils.common import save_json  # noqa: E402


def reresolve(man: pd.DataFrame, cat: dict) -> pd.DataFrame:
    """Recompute crop/part/condition/kind/label from the current catalog (rows mapped to null are dropped)."""
    by_name = {d["name"]: d for d in cat["datasets"]}
    out = []
    for r in man.to_dict("records"):
        ds = by_name.get(r["dataset"])
        if ds is None or ds["source"] == "tgz:imagenette":
            out.append(r)
            continue
        raw = ds["classes"].get(str(r["source_class"])) if isinstance(ds["classes"], dict) else None
        res = resolve(ds, raw)
        if res:
            out.append({**r, **res})
    return pd.DataFrame(out)


def merge_parts(man: pd.DataFrame) -> pd.DataFrame:
    """One label per crop + condition. Part = the single part seen, else "plant" (several / unspecified)."""
    plant = man["kind"] != "nonplant"
    parts = man[plant].groupby(["crop", "condition"])["part"].agg(lambda p: p.iloc[0] if p.nunique() == 1 else "plant")
    man = man.copy()
    man.loc[plant, "part"] = [parts[(c, d)] for c, d in zip(man.loc[plant, "crop"], man.loc[plant, "condition"])]
    man.loc[plant, "label"] = man.loc[plant, "crop"] + " | " + man.loc[plant, "part"] + " | " + man.loc[plant, "condition"]
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "web"))
    ap.add_argument("--out", default=str(ROOT / "splits" / "web"))
    ap.add_argument("--manifest", help="default: <data>/manifest.csv")
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--catalog", default=str(ROOT / "catalog" / "datasets.yaml"),
                    help="labels are re-resolved from this catalog, so mapping fixes need no re-download")
    ap.add_argument("--no-merge-parts", action="store_true",
                    help="keep one label per crop+part+condition instead of merging parts")
    ap.add_argument("--keep-test", help="CSV with a 'path' column; those images and their groups are forced into test")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    man = pd.read_csv(args.manifest or data / "manifest.csv")
    man["abs_path"] = [str(data / p) for p in man["path"]]
    man = man[[Path(p).exists() for p in man["abs_path"]]]
    man = reresolve(man, yaml.safe_load(open(args.catalog, encoding="utf-8")))
    if not args.no_merge_parts:
        man = merge_parts(man)

    counts = man["label"].value_counts()
    dropped = counts[counts < args.min_per_class]
    man = man[man["label"].isin(counts[counts >= args.min_per_class].index)].copy()
    labels = sorted(man["label"].unique(), key=lambda s: (s == "Not a plant", s))
    lid = {l: i for i, l in enumerate(labels)}
    man["label_name"] = man["label"]
    man["label"] = man["label_name"].map(lid)
    man["class_name"] = man["label_name"]

    man, stats = dedup_and_group(man, max_dist=4, workers=8, hash_cache=data / "hash_cache.csv")
    man["split"] = group_stratified_split(man, tuple(args.ratios), args.seed)
    if args.keep_test:
        keep = set(pd.read_csv(args.keep_test)["path"])
        pinned = man["group"].isin(man.loc[man["path"].isin(keep), "group"])
        moved = int((pinned & (man["split"] != "test")).sum())
        man.loc[pinned, "split"] = "test"
        stats["kept_test"] = {"listed": len(keep), "found": int(man["path"].isin(keep).sum()),
                              "moved_into_test": moved}

    out.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "class_name", "group", "phash", "md5", "dataset", "source_class", "crop", "part",
            "condition", "kind", "field"]
    for s in SPLITS:
        man[man["split"] == s][cols].to_csv(out / f"{s}.csv", index=False)
    meta = (man.drop_duplicates("label_name").set_index("label_name")[["crop", "part", "condition", "kind"]]
            .reindex(labels))
    save_json({l: {"id": lid[l], **meta.loc[l].to_dict()} for l in labels}, out / "labels.json")
    stats.update({"classes": labels, "split_sizes": {s: int((man["split"] == s).sum()) for s in SPLITS},
                  "dropped_labels_too_few": {k: int(v) for k, v in dropped.items()},
                  "n_crops": int(man["crop"].nunique()), "n_datasets": int(man["dataset"].nunique())})
    save_json(stats, out / "dedup_stats.json")
    man.groupby(["crop", "label_name", "split"]).size().unstack(fill_value=0).to_csv(out / "class_counts.csv")
    integ = check_split_integrity(out)
    assert not any(integ.values()), integ
    print({k: v for k, v in stats.items() if k not in ("classes",)})


if __name__ == "__main__":
    main()
