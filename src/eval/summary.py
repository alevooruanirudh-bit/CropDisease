"""Aggregate all runs into results/summary.{csv,md} and results/ablation.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import compute_metrics
from src.eval.stats import mcnemar, paired_bootstrap_diff
from src.utils.common import get_logger, load_json

log = get_logger(__name__)

# Numbers reported by Ozcan & Polat (2025) that the user asked us to compare against.
PAPER_REPORTED = {
    "plantvillage15": {"vgg16": 0.9978},
    "mangoleafbd": {"_best": 1.0},
}


def _pm(x: pd.Series, pct: bool = True, digits: int = 2) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "–"
    k = 100 if pct else 1
    if len(x) == 1:
        return f"{k * x.iloc[0]:.{digits}f} (n=1)"
    return f"{k * x.mean():.{digits}f} ± {k * x.std(ddof=1):.{digits}f} (n={len(x)})"


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """One row per finished run (``results/<exp>/<backbone>/seed*/metrics.json``)."""
    rows = []
    for f in sorted(results_dir.glob("*/*/seed*/metrics.json")):
        m = load_json(f)
        t = m.get("test", {})
        prof = m.get("profile", {})
        rows.append({
            "exp": m["experiment"], "dataset": m.get("dataset"), "backbone": m["backbone"], "seed": m["seed"],
            "group": m.get("group"), "ablation_axis": (m.get("ablation") or {}).get("axis"),
            "ablation_value": (m.get("ablation") or {}).get("value"),
            "segmentation": m["segmentation"], "seg_output": m.get("segmentation_output"),
            "val_acc": m["val"]["accuracy"], "test_acc": t.get("accuracy"),
            "test_ci_lo": (t.get("accuracy_ci95") or [None, None])[0],
            "test_ci_hi": (t.get("accuracy_ci95") or [None, None])[1],
            "test_f1_macro": t.get("f1_macro"), "test_f1_weighted": t.get("f1_weighted"),
            "test_precision_macro": t.get("precision_macro"), "test_recall_macro": t.get("recall_macro"),
            "test_kappa": t.get("cohen_kappa"), "test_top5": t.get("top5"),
            "test_tta_acc": (m.get("test_tta") or {}).get("accuracy"),
            "params_m": prof.get("params_m"), "gflops": prof.get("gflops"),
            "latency_ms": prof.get("latency_ms_per_img"), "latency_device": prof.get("latency_device"),
            "best_epoch": m["train"]["best_epoch"], "epochs_run": m["train"]["epochs_run"],
            "train_minutes": m["train"]["train_seconds"] / 60, "split_fingerprint": m.get("split_fingerprint"),
            "run_dir": str(f.parent),
        })
    return pd.DataFrame(rows)


def paired_tests(runs: pd.DataFrame, exp_a: str, exp_b: str) -> pd.DataFrame:
    """McNemar + paired bootstrap per (backbone, seed) on identical test images."""
    out = []
    a = runs[runs["exp"] == exp_a]
    b = runs[runs["exp"] == exp_b]
    for _, ra in a.iterrows():
        rb = b[(b["backbone"] == ra["backbone"]) & (b["seed"] == ra["seed"])]
        if rb.empty:
            continue
        rb = rb.iloc[0]
        if ra["split_fingerprint"] != rb["split_fingerprint"]:
            log.error("Split mismatch between %s and %s for %s seed %s — test skipped", exp_a, exp_b,
                      ra["backbone"], ra["seed"])
            continue
        pa = pd.read_csv(Path(ra["run_dir"]) / "predictions.csv")
        pb = pd.read_csv(Path(rb["run_dir"]) / "predictions.csv")
        mg = pa.merge(pb, on="path", suffixes=("_a", "_b"))
        assert len(mg) == len(pa) == len(pb), "prediction files do not cover the same images"
        y = mg["label_a"].to_numpy()
        mc = mcnemar(y, mg["pred_a"].to_numpy(), mg["pred_b"].to_numpy())
        d, lo, hi = paired_bootstrap_diff(y, mg["pred_a"].to_numpy(), mg["pred_b"].to_numpy())
        out.append({"backbone": ra["backbone"], "seed": ra["seed"], "acc_a": ra["test_acc"], "acc_b": rb["test_acc"],
                    "diff": d, "diff_ci_lo": lo, "diff_ci_hi": hi, **mc})
    return pd.DataFrame(out)


def ensemble_table(runs: pd.DataFrame, exp: str, top_k: tuple[int, ...] = (2, 3)) -> pd.DataFrame:
    """Top-k backbone ensembles per seed. Backbones are ranked by VALIDATION accuracy (never test)."""
    out = []
    r = runs[(runs["exp"] == exp)]
    for seed, g in r.groupby("seed"):
        g = g.sort_values("val_acc", ascending=False)
        for k in top_k:
            if len(g) < k:
                continue
            sel = g.head(k)
            probs = [np.load(Path(d) / "test_probs.npy").astype(np.float32) for d in sel["run_dir"]]
            preds = pd.read_csv(Path(sel.iloc[0]["run_dir"]) / "predictions.csv")
            p = np.mean(probs, 0)
            m = compute_metrics(preds["label"].to_numpy(), p, [str(i) for i in range(p.shape[1])])
            out.append({"exp": exp, "seed": seed, "k": k, "members": "+".join(sel["backbone"]),
                        "test_acc": m["accuracy"], "test_f1_macro": m["f1_macro"]})
    return pd.DataFrame(out)


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_no runs found_"


def write_summary(results_dir: str | Path, cfg: dict) -> None:
    """Write summary.csv, summary.md and ablation.md from whatever runs exist."""
    results_dir = Path(results_dir)
    runs = collect_runs(results_dir)
    if runs.empty:
        log.warning("No finished runs under %s", results_dir)
        return
    runs.to_csv(results_dir / "summary.csv", index=False)
    s = cfg.get("summary", {})
    md: list[str] = ["# Results summary", "",
                     "All numbers are test-set results of checkpoints selected by **validation** accuracy. "
                     "Mean ± sample std over seeds. Accuracy in %.", ""]

    # --- data integrity
    md += ["## Data integrity (pHash dedup, group-aware 60/20/20 split)", ""]
    for f in sorted(Path(cfg["paths"].get("split_root", "splits")).glob("*/dedup_stats.json")):
        st = load_json(f)
        md.append(f"**{f.parent.name}**: input {st['n_input']}, exact byte duplicates removed "
                  f"{st['exact_byte_duplicates_removed']}, identical-pHash (grouped) {st['identical_phash_within_class']}, "
                  f"kept {st['n_kept']}; near-duplicate pairs (≤{cfg['data'].get('dedup_max_dist', 4)} bits) "
                  f"{st['near_duplicate_pairs']} → {st['n_multi_groups']} multi-image groups covering "
                  f"{st['n_images_in_multi_groups']} images (largest {st['largest_group']}); cross-class "
                  f"near-duplicate pairs {st['cross_class_near_duplicate_pairs']}; split sizes {st['split_sizes']}.")
        md.append("")

    # --- main comparison
    for pair in s.get("pairs", []):
        ea, eb, key = pair["exp_a"], pair["exp_b"], pair.get("paper_key")
        sub = runs[runs["exp"].isin([ea, eb])]
        if sub.empty:
            continue
        md += [f"## {ea} vs {eb}", ""]
        rows = []
        for bb in sorted(sub["backbone"].unique()):
            ga = sub[(sub["exp"] == ea) & (sub["backbone"] == bb)]
            gb = sub[(sub["exp"] == eb) & (sub["backbone"] == bb)]
            paper = PAPER_REPORTED.get(key, {}).get(bb)
            d = (ga["test_acc"].mean() - gb["test_acc"].mean()) * 100 if len(ga) and len(gb) else np.nan
            rows.append({"backbone": bb, "paper (reported)": f"{100 * paper:.2f}" if paper else "–",
                         f"{ea} test acc": _pm(ga["test_acc"]), f"{eb} test acc": _pm(gb["test_acc"]),
                         "Δ (pp)": f"{d:+.2f}" if not np.isnan(d) else "–",
                         f"{ea} macro-F1": _pm(ga["test_f1_macro"]), f"{eb} macro-F1": _pm(gb["test_f1_macro"]),
                         f"{ea} κ": _pm(ga["test_kappa"], pct=False, digits=4),
                         f"{ea} TTA": _pm(ga["test_tta_acc"])})
        md += [_md(pd.DataFrame(rows)), ""]
        if key in PAPER_REPORTED and "_best" in PAPER_REPORTED[key]:
            md += [f"Paper best reported on this dataset: {100 * PAPER_REPORTED[key]['_best']:.2f}%.", ""]
        pt = paired_tests(runs, ea, eb)
        if len(pt):
            pt.to_csv(results_dir / f"mcnemar_{ea}_vs_{eb}.csv", index=False)
            show = pt.assign(acc_a=lambda x: (100 * x.acc_a).round(2), acc_b=lambda x: (100 * x.acc_b).round(2),
                             diff=lambda x: (100 * x["diff"]).round(2),
                             diff_ci=lambda x: [f"[{100 * a:+.2f}, {100 * b:+.2f}]" for a, b in zip(x.diff_ci_lo, x.diff_ci_hi)],
                             p_value=lambda x: x.p_value.map(lambda v: f"{v:.2e}"))
            md += ["**Paired tests on identical test images** (b = only A correct, c = only B correct):", "",
                   _md(show[["backbone", "seed", "acc_a", "acc_b", "diff", "diff_ci", "b", "c", "p_value", "method"]]), ""]
        ens = ensemble_table(runs, ea)
        if len(ens):
            md += [f"**Backbone ensembles for {ea}** (members chosen by val accuracy; reported separately from "
                   "single models):", "", _md(ens.assign(test_acc=lambda x: (100 * x.test_acc).round(2))), ""]

    # --- segmentation comparison
    seg = runs[runs["exp"].isin(s.get("segcomp_exps", []))]
    if len(seg):
        md += ["## Segmentation method comparison (same backbone/recipe; paper Table 11 style)", ""]
        q = {}
        for f in Path(cfg["paths"]["cache_dir"]).glob("*/*/mask_quality.json"):
            q[f.parent.name] = load_json(f)["failure_rate"]
        rows = []
        for (exp, meth, bb), g in seg.groupby(["exp", "segmentation", "backbone"]):
            out_mode = g["seg_output"].iloc[0]
            fr = [v for k, v in q.items() if k.startswith(f"{meth}_{out_mode}_")] if meth != "none" else []
            rows.append({"experiment": exp, "method": meth, "backbone": bb, "val acc": _pm(g["val_acc"]),
                         "test acc": _pm(g["test_acc"]), "macro-F1": _pm(g["test_f1_macro"]),
                         "mask failure rate": f"{100 * fr[0]:.2f}%" if fr else "–"})
        md += [_md(pd.DataFrame(rows)), ""]

    # --- robustness
    for f in sorted(results_dir.glob("*/robustness_summary.csv")):
        rs = pd.read_csv(f)
        md += [f"## Robustness ({f.parent.name})", "",
               "Perturbations are applied to the raw test image; the BorB model segments the perturbed image. "
               "mCA = mean accuracy over all corruption conditions; rel. drop = (clean − mCA)/clean.", ""]
        cols = [c for c in rs.columns if c not in ("exp", "backbone", "seed")]
        agg = rs.groupby(["backbone", "exp"])[cols].agg(lambda x: _pm(x, pct=True)).reset_index()
        agg.columns = [c.replace("fam:", "") for c in agg.columns]
        md += [_md(agg), ""]
        mcf = f.parent / "robustness_mcnemar.csv"
        if mcf.exists():
            mc = pd.read_csv(mcf)
            mc["borb_better"] = mc["b"] > mc["c"]
            g = mc.groupby(["backbone", "perturbation", "severity"]).agg(
                seeds=("seed", "count"), borb_better=("borb_better", "sum"),
                sig=("p_value", lambda x: int((x < 0.05).sum())), median_p=("p_value", "median")).reset_index()
            g["median_p"] = g["median_p"].map(lambda v: f"{v:.1e}")
            md += ["McNemar per condition (seeds where BorB > no-BorB / seeds with p < 0.05):", "", _md(g), ""]
        cam = f.parent / "gradcam_background_energy.csv"
        if cam.exists():
            c = pd.read_csv(cam)
            ce = c.groupby("backbone")[["bg_energy_borb", "bg_energy_noborb"]].mean().reset_index()
            md += ["Share of Grad-CAM energy outside the leaf mask (lower = attends to the leaf):", "",
                   _md(ce.round(4)), ""]
        if (f.parent / "robustness_plot.png").exists():
            md += [f"![robustness]({f.parent.name}/robustness_plot.png)", ""]

    # --- efficiency
    eff = runs.groupby("backbone")[["params_m", "gflops", "latency_ms"]].mean().round(2).reset_index()
    md += ["## Model cost", "", f"Latency: batch 1, device `{runs['latency_device'].iloc[0]}`.", "", _md(eff), ""]

    # --- all runs
    agg = runs.groupby(["exp", "dataset", "backbone"]).agg(
        seeds=("seed", "count"), val_acc=("val_acc", _pm), test_acc=("test_acc", _pm),
        f1_macro=("test_f1_macro", _pm), epochs=("epochs_run", "mean"), train_min=("train_minutes", "mean")).reset_index()
    md += ["## All experiments", "", _md(agg.round(1)), ""]
    (results_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # --- ablations
    ab = runs[runs["ablation_axis"].notna()]
    ref = s.get("ablation_reference", {})
    ref_runs = runs[runs["exp"] == ref.get("exp")]
    amd = ["# Ablations", "",
           "Each axis changes one factor relative to the Exp 1 recipe. Choose settings by **val** accuracy; "
           "test accuracy is shown for reporting only.", ""]
    for axis, g in ab.groupby("ablation_axis"):
        rows = []
        for bb, gg in ref_runs.groupby("backbone"):
            rows.append({"setting": ref.get("labels", {}).get(axis, "reference"), "backbone": bb,
                         "experiment": ref["exp"], "val acc": _pm(gg["val_acc"]), "test acc": _pm(gg["test_acc"]),
                         "macro-F1": _pm(gg["test_f1_macro"])})
        for (val, bb), gg in g.groupby(["ablation_value", "backbone"]):
            rows.append({"setting": val, "backbone": bb, "experiment": gg["exp"].iloc[0], "val acc": _pm(gg["val_acc"]),
                         "test acc": _pm(gg["test_acc"]), "macro-F1": _pm(gg["test_f1_macro"])})
        amd += [f"## {axis}", "", _md(pd.DataFrame(rows)), ""]
    if ab.empty:
        amd.append("_No ablation runs found yet. See README → Ablations._")
    (results_dir / "ablation.md").write_text("\n".join(amd), encoding="utf-8")
    log.info("Wrote %s, summary.md, ablation.md", results_dir / "summary.csv")
