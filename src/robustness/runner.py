"""Exp 3: evaluate Exp 1 (BorB) and Exp 2 (no BorB) checkpoints on identical perturbed test images.

For every condition the perturbation is applied once to the raw test image. The no-BorB model
receives it directly; the BorB model receives ``segment(perturbed)``, i.e. segmentation runs
*after* the perturbation, exactly as it would in deployment.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.classes import PLANTDOC_TO_PV15, iter_dirs, list_images, norm_key
from src.data.io import read_rgb
from src.data.splits import load_split
from src.data.transforms import build_eval_transform
from src.eval.gradcam import background_energy, gradcam, overlay
from src.eval.stats import mcnemar
from src.models.factory import build_model, load_checkpoint
from src.robustness.perturbations import DEFAULT_SEVERITIES, Perturbation
from src.robustness.textures import TextureBank
from src.segmentation.borb import BorbParams, BorbPlusOptions, borb_plus, build_segmenter
from src.segmentation.cache import thresholds_path
from src.utils.common import get_device, get_logger, load_json, save_json, seed_everything, seed_worker
from src.utils.config import load_config
from src.viz.plots import save_image_grid, save_robustness_plot

log = get_logger(__name__)


class PairedDataset(Dataset):
    """Yields (raw-perturbed tensor, segmented-perturbed tensor, label) for one condition."""

    def __init__(self, abs_paths: list[str], labels: np.ndarray, perturb, segmenter, transform) -> None:
        self.abs_paths, self.labels = abs_paths, labels
        self.perturb, self.segmenter, self.transform = perturb, segmenter, transform

    def __len__(self) -> int:
        return len(self.abs_paths)

    def __getitem__(self, i: int):
        img = read_rgb(self.abs_paths[i])
        if self.perturb is not None:
            img = self.perturb(img, i)
        seg = self.segmenter(img)[0] if self.segmenter is not None else img
        return self.transform(image=img)["image"], self.transform(image=seg)["image"], int(self.labels[i])


class _LeafExtractor:
    """BorB+ (default options) mask used to cut leaves out for the background-shift test (picklable)."""

    def __init__(self, thresholds: dict) -> None:
        self.p, self.o = BorbParams.from_dict(thresholds), BorbPlusOptions()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return borb_plus(img, self.p, self.o)[1]


def conditions(rcfg: dict) -> list[tuple[str, object, int]]:
    """List of (family, severity, severity_idx) including ``clean``."""
    fams = rcfg.get("perturbations", DEFAULT_SEVERITIES)
    out: list[tuple[str, object, int]] = [("clean", 0, 0)]
    for fam, sevs in fams.items():
        for i, s in enumerate(sevs if sevs is not None else DEFAULT_SEVERITIES[fam]):
            out.append((fam, s, i))
    return out


def find_runs(results_dir: Path, exp: str, backbone: str, seeds: list[int]) -> dict[int, Path]:
    """Run dirs with a checkpoint for the given experiment/backbone."""
    out = {}
    for s in seeds:
        d = results_dir / exp / backbone / f"seed{s}"
        if (d / "best.pt").exists() and (d / "config.yaml").exists():
            out[s] = d
        else:
            log.warning("Missing checkpoint for %s/%s/seed%d — skipped", exp, backbone, s)
    return out


@torch.no_grad()
def _eval_models(models: dict, loader: DataLoader, device) -> tuple[dict, np.ndarray]:
    preds = {k: [] for k in models}
    ys = []
    for xr, xs, y in loader:
        xr, xs = xr.to(device), xs.to(device)
        for k, (m, uses_seg) in models.items():
            preds[k].append(m(xs if uses_seg else xr).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return {k: np.concatenate(v) for k, v in preds.items()}, np.concatenate(ys)


def plantdoc_items(root: Path, class_names: list[str]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Collect PlantDoc images (train+test folders) for classes overlapping the 15 PV classes."""
    idx = {c: i for i, c in enumerate(class_names)}
    mapping = {norm_key(k): v for k, v in PLANTDOC_TO_PV15.items()}
    paths, labels, rows = [], [], []
    for d in iter_dirs(root):
        tgt = mapping.get(norm_key(d.name))
        if tgt is None or tgt not in idx:
            continue
        imgs = list_images(d)
        paths += [str(p) for p in imgs]
        labels += [idx[tgt]] * len(imgs)
        rows.append({"plantdoc_folder": d.relative_to(root).as_posix(), "mapped_to": tgt, "n_images": len(imgs)})
    return paths, np.array(labels, dtype=int), pd.DataFrame(rows)


def run_robustness(cfg: dict) -> pd.DataFrame:
    """Evaluate every available (backbone, seed) pair of Exp 1/Exp 2 models on all conditions."""
    r = cfg["robustness"]
    results_dir = Path(cfg["paths"]["results_dir"])
    out_root = results_dir / cfg["experiment"]["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    get_logger("plant-borb", out_root / "robustness.log")
    device = get_device(cfg.get("device", "auto"))
    seed_everything(0)
    textures = TextureBank(r.get("texture_dir"))
    for k in ("soil", "grass", "clutter"):
        log.info("Background textures '%s': %s", k, textures.source(k))

    all_rows, mc_rows, cam_rows = [], [], []
    for backbone in cfg["backbones"]:
        runs_b = find_runs(results_dir, r["exp_borb"], backbone, cfg["seeds"])
        runs_n = find_runs(results_dir, r["exp_noborb"], backbone, cfg["seeds"])
        seeds = sorted(set(runs_b) & set(runs_n))
        if not seeds:
            log.warning("No paired runs for %s — skipping", backbone)
            continue
        # Both experiments must share data + split; take dataset settings from the BorB run.
        bcfg = load_config(runs_b[seeds[0]] / "config.yaml")
        ncfg = load_config(runs_n[seeds[0]] / "config.yaml")
        assert bcfg["data"]["split_dir"] == ncfg["data"]["split_dir"], "Exp 1 and Exp 2 used different splits"
        class_names = bcfg["data"]["class_names"]
        th = load_json(thresholds_path(bcfg))
        segmenter = build_segmenter(bcfg["segmentation"], th)
        tf = build_eval_transform(bcfg["model"]["img_size"])

        models = {}
        for s in seeds:
            for tag, run, uses_seg in (("borb", runs_b[s], True), ("noborb", runs_n[s], False)):
                m = build_model(backbone, len(class_names), pretrained=False, img_size=bcfg["model"]["img_size"]).to(device)
                load_checkpoint(m, str(run / "best.pt"), device)
                m.eval()
                models[(tag, s)] = (m, uses_seg)

        test = load_split(bcfg["data"]["split_dir"], "test")
        if r.get("max_test_images"):
            test = test.groupby("label", group_keys=False).apply(
                lambda g: g.sample(frac=min(1.0, r["max_test_images"] / len(test)), random_state=0))
        root = Path(bcfg["data"]["root"])
        abs_paths = [str(root / p) for p in test["path"]]
        labels = test["label"].to_numpy()
        extractor = _LeafExtractor(th)

        conds = conditions(r)
        for fam, sev, si in tqdm(conds, desc=f"robustness[{backbone}]"):
            pert = None if fam == "clean" else Perturbation(fam, sev, list(test["path"]), r.get("seed", 1234),
                                                            extractor, textures)
            ds = PairedDataset(abs_paths, labels, pert, segmenter, tf)
            loader = DataLoader(ds, batch_size=r.get("batch_size", 64), num_workers=cfg.get("num_workers", 4),
                                worker_init_fn=seed_worker)
            preds, y = _eval_models(models, loader, device)
            for (tag, s), p in preds.items():
                all_rows.append({"backbone": backbone, "seed": s,
                                 "exp": r["exp_borb"] if tag == "borb" else r["exp_noborb"],
                                 "perturbation": fam, "severity": sev, "severity_idx": si,
                                 "accuracy": float((p == y).mean()), "n": int(len(y))})
            for s in seeds:
                mc = mcnemar(y, preds[("borb", s)], preds[("noborb", s)])
                mc_rows.append({"backbone": backbone, "seed": s, "perturbation": fam, "severity": sev, **mc})
            pd.DataFrame(all_rows).to_csv(out_root / "robustness_raw.csv", index=False)
            pd.DataFrame(mc_rows).to_csv(out_root / "robustness_mcnemar.csv", index=False)

        # Cross-dataset: PlantDoc zero-shot
        pd_root = r.get("plantdoc_root")
        if pd_root and Path(pd_root).exists():
            ppaths, plabels, mapping = plantdoc_items(Path(pd_root), class_names)
            mapping.to_csv(out_root / "plantdoc_mapping_used.csv", index=False)
            if len(ppaths):
                ds = PairedDataset(ppaths, plabels, None, segmenter, tf)
                loader = DataLoader(ds, batch_size=r.get("batch_size", 64), num_workers=cfg.get("num_workers", 4))
                overlap = sorted(set(plabels.tolist()))
                with torch.no_grad():
                    for (tag, s), (m, uses_seg) in models.items():
                        logits = []
                        for xr, xs, _ in loader:
                            logits.append(m((xs if uses_seg else xr).to(device)).float().cpu().numpy())
                        lg = np.concatenate(logits)
                        restricted = np.full_like(lg, -np.inf)
                        restricted[:, overlap] = lg[:, overlap]
                        exp = r["exp_borb"] if tag == "borb" else r["exp_noborb"]
                        for sev_name, pred in (("all15", lg.argmax(1)), ("overlap13", restricted.argmax(1))):
                            all_rows.append({"backbone": backbone, "seed": s, "exp": exp, "perturbation": "plantdoc",
                                             "severity": sev_name, "severity_idx": 0 if sev_name == "all15" else 1,
                                             "accuracy": float((pred == plabels).mean()), "n": int(len(plabels))})
        else:
            log.warning("PlantDoc not found at %s — cross-dataset test skipped", pd_root)

        cam_rows += paired_gradcam(models, seeds[0], test, root, class_names, segmenter, tf, th, device,
                                   out_root / backbone, r.get("gradcam_n", 8), backbone)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "robustness_raw.csv", index=False)
    if cam_rows:
        pd.DataFrame(cam_rows).to_csv(out_root / "gradcam_background_energy.csv", index=False)
    if len(df):
        summary = summarize_robustness(df)
        summary.to_csv(out_root / "robustness_summary.csv", index=False)
        save_robustness_plot(df, out_root / "robustness_plot.png")
        save_json({"n_conditions": len(conditions(r)), "textures": {k: textures.source(k) for k in ("soil", "grass", "clutter")}},
                  out_root / "robustness_meta.json")
    return df


def summarize_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Per (exp, backbone, seed): clean acc, mean corruption accuracy (mCA), relative drop; then mean ± std."""
    rows = []
    for (exp, bb, s), g in df.groupby(["exp", "backbone", "seed"]):
        clean = g[g["perturbation"] == "clean"]["accuracy"].mean()
        corr = g[~g["perturbation"].isin(["clean", "plantdoc"])]
        mca = corr["accuracy"].mean()
        row = {"exp": exp, "backbone": bb, "seed": s, "clean": clean, "mCA": mca, "rel_drop": (clean - mca) / clean}
        for fam, gf in corr.groupby("perturbation"):
            row[f"fam:{fam}"] = gf["accuracy"].mean()
        pdc = g[(g["perturbation"] == "plantdoc") & (g["severity"] == "all15")]["accuracy"]
        if len(pdc):
            row["plantdoc_all15"] = pdc.mean()
            row["plantdoc_overlap13"] = g[(g["perturbation"] == "plantdoc") & (g["severity"] == "overlap13")]["accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gradcam(models: dict, seed: int, test: pd.DataFrame, root: Path, class_names: list[str], segmenter, tf,
                   th: dict, device, out_dir: Path, n: int, backbone: str) -> list[dict]:
    """Grad-CAM of the BorB and no-BorB models on the same raw test images, plus the share of
    CAM energy outside the leaf (BorB+ mask) over the whole test set sample."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = _LeafExtractor(th)
    mb, _ = models[("borb", seed)]
    mn, _ = models[("noborb", seed)]
    size = tf.transforms[0].height
    t2 = test.reset_index(drop=True)
    pick = t2.sort_values("path").groupby("label").head(1).index.tolist()[:n]
    sample = t2.sample(n=min(len(t2), 300), random_state=0).index.tolist()
    grid, titles, rows = [], [], []
    for i in sorted(set(pick) | set(sample)):
        raw = read_rgb(root / t2.loc[i, "path"])
        seg = segmenter(raw)[0]
        leaf = extractor(raw)
        cam_b, cb = gradcam(mb, tf(image=seg)["image"][None].to(device))
        cam_n, cn = gradcam(mn, tf(image=raw)["image"][None].to(device))
        leaf_r = cv2.resize(leaf.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        rows.append({"backbone": backbone, "seed": seed, "path": t2.loc[i, "path"],
                     "bg_energy_borb": background_energy(cam_b, leaf_r), "bg_energy_noborb": background_energy(cam_n, leaf_r)})
        if i in pick:
            raw_r = cv2.resize(raw, (size, size))
            grid.append([raw_r, overlay(raw_r, cam_n), overlay(cv2.resize(seg, (size, size)), cam_b)])
            y = int(t2.loc[i, "label"])
            titles.append(f"{class_names[y][:22]}\nnoBorB:{'ok' if cn == y else 'X'} BorB:{'ok' if cb == y else 'X'}")
    if grid:
        save_image_grid(grid, ["raw", "Grad-CAM no-BorB", "Grad-CAM BorB"], out_dir / f"gradcam_compare_seed{seed}.png",
                        titles, cell=2.2)
    return rows
