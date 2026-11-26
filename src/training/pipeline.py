"""One training run (backbone x seed): train, select by val, evaluate on test, save artifacts."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.dataset import LeafDataset, make_loader
from src.data.io import read_rgb
from src.data.splits import class_weights, load_split, split_fingerprint
from src.data.transforms import build_transforms
from src.eval.gradcam import gradcam, overlay
from src.eval.metrics import compute_metrics, predict, text_report
from src.eval.profile import profile_model
from src.models.factory import build_model, init_from_checkpoint, load_checkpoint
from src.training.trainer import Trainer, resolve_amp
from src.utils.common import get_device, get_logger, save_json, seed_everything
from src.utils.config import save_config
from src.viz.plots import save_confusion, save_curves, save_image_grid

log = get_logger(__name__)


def build_datasets(cfg: dict, cache_dir: Path | None, segmenter=None) -> dict[str, LeafDataset]:
    """Train/val/test datasets sharing the split CSVs; segmented images come from the cache."""
    train_tf, eval_tf = build_transforms(cfg)
    root = cfg["data"]["root"]
    sd = cfg["data"]["split_dir"]
    out = {}
    for s in ("train", "val", "test"):
        df = load_split(sd, s)
        out[s] = LeafDataset(df, root, train_tf if s == "train" else eval_tf, cache_dir=cache_dir,
                             segmenter=segmenter if cache_dir is None else None)
    return out


def gradcam_examples(test_df: pd.DataFrame, n: int) -> list[int]:
    """Deterministic example indices: first test image of each class (by path), up to ``n``."""
    first = test_df.reset_index().sort_values("path").groupby("label").head(1)
    return first["index"].tolist()[:n]


def save_gradcam_grid(model, ds: LeafDataset, idxs: list[int], class_names: list[str], device, path: Path,
                      size: int) -> None:
    """Rows: input as seen by the model, Grad-CAM overlay on it, Grad-CAM overlay on the raw image."""
    rows, titles = [], []
    for i in idxs:
        inp = ds.load(i)
        raw = read_rgb(ds.root / ds.paths[i])
        cam, cls = gradcam(model, ds.transform(image=inp)["image"][None].to(device))
        inp_r, raw_r = cv2.resize(inp, (size, size)), cv2.resize(raw, (size, size))
        rows.append([inp_r, overlay(inp_r, cam), overlay(raw_r, cam)])
        titles.append(f"true {class_names[ds.labels[i]][:24]}\npred {class_names[cls][:24]}")
    if rows:
        save_image_grid(rows, ["model input", "Grad-CAM (input)", "Grad-CAM (raw)"], path, titles, cell=2.2)


def run_single(cfg: dict, backbone: str, seed: int, out_dir: Path, cache_dir: Path | None,
               segmenter=None, evaluate_test: bool = True) -> dict:
    """Train + evaluate one run; writes everything under ``out_dir``. Returns metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_log = get_logger("plant-borb")
    for h in [h for h in root_log.handlers if isinstance(h, logging.FileHandler)]:
        root_log.removeHandler(h)  # one log file per run
        h.close()
    get_logger("plant-borb", out_dir / "train.log")
    seed_everything(seed, cfg["train"].get("deterministic", True))
    device = get_device(cfg.get("device", "auto"))
    cfg = {**cfg, "model": {**cfg["model"], "backbone": backbone}, "seed": seed}
    save_config(cfg, out_dir / "config.yaml")
    class_names: list[str] = cfg["data"]["class_names"]
    n_classes = len(class_names)

    ds = build_datasets(cfg, cache_dir, segmenter)
    t = cfg["train"]
    nw = cfg.get("num_workers", 4)
    train_loader = make_loader(ds["train"], t["batch_size"], True, nw, seed, t.get("balanced_sampling", False))
    val_loader = make_loader(ds["val"], t.get("eval_batch_size", 64), False, nw, seed)
    test_loader = make_loader(ds["test"], t.get("eval_batch_size", 64), False, nw, seed)

    model = build_model(backbone, n_classes, cfg["model"].get("pretrained", True), cfg["model"].get("drop_rate", 0.2),
                        cfg["model"]["img_size"])
    if cfg["model"].get("init_from"):
        init_from_checkpoint(model, cfg["model"]["init_from"])
    cw = class_weights(load_split(cfg["data"]["split_dir"], "train"), n_classes) if t.get("weighted_loss") else None
    trainer = Trainer(model, cfg, device, out_dir, cw)
    res = trainer.fit(train_loader, val_loader)
    save_curves(res.history, out_dir / "curves.png")

    meta = load_checkpoint(model, str(res.ckpt_path), device)
    amp = resolve_amp(t.get("amp", "auto"), device)
    metrics: dict = {
        "experiment": cfg["experiment"]["name"], "backbone": backbone, "seed": seed,
        "group": cfg["experiment"].get("group"), "ablation": cfg["experiment"].get("ablation"),
        "segmentation": cfg["segmentation"].get("method", "none"),
        "segmentation_output": cfg["segmentation"].get("output", "black"),
        "split_fingerprint": split_fingerprint(cfg["data"]["split_dir"]),
        "dataset": cfg["data"]["name"], "n_classes": n_classes,
        "train": {"best_epoch": res.best_epoch, "epochs_run": res.epochs_run, "best_val_acc": res.best_val_acc,
                  "best_val_loss": res.best_val_loss, "train_seconds": res.train_seconds, "ckpt_meta": meta},
    }
    vp, vy = predict(model, val_loader, device, amp=amp)
    metrics["val"] = {k: v for k, v in compute_metrics(vy, vp, class_names).items() if k != "report"}
    if evaluate_test:
        tp, ty = predict(model, test_loader, device, amp=amp)
        m = compute_metrics(ty, tp, class_names)
        metrics["test"] = m
        if cfg.get("eval", {}).get("tta", False):
            ttp, _ = predict(model, test_loader, device, tta=True, amp=amp)
            metrics["test_tta"] = {k: v for k, v in compute_metrics(ty, ttp, class_names).items() if k != "report"}
        (out_dir / "classification_report.txt").write_text(text_report(ty, tp, class_names), encoding="utf-8")
        save_confusion(np.array(m["confusion_matrix"]), class_names, out_dir / "cm_abs.png", normalize=False)
        save_confusion(np.array(m["confusion_matrix"]), class_names, out_dir / "cm_norm.png", normalize=True)
        test_df = load_split(cfg["data"]["split_dir"], "test")
        pd.DataFrame({"path": test_df["path"], "label": ty, "pred": tp.argmax(1), "conf": tp.max(1)}).to_csv(
            out_dir / "predictions.csv", index=False)
        np.save(out_dir / "test_probs.npy", tp.astype(np.float16))
        try:
            save_gradcam_grid(model, ds["test"], gradcam_examples(test_df, cfg.get("eval", {}).get("gradcam_n", 8)),
                              class_names, device, out_dir / "gradcam.png", cfg["model"]["img_size"])
        except Exception as e:  # Grad-CAM is diagnostic; never fail a run because of it
            log.warning("Grad-CAM failed: %s", e)
    metrics["profile"] = profile_model(model, cfg["model"]["img_size"], device)
    save_json(metrics, out_dir / "metrics.json")
    if not cfg.get("paths", {}).get("keep_checkpoints", True):
        res.ckpt_path.unlink(missing_ok=True)
    log.info("[%s/%s/seed%d] val acc %.4f%s", cfg["experiment"]["name"], backbone, seed, metrics["val"]["accuracy"],
             f" | test acc {metrics['test']['accuracy']:.4f} CI95 {metrics['test']['accuracy_ci95']}" if evaluate_test else "")
    return metrics
