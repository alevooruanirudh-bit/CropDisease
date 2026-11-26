"""Plant health website: upload or photograph a plant part, get plant + problem + treatment advice.

    python webapp/server.py --run results/web_model/efficientnet_b0/seed0 [--host 0.0.0.0] [--port 8000]

Open http://localhost:8000 . Use --host 0.0.0.0 to open it from a phone on the same Wi-Fi
(http://<this-computer's-IP>:8000); the "Take photo" button opens the phone camera.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

from src.data.transforms import build_eval_transform  # noqa: E402
from src.kb import advice, parse_label  # noqa: E402
from src.models.factory import build_model, load_checkpoint  # noqa: E402
from src.utils.common import get_device, load_json  # noqa: E402
from src.utils.config import load_config  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 15 * 1024 * 1024
CONFIDENT, POSSIBLE = 0.55, 0.30
POSSIBLE_PROBLEM = 0.25  # an unsure answer still shows advice when one problem has this much in total
# "Healthy" is only the answer when all healthy labels together reach this; otherwise the most likely problem is
# shown. Chosen on validation: halves sick plants called healthy (3.1% -> 1.7%) at the same accuracy.
HEALTHY_MIN = 0.5
ZOOMS = (0.7, 0.5)


class Predictor:
    """Loads a trained run directory (config.yaml + best.pt) and classifies RGB images."""

    def __init__(self, run_dir: Path, device: str = "auto") -> None:
        cfg = load_config(run_dir / "config.yaml")
        self.labels: list[str] = cfg["data"]["class_names"]
        self.img_size = cfg["model"]["img_size"]
        self.device = get_device(device)
        self.model = build_model(cfg["model"]["backbone"], len(self.labels), pretrained=False,
                                 img_size=self.img_size).to(self.device).eval()
        self.meta = load_checkpoint(self.model, str(run_dir / "best.pt"), self.device)
        self.tf = build_eval_transform(self.img_size)
        self.backbone = cfg["model"]["backbone"]
        metrics = run_dir / "metrics.json"
        self.metrics = load_json(metrics) if metrics.exists() else {}

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> np.ndarray:
        # Whole photo plus two centre zooms (70%, 50%), each also mirrored: small spots and pests that are lost
        # when a wide shot is shrunk to 224 px show up in the zooms. +2 points on validation field photos.
        w, h = img.size
        views = [img] + [img.crop(((w - w * f) / 2, (h - h * f) / 2, (w + w * f) / 2, (h + h * f) / 2))
                         for f in ZOOMS]
        x = torch.stack([self.tf(image=np.array(v))["image"] for v in views]).to(self.device)
        p = self.model(x).float().softmax(1) + self.model(x.flip(3)).float().softmax(1)
        return (p / 2).mean(0).cpu().numpy()


def summarise(probs: np.ndarray, labels: list[str], top_k: int = 5) -> dict:
    """Turn class probabilities into the website's answer."""
    order = np.argsort(-probs)
    healthy = np.array([parse_label(l)[2] == "Healthy" for l in labels])
    if probs[healthy].sum() < HEALTHY_MIN:  # not clearly healthy: every problem ranks ahead of every healthy label
        order = np.concatenate([order[~healthy[order]], order[healthy[order]]])
    top = [{"label": labels[i], "prob": float(probs[i]), **dict(zip(("crop", "part", "condition"),
                                                                     parse_label(labels[i])))}
           for i in order[:top_k]]
    crop_p: dict[str, float] = defaultdict(float)
    for i, l in enumerate(labels):
        crop_p[parse_label(l)[0]] += float(probs[i])
    plants = sorted(({"crop": c, "prob": p} for c, p in crop_p.items() if c != "Not a plant"),
                    key=lambda d: -d["prob"])[:3]
    # The same problem (rust, powdery mildew...) looks alike across plants, so its total over all
    # plants is often clear even when the plant is not.
    cond_p: dict[str, float] = defaultdict(float)
    cond_best: dict[str, int] = {}
    for i in order:
        c = parse_label(labels[i])[2]
        cond_p[c] += float(probs[i])
        cond_best.setdefault(c, int(i))
    clearly_healthy = cond_p.get("Healthy", 0.0) >= HEALTHY_MIN
    conditions = sorted(({"condition": c, "prob": p, "crop": parse_label(labels[cond_best[c]])[0]}
                         for c, p in cond_p.items() if c != "Not a plant" and (c != "Healthy" or clearly_healthy)),
                        key=lambda d: -d["prob"])[:3]
    best = top[0]
    not_plant = best["crop"] == "Not a plant"
    if not_plant:
        status, message = "not_plant", "This does not look like a plant. Take a close photo of the leaf, fruit or stem."
    elif best["prob"] >= CONFIDENT:
        status, message = "confident", ""
    elif best["prob"] >= POSSIBLE:
        status, message = "possible", "The model is not fully sure. Compare the symptoms below with your plant."
    else:
        status, message = "uncertain", ("The model is not sure. Check the symptoms below against your plant, or "
                                        "try a sharper photo in daylight with the affected part filling the frame.")
    adv = advice(best["label"])
    if status == "uncertain":
        top_c = conditions[0] if conditions else None
        adv = advice(labels[cond_best[top_c["condition"]]]) if top_c and top_c["prob"] >= POSSIBLE_PROBLEM else None
    return {"status": status, "message": message, "top": top, "plants": plants, "conditions": conditions,
            "advice": adv}


def create_app(pred: Predictor | None) -> FastAPI:
    app = FastAPI(title="Plant Health Check")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        if pred is None:
            return {"ready": False}
        by_crop: dict[str, list[dict]] = defaultdict(list)
        for l in pred.labels:
            c, part, cond = parse_label(l)
            if c != "Not a plant":
                by_crop[c].append({"part": part, "condition": cond})
        t = pred.metrics.get("test", {})
        return {"ready": True, "backbone": pred.backbone, "n_labels": len(pred.labels), "n_crops": len(by_crop),
                "crops": {c: sorted(v, key=lambda d: (d["condition"] != "Healthy", d["part"], d["condition"]))
                          for c, v in sorted(by_crop.items())},
                "test_accuracy": t.get("accuracy"), "test_f1_macro": t.get("f1_macro")}

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        if pred is None:
            raise HTTPException(503, "The model is not trained yet.")
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image is larger than 15 MB.")
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file is not an image we can read (use JPG or PNG).")
        if min(img.size) < 64:
            raise HTTPException(400, "The image is too small. Use a photo at least 64 pixels wide.")
        t0 = time.perf_counter()
        out = summarise(pred(img), pred.labels)
        out["ms"] = round(1000 * (time.perf_counter() - t0))
        return JSONResponse(out)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "web_model" / "efficientnet_b0" / "seed0"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    run = Path(args.run)
    pred = Predictor(run, args.device) if (run / "best.pt").exists() else None
    if pred is None:
        print(f"No trained model at {run}; the site will say the model is not ready.")
    uvicorn.run(create_app(pred), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
