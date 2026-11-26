"""Download and unify every dataset listed in catalog/datasets.yaml.

For each dataset: fetch original (never ``augmented``) data, map source classes to canonical
``Crop | part | Condition`` labels, sample up to ``cap_per_class`` images per source class,
resize to 288 px on the short side (JPEG q90) and append to ``data/web/manifest.csv``.
Raw downloads are deleted as soon as they are processed. Resumable: finished datasets are
recorded in ``data/web/ingest_state.json``.

Usage:
    python scripts/ingest_catalog.py                    # everything
    python scripts/ingest_catalog.py --only mango coffee
    python scripts/ingest_catalog.py --cap 400          # override cap_per_class
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.data.classes import IMG_EXTS, PLANTVILLAGE_38  # noqa: E402

OUT = ROOT / "data" / "web"
IMG_DIR = OUT / "images"
TMP = ROOT / "data" / "_downloads" / "web_tmp"
STATE = OUT / "ingest_state.json"
MANIFEST = OUT / "manifest.csv"
SHORT_SIDE = 288
HF = "https://huggingface.co"
# Datasets fetched separately (e.g. from Kaggle, which needs a personal account) live outside the project;
# the project only reads those local copies. Override with PLANT_LOCAL_DATA.
LOCAL_DATA = Path(os.environ.get("PLANT_LOCAL_DATA", ROOT.parent / "kaggle_data"))
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "x"


def file_slug(s: str) -> str:
    """Readable but unique file name for a source path: ``slug`` truncates, so add a short hash of the full path
    (otherwise long paths that share a 60-character prefix overwrite each other)."""
    return f"{slug(s)[:50]}_{hashlib.md5(s.encode('utf-8')).hexdigest()[:10]}"


def get_json(url: str, retries: int = 4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def download(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dest)
            return dest
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))
    return dest


def save_resized(img: Image.Image, dest: Path) -> bool:
    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if min(w, h) < 32:
            return False
        s = SHORT_SIDE / min(w, h)
        if s < 1:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------- label resolution
def resolve(ds: dict, raw) -> dict | None:
    """Map a catalog class value to canonical fields, or None to skip."""
    if raw is None:
        return None
    part = ds.get("part", "leaf")
    if "crop" in ds:
        crop = ds["crop"]
        vals = [raw] if isinstance(raw, str) else list(raw)
        cond = vals[0]
        part = vals[1] if len(vals) > 1 else part
        kind = vals[2] if len(vals) > 2 else None
    else:
        vals = list(raw)
        crop, cond = vals[0], vals[1]
        part = vals[2] if len(vals) > 2 else part
        kind = vals[3] if len(vals) > 3 else None
    if kind is None:
        kind = "healthy" if cond == "Healthy" else "disease"
    if kind == "nonplant":
        crop, part, cond = "Not a plant", "none", "Not a plant"
    label = cond if kind == "nonplant" else f"{crop} | {part} | {cond}"
    return {"crop": crop, "part": part, "condition": cond, "kind": kind, "label": label}


# ---------------------------------------------------------------------------- HF parquet datasets
def hf_meta(repo: str, folder: str | None) -> tuple[list[str], list[str] | None, int]:
    """(parquet file paths in chosen folder, class names or None, total bytes)."""
    files = get_json(f"{HF}/api/datasets/{repo}/tree/main?recursive=true")
    pqs = [f for f in files if f["type"] == "file" and f["path"].endswith(".parquet")]
    tops = sorted({(f["path"].rsplit("/", 1)[0] if "/" in f["path"] else ".") for f in pqs})
    if folder is None:
        cand = [t for t in tops if "augment" not in t.lower()]
        for pref in ("raw", "original", "data", "."):
            if pref in cand:
                folder = pref
                break
        else:
            folder = cand[0] if cand else None
    if folder is None:
        raise RuntimeError(f"{repo}: no non-augmented parquet folder among {tops}")
    chosen = [f for f in pqs if (f["path"].rsplit("/", 1)[0] if "/" in f["path"] else ".") == folder]
    info = get_json(f"{HF}/api/datasets/{repo}")
    di = (info.get("cardData") or {}).get("dataset_info") or {}
    configs = di if isinstance(di, list) else [di]
    names = None
    for c in configs:
        if names is None or c.get("config_name") == folder:
            for feat in c.get("features") or []:
                if feat.get("name") not in (None, "label", "labels"):
                    continue
                dt = feat.get("dtype") or feat.get("sequence") or feat.get("list")
                while isinstance(dt, dict) and "class_label" not in dt and len(dt) == 1:
                    dt = next(iter(dt.values()))  # unwrap {"list": {...}} / {"sequence": {...}}
                if isinstance(dt, dict) and "class_label" in dt:
                    cl = dt["class_label"]["names"]
                    names = [cl[str(i)] for i in range(len(cl))] if isinstance(cl, dict) else list(cl)
    return [f["path"] for f in chosen], names, sum(f.get("size", 0) for f in chosen)


def server_names(repo: str, col: str) -> list[str] | None:
    """ClassLabel names of column ``col`` from the HF dataset viewer (None if it is a plain value)."""
    info = get_json(f"https://datasets-server.huggingface.co/info?dataset={urllib.parse.quote(repo, safe='')}")
    for v in info["dataset_info"].values():
        f = v["features"].get(col) or {}
        if f.get("_type") == "ClassLabel":
            return list(f["names"])
    return None


def hf_file_bytes(ref: str) -> bytes | None:
    """Fetch ``hf://datasets/<repo>@<revision>/<file>`` (how converted parquet points at image files)."""
    m = re.match(r"hf://datasets/(.+?)@([^/]+)/(.+)", ref)
    if not m:
        return None
    url = f"{HF}/datasets/{m[1]}/resolve/{m[2]}/{urllib.parse.quote(m[3])}"
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def hf_converted(repo: str) -> list[str]:
    """URLs of the viewer's auto-converted parquet shards (for repos that store image folders or zips)."""
    by_cfg = get_json(f"{HF}/api/datasets/{repo}/parquet")
    cfg = "default" if "default" in by_cfg else sorted(by_cfg)[0]
    return [u for split in sorted(by_cfg[cfg]) for u in by_cfg[cfg][split]]


def ingest_hf(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    kind, src = ds["source"].split(":", 1)
    repo, _, folder = src.partition("@")
    if kind == "hf-conv":
        urls, names, total = hf_converted(repo), None, 0
    else:
        paths, names, total = hf_meta(repo, folder or None)
        urls = [f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(p)}" for p in paths]
    if ds.get("label_col") or names is None:
        names = server_names(repo, ds.get("label_col") or "label") or names
    budget = ds.get("budget_mb", 2500) * 1e6
    rng.shuffle(urls)
    log(f"  {repo}: {len(urls)} shards, {total / 1e6:.0f} MB available, budget {budget / 1e6:.0f} MB, classes {names}")
    classes = ds["classes"]
    counts: dict[str, int] = {}
    rows: list[dict] = []
    used = 0.0

    def fetch(u: str) -> Path:
        return download(u, TMP / slug(repo) / (slug("_".join(u.split("/")[-3:])) + ".parquet"))

    wanted = {k for k, v in classes.items() if v is not None}
    with ThreadPoolExecutor(2) as ex:
        futures = [ex.submit(fetch, u) for u in urls[:2]]
        nxt = 2
        while futures:
            local = futures.pop(0).result()
            if nxt < len(urls):
                futures.append(ex.submit(fetch, urls[nxt]))
                nxt += 1
            used += local.stat().st_size
            table = pq.read_table(local)
            cols = table.column_names
            img_col = next((c for c in cols if c in ("image", "img", "images")), None) or \
                next(c for c in cols if "image" in c.lower() and not c.lower().endswith("_id"))
            lab_col = ds.get("label_col") or next(c for c in cols if c in ("label", "labels") or "label" in c.lower())
            labels = table.column(lab_col).to_pylist()
            groups = table.column(ds["group_col"]).to_pylist() if ds.get("group_col") else None
            order = list(range(len(labels)))
            rng.shuffle(order)
            keep = []
            for i in order:
                lab = labels[i]
                if isinstance(lab, list):
                    if len(lab) != 1:
                        continue
                    lab = lab[0]
                name = names[lab] if (names is not None and isinstance(lab, int)) else str(lab)
                if groups is not None:
                    name = f"{groups[i]} / {name}"
                if name not in classes or classes[name] is None:
                    continue
                if counts.get(name, 0) >= cap:
                    continue
                counts[name] = counts.get(name, 0) + 1
                keep.append((i, name))
            imgs = table.column(img_col)

            def save_one(job: tuple[int, str]) -> Path | None:
                i, name = job
                cell = imgs[i].as_py()
                data = cell.get("bytes") if isinstance(cell, dict) else cell
                if not data and isinstance(cell, dict):  # converted image-folder repos store a file reference
                    data = hf_file_bytes(cell.get("path") or "")
                if not data:
                    return None
                dest = IMG_DIR / ds["name"] / slug(name) / f"{local.stem}_{i}.jpg"
                try:
                    return dest if save_resized(Image.open(io.BytesIO(data)), dest) else None
                except Exception:
                    return None

            with ThreadPoolExecutor(16) as pool:
                for (i, name), dest in zip(keep, pool.map(save_one, keep)):
                    if dest is None:
                        counts[name] -= 1
                        continue
                    rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"],
                                 "source_class": name, **resolve(ds, classes[name])})
            local.unlink(missing_ok=True)
            done = all(counts.get(k, 0) >= cap for k in wanted)
            if done or used >= budget:
                for f in futures:
                    f.cancel()
                break
    unseen = sorted(wanted - set(counts))
    log(f"  {ds['name']}: {len(rows)} images, {used / 1e6:.0f} MB downloaded; per class {counts}"
        + (f"; classes not reached within budget: {unseen}" if unseen else ""))
    if names and not ds.get("group_col"):
        unknown = sorted(set(names) - set(classes))
        if unknown:
            log(f"  WARNING {ds['name']}: source classes missing from catalog (skipped): {unknown}")
    shutil.rmtree(TMP / slug(repo), ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------- other sources
def ingest_plantvillage(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    repo = ROOT / "data" / "_downloads" / "PlantVillage-Dataset"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/spMohanty/PlantVillage-Dataset.git", str(repo)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *[f"/raw/color/{c}/" for c in PLANTVILLAGE_38]],
                   cwd=repo, check=True)
    rows = []
    for cname, raw in ds["classes"].items():
        files = sorted(p for p in (repo / "raw" / "color" / cname).iterdir() if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        n = 0
        for p in files:
            if n >= cap:
                break
            dest = IMG_DIR / ds["name"] / slug(cname) / (p.stem + ".jpg")
            with Image.open(p) as im:
                if save_resized(im, dest):
                    rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"], "source_class": cname,
                                 **resolve(ds, raw)})
                    n += 1
    log(f"  plantvillage: {len(rows)} images")
    return rows


def ingest_cassava(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    """Kaggle Cassava (Makerere) mirror: train.csv + images spread over train_images/part_00X/."""
    repo = ds["source"].split(":", 1)[1]
    siblings = [x["rfilename"] for x in get_json(f"{HF}/api/datasets/{repo}")["siblings"]]
    where = {Path(p).name: p for p in siblings if p.startswith("train_images/")}
    local_csv = download(f"{HF}/datasets/{repo}/resolve/main/train.csv", TMP / "cassava" / "train.csv")
    df = pd.read_csv(local_csv)
    df = df[df["image_id"].isin(where)]
    rows = []
    for lab, g in df.groupby("label"):
        raw = ds["classes"].get(str(lab))
        if raw is None:
            continue
        ids = g["image_id"].tolist()
        rng.shuffle(ids)
        ids = ids[:cap]

        def one(img_id: str):
            try:
                tmp = download(f"{HF}/datasets/{repo}/resolve/main/{where[img_id]}", TMP / "cassava" / img_id)
                dest = IMG_DIR / ds["name"] / str(lab) / (Path(img_id).stem + ".jpg")
                with Image.open(tmp) as im:
                    ok = save_resized(im, dest)
                tmp.unlink(missing_ok=True)
                return dest if ok else None
            except Exception:
                return None

        with ThreadPoolExecutor(8) as ex:
            for dest in ex.map(one, ids):
                if dest:
                    rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"], "source_class": str(lab),
                                 **resolve(ds, raw)})
    shutil.rmtree(TMP / "cassava", ignore_errors=True)
    log(f"  cassava: {len(rows)} images")
    return rows


def ingest_fiftyone(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    """FiftyOne export on HF: samples.json lists filepath + ground_truth.label; images are separate files."""
    repo = ds["source"].split(":", 1)[1]
    meta = download(f"{HF}/datasets/{repo}/resolve/main/samples.json", TMP / slug(repo) / "samples.json")
    samples = json.loads(meta.read_text(encoding="utf-8"))["samples"]
    meta.unlink(missing_ok=True)
    by_cls: dict[str, list[str]] = {}
    for x in samples:
        by_cls.setdefault(x["ground_truth"]["label"], []).append(x["filepath"])
    unknown = sorted(set(by_cls) - set(ds["classes"]))
    if unknown:
        log(f"  WARNING {ds['name']}: source classes missing from catalog (skipped): {unknown}")
    jobs = []
    for name, paths in sorted(by_cls.items()):
        if ds["classes"].get(name) is None:
            continue
        rng.shuffle(paths)
        jobs += [(name, p) for p in paths[:cap]]

    def one(job):
        name, p = job
        dest = IMG_DIR / ds["name"] / slug(name) / (file_slug(p) + ".jpg")
        try:
            tmp = download(f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(p)}", TMP / slug(repo) / slug(p))
            with Image.open(tmp) as im:
                ok = save_resized(im, dest)
            tmp.unlink(missing_ok=True)
            return (name, dest) if ok else None
        except Exception:
            return None

    rows = []
    with ThreadPoolExecutor(16) as ex:
        for k, res in enumerate(ex.map(one, jobs), 1):
            if res:
                name, dest = res
                rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"], "source_class": name,
                             **resolve(ds, ds["classes"][name])})
            if k % 2000 == 0:
                log(f"  {ds['name']}: {k}/{len(jobs)} fetched")
    shutil.rmtree(TMP / slug(repo), ignore_errors=True)
    log(f"  {ds['name']}: {len(rows)} images of {len(jobs)} requested")
    return rows


def ingest_local(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    """Images already on disk under LOCAL_DATA/<dir>. The class key is the image's parent folder name
    (``class_level: -1``), a higher folder (-2, ...), or the whole relative folder (``class_level: path``).
    Optional ``include`` keeps only files whose relative path contains one of the given substrings.
    ``class_regex`` takes the class from the file name instead (its first group), e.g. ``^([a-z-]+?)-+\\d``."""
    root = LOCAL_DATA / ds["source"].split(":", 1)[1]
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found - fetch this dataset first (see catalog comment)")
    level = ds.get("class_level", -1)
    inc = ds.get("include")
    rx = re.compile(ds["class_regex"]) if ds.get("class_regex") else None
    by_cls: dict[str, list[Path]] = {}
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        rel = f.relative_to(root)
        if inc and not any(s in rel.as_posix() for s in inc):
            continue
        if rx:
            m = rx.match(f.name)
            key = m.group(1) if m else None
        else:
            key = rel.parent.as_posix() if level == "path" else rel.parts[level - 1] if len(rel.parts) > -level else None
        if key is not None:
            by_cls.setdefault(key, []).append(f)
    unknown = sorted(set(by_cls) - set(ds["classes"]))
    if unknown:
        log(f"  WARNING {ds['name']}: source classes missing from catalog (skipped): {unknown[:40]}")
    rows = []
    for key, files in sorted(by_cls.items()):
        raw = ds["classes"].get(key)
        if raw is None:
            continue
        rng.shuffle(files)
        for f in files[:cap]:
            dest = IMG_DIR / ds["name"] / slug(key) / (file_slug(f.relative_to(root).with_suffix("").as_posix()) + ".jpg")
            with Image.open(f) as im:
                if save_resized(im, dest):
                    rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"], "source_class": key,
                                 **resolve(ds, raw)})
    log(f"  {ds['name']}: {len(rows)} images; per class {dict((k, min(len(v), cap)) for k, v in by_cls.items() if ds['classes'].get(k) is not None)}")
    return rows


def ingest_imagenette(ds: dict, cap: int, rng: random.Random) -> list[dict]:
    tgz = download(IMAGENETTE_URL, TMP / "imagenette2-160.tgz")
    rows = []
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg"))]
        by_cls: dict[str, list] = {}
        for m in members:
            by_cls.setdefault(m.name.split("/")[-2], []).append(m)
        for cls, ms in sorted(by_cls.items()):
            rng.shuffle(ms)
            for m in ms[:cap]:
                dest = IMG_DIR / ds["name"] / cls / (Path(m.name).stem + ".jpg")
                with Image.open(tf.extractfile(m)) as im:
                    if save_resized(im, dest):
                        rows.append({"path": dest.relative_to(OUT).as_posix(), "dataset": ds["name"], "source_class": cls,
                                     **resolve({"crop": "Not a plant"}, ["Not a plant", "none", "nonplant"])})
    tgz.unlink(missing_ok=True)
    log(f"  imagenette: {len(rows)} images")
    return rows


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "catalog" / "datasets.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--cap", type=int)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    cat = yaml.safe_load(open(args.catalog, encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    for ds in cat["datasets"]:
        name = ds["name"]
        if args.only and name not in args.only:
            continue
        if name in state and not args.redo:
            continue
        cap = args.cap or ds.get("cap_per_class", cat["defaults"]["cap_per_class"])
        ds.setdefault("budget_mb", cat["defaults"]["budget_mb"])
        rng = random.Random(f"ingest-{name}")
        log(f"== {name} ({ds['source']})")
        t0 = time.time()
        shutil.rmtree(IMG_DIR / name, ignore_errors=True)
        try:
            kind = ds["source"].split(":", 1)[0]
            if kind in ("hf", "hf-conv"):
                rows = ingest_hf(ds, cap, rng)
            elif ds["source"] == "git:plantvillage":
                rows = ingest_plantvillage(ds, cap, rng)
            elif kind == "local":
                rows = ingest_local(ds, cap, rng)
            elif kind == "fo":
                rows = ingest_fiftyone(ds, cap, rng)
            elif kind == "hf-files":
                rows = ingest_cassava(ds, cap, rng)
            elif ds["source"] == "tgz:imagenette":
                rows = ingest_imagenette(ds, cap, rng)
            else:
                raise ValueError(ds["source"])
        except Exception as e:
            log(f"  FAILED {name}: {e!r}")
            continue
        for r in rows:
            r["license"] = ds.get("license", "")
            r["field"] = bool(ds.get("field", False))
        part = OUT / "parts" / f"{name}.csv"
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(part, index=False)
        state[name] = {"n": len(rows), "seconds": round(time.time() - t0)}
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    parts = sorted((OUT / "parts").glob("*.csv"))
    dfs = [pd.read_csv(p) for p in parts if p.stat().st_size > 1]
    if dfs:
        man = pd.concat(dfs, ignore_index=True)
        man.to_csv(MANIFEST, index=False)
        log(f"Manifest: {len(man)} images, {man['label'].nunique()} labels, {man['crop'].nunique()} crops")


if __name__ == "__main__":
    main()
