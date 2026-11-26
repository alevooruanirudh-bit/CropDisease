"""Download / verify datasets into ``data/``.

Sources (all public, no account or API key needed; the project never contacts Kaggle):

* PlantVillage (color): sparse git checkout of github.com/spMohanty/PlantVillage-Dataset ``raw/color``
  limited to the requested classes (15-class subset ~0.4 GB; 38 classes ~1 GB).
* MangoLeafBD: HuggingFace mirror ``Project-AgML/mango_leaf_disease_classification``
  (4,000 images, original JPEG bytes, 141 MB).
* PlantDoc: sparse git checkout of github.com/pratikkayal/PlantDoc-Dataset, only the 13 folders
  that overlap the paper's 15 PlantVillage classes (~0.2 GB).
* Textures (optional): DTD (Cimpoi et al., 2014), ~0.6 GB, for the background-shift test.

Raw downloads go to ``data/_downloads/``; the experiment-ready layout is ``data/<dataset>/<class>/*``.

Usage:
    python scripts/download_data.py                         # plantvillage mango plantdoc
    python scripts/download_data.py --datasets plantvillage --pv-classes 38
    python scripts/download_data.py --datasets textures
    python scripts/download_data.py --verify-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.classes import (IMG_EXTS, MANGOLEAFBD_8, PLANTDOC_TO_PV15, PLANTVILLAGE_15,  # noqa: E402
                              PLANTVILLAGE_38, iter_dirs, list_images)

DATA = ROOT / "data"
DL = DATA / "_downloads"
SIZE_LIMIT_GB = 5.0

PV_GIT = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PD_GIT = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
MANGO_HF = ("https://huggingface.co/datasets/Project-AgML/mango_leaf_disease_classification/"
            "resolve/main/data/train-00000-of-00001.parquet")
DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
# DTD categories used as held-out backgrounds. DTD has no grass class, so grass stays procedural.
DTD_MAP = {"soil": ["cracked", "porous", "pitted", "bumpy", "stratified"],
           "clutter": ["paisley", "marbled", "banded", "chequered", "zigzagged", "swirly", "polka-dotted", "studded"]}

MANUAL = """
Manual download instructions
----------------------------
PlantVillage (color):  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    unzip and point configs `data.root` at the folder holding the class folders, or arrange:
    data/plantvillage/Tomato___Bacterial_spot/*.JPG ... (folder spellings from any mirror are accepted)
MangoLeafBD:           https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset
    data/mangoleafbd/{Anthracnose,Bacterial Canker,...,Sooty Mould}/*.jpg
PlantDoc:              https://github.com/pratikkayal/PlantDoc-Dataset
    data/plantdoc/{train,test}/<PlantDoc class>/*.jpg
Textures (optional):   data/textures/{soil,grass,clutter}/*.jpg  (any held-out photos)
"""


def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sparse_clone(url: str, dest: Path, paths: list[str]) -> None:
    """Blob-less, sparse, shallow clone that only materialises ``paths``."""
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)])
    run(["git", "sparse-checkout", "set", "--no-cone", *[f"/{p}/" for p in paths]], cwd=dest)


def link(src: Path, dst: Path) -> None:
    """Symlink ``dst`` -> ``src``; on Windows without symlink rights, copy the folder instead."""
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve() or (dst.is_dir() and not dst.is_symlink()):
            return
        raise FileExistsError(f"{dst} exists; remove it to re-link")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:  # Windows needs admin rights or Developer Mode for symlinks
        log(f"Symlink not allowed here; copying {src} -> {dst}")
        shutil.copytree(src, dst)


# ----------------------------------------------------------------------------- datasets
def get_plantvillage(n_classes: int) -> None:
    classes = PLANTVILLAGE_38 if n_classes == 38 else PLANTVILLAGE_15
    target = DATA / "plantvillage"
    repo = DL / "PlantVillage-Dataset"
    sparse_clone(PV_GIT, repo, [f"raw/color/{c}" for c in classes])
    link(repo / "raw" / "color", target)


def get_mango() -> None:
    target = DATA / "mangoleafbd"
    import pandas as pd
    dest = DL / "mango_hf" / "train.parquet"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"Downloading {MANGO_HF}")
        urllib.request.urlretrieve(MANGO_HF, dest)
    df = pd.read_parquet(dest)
    # HF ClassLabel order (Anthracnose ... Sooty_Mold) matches MANGOLEAFBD_8 order.
    for i, row in df.iterrows():
        cname = MANGOLEAFBD_8[int(row["label"])]
        img = row["image"]
        name = Path(img.get("path") or f"{i:05d}.jpg").name
        out = target / cname / name
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(img["bytes"])
    log(f"MangoLeafBD: wrote {len(df)} images to {target}")


def get_plantdoc() -> None:
    repo = DL / "PlantDoc-Dataset"
    folders = [f"{split}/{c}" for split in ("train", "test") for c in PLANTDOC_TO_PV15]
    sparse_clone(PD_GIT, repo, folders)
    link(repo, DATA / "plantdoc")


def get_textures() -> None:
    tgz = DL / "dtd-r1.0.1.tar.gz"
    if not tgz.exists():
        DL.mkdir(parents=True, exist_ok=True)
        log(f"Downloading DTD (~0.6 GB) from {DTD_URL}")
        urllib.request.urlretrieve(DTD_URL, tgz)
    out = DL / "dtd"
    if not out.exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(out, filter="data")
    imgs = next(out.rglob("images"))
    for kind, cats in DTD_MAP.items():
        d = DATA / "textures" / kind
        d.mkdir(parents=True, exist_ok=True)
        for c in cats:
            for p in list_images(imgs / c):
                dst = d / f"{c}_{p.name}"
                if not dst.exists():
                    shutil.copy2(p, dst)
    log("Textures: data/textures/{soil,clutter} from DTD; grass stays procedural.")


# ----------------------------------------------------------------------------- verify
def count_images(root: Path) -> int:
    return sum(len(list_images(d)) for d in [root, *iter_dirs(root)]) if root.exists() else 0


def verify() -> bool:
    ok = True
    from src.data.classes import match_class_dirs
    for name, root, classes in (("plantvillage-15", DATA / "plantvillage", PLANTVILLAGE_15),
                                ("mangoleafbd", DATA / "mangoleafbd", MANGOLEAFBD_8)):
        try:
            dirs = match_class_dirs(root, classes)
            n = {c: len(list_images(d)) for c, d in dirs.items()}
            log(f"{name}: OK, {sum(n.values())} images in {len(n)} classes")
            for c, k in n.items():
                log(f"    {k:6d}  {c}")
        except FileNotFoundError as e:
            ok = False
            log(f"{name}: MISSING ({e})")
    n_pd = count_images(DATA / "plantdoc")
    log(f"plantdoc: {n_pd} images" if n_pd else "plantdoc: MISSING (cross-dataset test will be skipped)")
    n_tex = count_images(DATA / "textures")
    log(f"textures: {n_tex} photos" if n_tex else "textures: none (procedural textures will be used)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["plantvillage", "mango", "plantdoc"],
                    choices=["plantvillage", "mango", "plantdoc", "textures"])
    ap.add_argument("--pv-classes", type=int, default=15, choices=[15, 38])
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        if shutil.which("git") is None:
            print(MANUAL)
            sys.exit(1)
        fns = {"plantvillage": lambda: get_plantvillage(args.pv_classes), "mango": get_mango,
               "plantdoc": get_plantdoc, "textures": get_textures}
        for d in args.datasets:
            try:
                fns[d]()
            except Exception as e:  # keep going; print instructions for whatever failed
                log(f"{d}: automatic download failed ({e!r})")
                print(MANUAL)
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
