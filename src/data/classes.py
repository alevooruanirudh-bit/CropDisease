"""Class lists for each dataset and folder-name normalisation.

Different PlantVillage mirrors spell class folders differently, e.g.
``Pepper,_bell___Bacterial_spot`` (spMohanty / abdallahalidev) vs
``Pepper__bell___Bacterial_spot`` (emmarex 15-class subset). We match folders to
canonical names through :func:`norm_key`, which strips everything but letters/digits.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# The 15 classes used by Ozcan & Polat (2025): Pepper x2, Potato x3, Tomato x10.
PLANTVILLAGE_15 = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

PLANTVILLAGE_38 = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

MANGOLEAFBD_8 = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould",
]

# PlantDoc (Singh et al., 2020) folder -> PlantVillage-15 class. Only overlapping classes.
# PlantDoc has no "Potato healthy" or "Tomato Target Spot" class, so 13/15 classes overlap.
PLANTDOC_TO_PV15 = {
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf": "Tomato___healthy",
}


def norm_key(name: str) -> str:
    """Normalise a class/folder name to lowercase alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def class_list(dataset: str, num_classes: int | None = None, root: str | Path | None = None) -> list[str]:
    """Return the canonical, ordered class list for a dataset.

    ``synthetic`` (used by smoke tests) and unknown datasets fall back to the
    sorted sub-folder names of ``root``.
    """
    if dataset == "plantvillage":
        return list(PLANTVILLAGE_38 if num_classes == 38 else PLANTVILLAGE_15)
    if dataset == "mangoleafbd":
        return list(MANGOLEAFBD_8)
    if root is None:
        raise ValueError(f"Dataset {dataset!r} needs a root to infer classes")
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def iter_dirs(root: str | Path) -> list[Path]:
    """All sub-directories of ``root`` (recursive, following symlinks, skipping hidden ones), sorted."""
    out = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        out += [Path(dirpath) / d for d in dirnames]
    return sorted(out)


def match_class_dirs(root: str | Path, classes: list[str]) -> dict[str, Path]:
    """Map each canonical class to its folder under ``root`` (searched recursively).

    Raises if any class is missing or matched by two different folders.
    """
    root = Path(root)
    wanted = {norm_key(c): c for c in classes}
    found: dict[str, Path] = {}
    for d in iter_dirs(root):
        k = norm_key(d.name)
        if k in wanted:
            c = wanted[k]
            if c in found and found[c] != d:
                raise RuntimeError(f"Class {c!r} matched twice: {found[c]} and {d}. "
                                   "Point data.root at a single image-folder tree (e.g. .../color).")
            found[c] = d
    missing = [c for c in classes if c not in found]
    if missing:
        raise FileNotFoundError(f"Missing class folders under {root}: {missing}")
    return found


def list_images(folder: str | Path) -> list[Path]:
    """Sorted list of image files directly inside ``folder``."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())
