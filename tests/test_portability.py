"""Things that differ between Windows and macOS/Linux: path separators, text encodings, odd paths."""
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_catalog  # noqa: E402
from src.data.io import read_mask, read_rgb, write_mask, write_rgb  # noqa: E402


def test_image_io_with_non_ascii_path(tmp_path):
    d = tmp_path / "ünïcödé folder"
    img = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8)
    write_rgb(d / "a.png", img)
    assert (read_rgb(d / "a.png") == img).all()
    mask = np.random.default_rng(1).random((40, 50)) > 0.5
    write_mask(d / "m.png", mask)
    assert (read_mask(d / "m.png") == mask).all()


def test_text_files_are_utf8():
    for f in [*ROOT.glob("kb/*.yaml"), *ROOT.glob("configs/*.yaml"), ROOT / "catalog" / "datasets.yaml"]:
        yaml.safe_load(f.read_text(encoding="utf-8"))


def test_local_ingest_writes_forward_slash_paths(tmp_path, monkeypatch):
    src = tmp_path / "local" / "some__dataset" / "Original Dataset" / "Leaf Spot"
    src.mkdir(parents=True)
    Image.new("RGB", (64, 64), (30, 140, 30)).save(src / "1.jpg")
    (tmp_path / "local" / "some__dataset" / "Augmented" / "Leaf Spot").mkdir(parents=True)
    Image.new("RGB", (64, 64), (30, 140, 30)).save(tmp_path / "local" / "some__dataset" / "Augmented" / "Leaf Spot" / "2.jpg")
    monkeypatch.setattr(ingest_catalog, "LOCAL_DATA", tmp_path / "local")
    monkeypatch.setattr(ingest_catalog, "OUT", tmp_path / "web")
    monkeypatch.setattr(ingest_catalog, "IMG_DIR", tmp_path / "web" / "images")
    ds = {"name": "t", "source": "local:some__dataset", "crop": "Pumpkin", "part": "leaf",
          "include": ["Original Dataset/"], "classes": {"Leaf Spot": "Leaf spot"}}
    rows = ingest_catalog.ingest_local(ds, 10, __import__("random").Random(0))
    assert len(rows) == 1  # the Augmented copy is filtered out by the "/"-style include on every OS
    assert "\\" not in rows[0]["path"] and rows[0]["label"] == "Pumpkin | leaf | Leaf spot"


def test_local_ingest_class_from_file_name(tmp_path, monkeypatch):
    src = tmp_path / "local" / "pests" / "train" / "images"
    src.mkdir(parents=True)
    for n in ["kutu-daun--4-_jpg.rf.a.jpg", "ulat-12-_jpg.rf.b.jpg", "IMG_1.jpg"]:
        Image.new("RGB", (64, 64), (30, 140, 30)).save(src / n)
    monkeypatch.setattr(ingest_catalog, "LOCAL_DATA", tmp_path / "local")
    monkeypatch.setattr(ingest_catalog, "OUT", tmp_path / "web")
    monkeypatch.setattr(ingest_catalog, "IMG_DIR", tmp_path / "web" / "images")
    ds = {"name": "p", "source": "local:pests", "crop": "Chili pepper", "class_regex": r"^([a-z]+(?:-[a-z]+)?)-+\d",
          "classes": {"kutu-daun": ["Aphids", "leaf", "pest"], "ulat": ["Leaf-eating caterpillars", "leaf", "pest"]}}
    rows = ingest_catalog.ingest_local(ds, 10, __import__("random").Random(0))
    assert sorted(r["condition"] for r in rows) == ["Aphids", "Leaf-eating caterpillars"]


def test_local_ingest_long_paths_do_not_collide(tmp_path, monkeypatch):
    src = tmp_path / "local" / "d" / ("A very long dataset folder name that is longer than sixty characters " * 2) / "Spot"
    src.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (64, 64), (30, 140, 30 + i)).save(src / f"{i}.jpg")
    monkeypatch.setattr(ingest_catalog, "LOCAL_DATA", tmp_path / "local")
    monkeypatch.setattr(ingest_catalog, "OUT", tmp_path / "web")
    monkeypatch.setattr(ingest_catalog, "IMG_DIR", tmp_path / "web" / "images")
    ds = {"name": "t", "source": "local:d", "crop": "Pumpkin", "classes": {"Spot": "Leaf spot"}}
    rows = ingest_catalog.ingest_local(ds, 10, __import__("random").Random(0))
    assert len({r["path"] for r in rows}) == 3
