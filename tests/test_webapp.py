"""Knowledge base coverage and website API behaviour (with a stub model, no training needed)."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_catalog import resolve  # noqa: E402
from src.kb import SECTIONS, advice, load_kb, parse_label  # noqa: E402


def catalog_labels() -> set[tuple[str, str, str]]:
    cat = yaml.safe_load(open(ROOT / "catalog" / "datasets.yaml", encoding="utf-8"))
    out = set()
    for ds in cat["datasets"]:
        if isinstance(ds["classes"], dict):
            for v in ds["classes"].values():
                r = resolve(ds, v)
                if r:
                    out.add((r["crop"], r["part"], r["condition"]))
    return out


def test_every_catalog_condition_has_advice():
    kb = load_kb()
    missing = sorted({c for _, _, c in catalog_labels()} - set(kb))
    assert not missing, missing


def test_kb_entries_complete():
    for name, e in load_kb().items():
        assert e.get("type") and e.get("symptoms"), name
        if name != "Not a plant":
            assert e.get("cultural") and e.get("prevention"), name


def test_crop_specific_cause():
    assert "Guignardia" in advice("Grape | leaf | Black rot")["cause"]
    assert "Xanthomonas" in advice("Cauliflower | leaf | Black rot")["cause"]
    a = advice("Tomato | leaf | Late blight")
    assert a["found"] and all(isinstance(a[s], list) for s in SECTIONS)


def test_parse_label():
    assert parse_label("Rice | stem | Sheath blight") == ("Rice", "stem", "Sheath blight")
    assert parse_label("Not a plant") == ("Not a plant", "none", "Not a plant")


# ------------------------------------------------------------------ API
LABELS = ["Tomato | leaf | Healthy", "Tomato | leaf | Late blight", "Potato | tuber | Common scab", "Not a plant",
          "Potato | leaf | Late blight"]


class StubPredictor:
    labels = LABELS
    backbone = "stub"
    metrics: dict = {}

    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def __call__(self, img):
        return self.probs


@pytest.fixture
def client_factory():
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT / "webapp"))
    from server import create_app

    return lambda probs: TestClient(create_app(StubPredictor(probs)))


def jpeg() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (128, 128), (40, 120, 40)).save(b, "JPEG")
    return b.getvalue()


def test_confident_answer_has_advice(client_factory):
    r = client_factory([0.05, 0.85, 0.05, 0.05, 0.0]).post("/api/predict", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    d = r.json()
    assert r.status_code == 200 and d["status"] == "confident"
    assert d["top"][0]["condition"] == "Late blight" and d["plants"][0]["crop"] == "Tomato"
    assert abs(d["plants"][0]["prob"] - 0.90) < 1e-6
    assert d["advice"]["cause"] == "Phytophthora infestans"


def test_uncertain_and_not_plant(client_factory):
    up = {"file": ("a.jpg", jpeg(), "image/jpeg")}
    assert client_factory([0.26, 0.25, 0.25, 0.24, 0.0]).post("/api/predict", files=up).json()["status"] == "uncertain"
    d = client_factory([0.05, 0.05, 0.05, 0.85, 0.0]).post("/api/predict", files=up).json()
    assert d["status"] == "not_plant" and d["plants"][0]["crop"] != "Not a plant"


def test_unsure_plant_still_names_problem(client_factory):
    # Plant split between tomato and potato, but both guesses say late blight (0.48 in total).
    d = client_factory([0.22, 0.24, 0.10, 0.20, 0.24]).post("/api/predict",
                                                             files={"file": ("a.jpg", jpeg(), "image/jpeg")}).json()
    assert d["status"] == "uncertain"
    assert d["conditions"][0]["condition"] == "Late blight" and abs(d["conditions"][0]["prob"] - 0.48) < 1e-6
    assert d["advice"]["condition"] == "Late blight"


def test_rejects_non_image(client_factory):
    r = client_factory([0.2] * 5).post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_info_lists_crops(client_factory):
    d = client_factory([0.2] * 5).get("/api/info").json()
    assert d["ready"] and set(d["crops"]) == {"Tomato", "Potato"}
