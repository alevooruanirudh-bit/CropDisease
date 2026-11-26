"""Plant-health knowledge base: symptoms, causes, treatment and prevention per condition.

Entries live in ``kb/*.yaml`` keyed by condition name (the third part of a canonical label
``Crop | part | Condition``). ``pathogen_by_crop`` gives a crop-specific cause where the same
condition name is caused by different organisms on different crops.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parents[1] / "kb"
SECTIONS = ("cultural", "organic", "chemical", "prevention")
DISCLAIMER = ("This is general guidance, not a diagnosis by an expert. Confirm with your local agricultural "
              "extension officer before spraying. For any chemical, use only products registered in your "
              "country and follow the label for dose, safety equipment and pre-harvest interval.")


@lru_cache(maxsize=1)
def load_kb(kb_dir: str | None = None) -> dict[str, dict]:
    """Merge every YAML file in ``kb/`` into one dict. Duplicate keys are an error."""
    out: dict[str, dict] = {}
    extra: dict[str, dict] = {}
    for f in sorted(Path(kb_dir or KB_DIR).glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        # "_pathogen_by_crop": {Condition: {Crop: cause}} adds crop-specific causes to entries in other files
        for cond, by_crop in (data.pop("_pathogen_by_crop", None) or {}).items():
            extra.setdefault(cond, {}).update(by_crop)
        for k, v in data.items():
            if k in out:
                raise ValueError(f"Duplicate KB entry {k!r} in {f.name}")
            out[k] = v
    for cond, by_crop in extra.items():
        if cond not in out:
            raise ValueError(f"_pathogen_by_crop refers to unknown KB entry {cond!r}")
        out[cond]["pathogen_by_crop"] = {**(out[cond].get("pathogen_by_crop") or {}), **by_crop}
    return out


def parse_label(label: str) -> tuple[str, str, str]:
    """``'Tomato | leaf | Late blight'`` -> ('Tomato', 'leaf', 'Late blight'); 'Not a plant' -> itself."""
    parts = [p.strip() for p in label.split("|")]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "Not a plant", "none", "Not a plant"


def advice(label: str) -> dict:
    """Everything the website shows for a predicted label."""
    crop, part, cond = parse_label(label)
    e = load_kb().get(cond)
    if e is None:
        return {"crop": crop, "part": part, "condition": cond, "found": False, "disclaimer": DISCLAIMER}
    cause = (e.get("pathogen_by_crop") or {}).get(crop, e.get("pathogen"))
    return {
        "crop": crop, "part": part, "condition": cond, "found": True,
        "type": e.get("type"), "cause": cause, "symptoms": e.get("symptoms"), "spread": e.get("spread"),
        **{s: list(e.get(s) or []) for s in SECTIONS},
        "disclaimer": DISCLAIMER,
    }
