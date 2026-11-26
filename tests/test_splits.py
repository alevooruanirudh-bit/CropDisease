"""Split integrity: no overlap between train/val/test and no duplicate group across splits."""
import numpy as np
import pandas as pd
import pytest

from src.data.splits import SPLITS, check_split_integrity, group_stratified_split, load_split, make_splits
from src.data.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("splits")
    root = make_synthetic_dataset(tmp / "data", n_per_class=25)
    cfg = {"data": {"dataset": "synthetic", "root": str(root), "split_dir": str(tmp / "split"),
                    "split_ratios": [0.6, 0.2, 0.2], "split_seed": 42, "dedup_max_dist": 4, "hash_workers": 2}}
    stats = make_splits(cfg)
    return tmp / "split", stats


def test_no_overlap(split_dir):
    d, _ = split_dir
    report = check_split_integrity(d)
    assert all(v == 0 for v in report.values()), report


def test_duplicates_removed_and_near_duplicates_grouped(split_dir):
    d, stats = split_dir
    assert stats["exact_byte_duplicates_removed"] == 2  # one planted per class
    df = pd.concat([load_split(d, s).assign(split=s) for s in SPLITS])
    assert not df["path"].str.contains("dup_exact").any()
    for cname in ("healthy", "spotted"):
        near = df[df["path"].str.contains(f"{cname}_dup_near")]
        assert len(near) == 1
        same_group = df[df["group"] == near["group"].iloc[0]]
        assert len(same_group) >= 2, "near-duplicate must share a group with its source"
        assert same_group["split"].nunique() == 1


def test_every_class_in_every_split(split_dir):
    d, _ = split_dir
    for s in SPLITS:
        assert set(load_split(d, s)["label"]) == {0, 1}


def test_group_split_never_breaks_groups():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"label": rng.integers(0, 3, n)})
    df["group"] = [f"{l}_{g}" for l, g in zip(df["label"], rng.integers(0, 60, n))]
    split = group_stratified_split(df, (0.6, 0.2, 0.2), seed=0)
    assert df.assign(split=split).groupby("group")["split"].nunique().max() == 1
    frac = split.value_counts(normalize=True)
    assert abs(frac["test"] - 0.2) < 0.07 and abs(frac["val"] - 0.2) < 0.07


def test_hash_cache_reused(tmp_path):
    from PIL import Image
    from src.data.dedup import compute_hashes
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (40, 40), (i * 60, 100, 30)).save(p)
        paths.append(p)
    cache = tmp_path / "cache.csv"
    md5a, pha = compute_hashes(paths, 2, cache)
    assert cache.exists()
    import src.data.dedup as d
    orig, d._hash_one = d._hash_one, None   # any re-hash would now fail
    try:
        md5b, phb = compute_hashes(paths, 2, cache)
    finally:
        d._hash_one = orig
    assert md5a == md5b and (pha == phb).all()
