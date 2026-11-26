"""BorB on synthetic images with a known ground-truth mask."""
import cv2
import numpy as np
import pytest

from src.data.synthetic import synth_leaf
from src.segmentation.baselines import kmeans_segment, otsu_segment
from src.segmentation.borb import (BorbParams, BorbPlusOptions, borb_original, borb_plus, borb_plus_mask,
                                   estimate_polarity)
from src.segmentation.thresholds import estimate_thresholds, find_valley


def leaf_on_white(size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Uniform green ellipse on pure white: B mask and b mask both recover it exactly."""
    img = np.full((size, size, 3), 255, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 3, size // 5), 30, 0, 360, 1, -1)
    img[mask > 0] = (60, 140, 40)
    return img, mask.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max((a | b).sum(), 1)


def test_borb_original_exact_on_white_background():
    img, gt = leaf_on_white()
    seg, mask = borb_original(img)
    assert np.array_equal(mask, gt)
    assert (seg[~gt] == 0).all(), "background must be black"
    assert np.array_equal(seg[gt], img[gt]), "leaf pixels keep their RGB value"


def test_channel_polarity_matches_paper_on_white():
    img, gt = leaf_on_white()
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    assert img[gt][:, 2].mean() < 130 < img[~gt][:, 2].mean()          # leaf is low-blue
    assert lab[gt][:, 2].mean() > 135 > lab[~gt][:, 2].mean()          # leaf is high-b


def test_estimate_polarity_on_synthetic_leaves():
    rng = np.random.default_rng(0)
    imgs = [synth_leaf(rng, 128)[0] for _ in range(20)]
    pol = estimate_polarity(imgs)
    assert pol["pol_B"] == "low" and pol["pol_b"] == "high"
    assert pol["vote_B_low"] > 0.9 and pol["vote_b_high"] > 0.9


def test_borb_plus_fills_pale_interior_lesion_and_drops_specks():
    img, gt = leaf_on_white(200)
    cv2.circle(img, (100, 100), 8, (205, 205, 205), -1)        # pale lesion: fails both channel masks
    specks = [(15, 15), (185, 20), (20, 180)]
    for c in specks:
        cv2.circle(img, c, 2, (60, 140, 40), -1)                 # isolated green specks in background
    _, m_orig = borb_original(img)
    assert not m_orig[100, 100] and m_orig[15, 15]
    p = BorbParams()
    _, m_plus = borb_plus(img, p, BorbPlusOptions())
    assert m_plus[100, 100], "interior lesion must be kept"
    assert not any(m_plus[y, x] for x, y in specks), "background specks must be removed"
    assert iou(m_plus, gt) > 0.97


def test_lesion_protection_keeps_edge_lesion():
    img, gt = leaf_on_white(200)
    ys, xs = np.nonzero(gt)
    edge = (int(xs.max()) - 4, int(ys[xs.argmax()]))
    cv2.circle(img, edge, 9, (40, 35, 120), -1)    # dark bluish necrosis on the leaf margin: fails b, B<130 -> kept
    masks = borb_plus_mask(img, BorbParams(), BorbPlusOptions())
    assert masks["clean"][edge[1], edge[0]]


def test_crop_output_is_square_and_contains_leaf():
    img, gt = leaf_on_white(200)
    seg, mask = borb_plus(img, BorbParams(), BorbPlusOptions(output="crop", crop_pad=0.05))
    assert seg.shape[0] == seg.shape[1] < 200
    assert np.count_nonzero(seg.any(-1)) == np.count_nonzero(mask)


def test_find_valley_bimodal_and_otsu_fallback():
    x = np.arange(256)
    bimodal = np.exp(-((x - 70) ** 2) / 200) + 0.8 * np.exp(-((x - 190) ** 2) / 300)
    t, method = find_valley(bimodal)
    assert method == "valley" and 110 < t < 150
    unimodal = np.exp(-((x - 120) ** 2) / 400)
    _, method = find_valley(unimodal)
    assert method == "otsu"


def test_auto_thresholds_separate_leaf_and_background():
    rng = np.random.default_rng(1)
    imgs = [synth_leaf(rng, 128)[0] for _ in range(30)]
    th = estimate_thresholds(imgs)
    assert th["pol_B"] == "low" and th["pol_b"] == "high"
    # leaf B is 20-60 and background B is 175-215 in the generator
    assert 60 < th["t_B"] < 175


@pytest.mark.parametrize("fn", [otsu_segment, kmeans_segment])
def test_baselines_find_leaf(fn):
    img, gt = leaf_on_white()
    _, m = fn(img)
    assert iou(m, gt) > 0.95
