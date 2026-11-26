"""Perturbations must be deterministic functions of (seed, family, severity, image path)."""
import numpy as np
import pytest

from src.data.synthetic import synth_leaf
from src.robustness.perturbations import DEFAULT_SEVERITIES, Perturbation
from src.robustness.textures import TextureBank

PATHS = ["a/img0.jpg", "a/img1.jpg"]
STOCHASTIC = {"gaussian_noise", "rotation", "occlusion", "background"}


def extractor(img):
    return img[..., 1] > img[..., 2] + 30  # green-ish pixels


@pytest.fixture(scope="module")
def img():
    return synth_leaf(np.random.default_rng(0), 96)[0]


CASES = [(fam, sev) for fam, sevs in DEFAULT_SEVERITIES.items() for sev in sevs]


@pytest.mark.parametrize("fam,sev", CASES)
def test_same_inputs_same_output(img, fam, sev):
    p1 = Perturbation(fam, sev, PATHS, 7, extractor, TextureBank(None))
    p2 = Perturbation(fam, sev, PATHS, 7, extractor, TextureBank(None))
    a, b, c = p1(img.copy(), 0), p1(img.copy(), 0), p2(img.copy(), 0)
    assert a.dtype == np.uint8 and a.shape == img.shape
    assert np.array_equal(a, b) and np.array_equal(a, c)


@pytest.mark.parametrize("fam,sev", [(f, s) for f, s in CASES if f in STOCHASTIC])
def test_seed_and_image_change_output(img, fam, sev):
    base = Perturbation(fam, sev, PATHS, 7, extractor, TextureBank(None))(img.copy(), 0)
    other_seed = Perturbation(fam, sev, PATHS, 8, extractor, TextureBank(None))(img.copy(), 0)
    other_img = Perturbation(fam, sev, PATHS, 7, extractor, TextureBank(None))(img.copy(), 1)
    assert not np.array_equal(base, other_seed) or not np.array_equal(base, other_img)


def test_perturbation_does_not_mutate_input(img):
    ref = img.copy()
    for fam, sev in CASES:
        Perturbation(fam, sev, PATHS, 7, extractor, TextureBank(None))(img, 0)
    assert np.array_equal(img, ref)


def test_background_keeps_leaf_pixels(img):
    out = Perturbation("background", "soil", PATHS, 7, extractor, TextureBank(None))(img, 0)
    m = extractor(img)
    assert np.array_equal(out[m], img[m]) and not np.array_equal(out[~m], img[~m])
