"""Paired significance tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test on paired predictions over the same samples.

    Uses the exact binomial test when the number of discordant pairs is < 25,
    otherwise the chi-square statistic with continuity correction.

    Returns counts ``b`` (A right, B wrong) and ``c`` (A wrong, B right), the statistic and p-value.
    """
    ca, cb = pred_a == y, pred_b == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "none (no discordant pairs)"}
    if n < 25:
        return {"b": b, "c": c, "statistic": float(min(b, c)),
                "p_value": float(binomtest(min(b, c), n, 0.5).pvalue), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
            "method": "chi2 with continuity correction"}


def paired_bootstrap_diff(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Accuracy difference A-B with a paired percentile-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    da = (pred_a == y).astype(float) - (pred_b == y).astype(float)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boots = da[idx].mean(1)
    return float(da.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
