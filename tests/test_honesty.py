"""Tests for the honesty layer. Run: python -m pytest tests/ -q  (or python tests/test_honesty.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import honesty as H  # noqa: E402


def test_proper_scores_reward_truth():
    y = np.array([0, 0, 1, 1.0])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = np.array([0.9, 0.8, 0.2, 0.1])
    assert H.brier_score(y, good) < H.brier_score(y, bad)
    assert H.log_score(y, good) < H.log_score(y, bad)


def test_crps_gaussian_prefers_correct_mean():
    y = np.zeros(500)
    assert H.crps_gaussian(y, np.zeros(500), np.ones(500)) < H.crps_gaussian(y, np.ones(500) * 2, np.ones(500))


def test_conformal_regression_covers():
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 1, 4000)
    q = H.conformal_regression_qhat(resid[:2000], alpha=0.1)
    cov = np.mean(np.abs(resid[2000:]) <= q)
    assert abs(cov - 0.9) <= 0.03


def test_mdes_monotonic_in_n():
    assert H.mdes_correlation(50) > H.mdes_correlation(500) > H.mdes_correlation(5000)


def test_gate_abstains_when_underpowered():
    weak = H.gate_effect("weak", effect=0.15, n=40, kind="correlation")
    strong = H.gate_effect("strong", effect=0.35, n=800, kind="correlation")
    assert weak.abstained and weak.value is None
    assert (not strong.abstained) and strong.value is not None


def test_selfcheck_positive_control():
    res = H._selfcheck(seed=0)
    assert res["all_pass"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
