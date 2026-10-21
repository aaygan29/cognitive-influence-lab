"""Tests for the improved-composite model comparison. Skipped if Knutson data absent.

The key claims: the parsimonious composites generalize out-of-sample, the multivariate/learned models do
not beat them (parsimony wins at this sample size), and the affect composite is robust across time windows.

No em dashes.
"""

import os
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "raw", "knutson", "knutson_stock.csv")
HAS = os.path.exists(DATA)


@pytest.mark.skipif(not HAS, reason="Knutson data not present")
def test_parsimony_beats_complexity_out_of_sample():
    from src.composite_model import load, _oos_scores, _make_predict_fn
    d = load()
    r_affect = _oos_scores(d, _make_predict_fn(d, "affect"))[0]
    r_ridge_full = _oos_scores(d, _make_predict_fn(d, "ridge_full"))[0]
    r_ridge_means = _oos_scores(d, _make_predict_fn(d, "ridge_means"))[0]
    assert r_affect > 0.1, "parsimonious affect composite should generalize"
    assert r_affect > r_ridge_full, "affect composite should beat full multivariate ridge"
    assert r_affect > r_ridge_means, "affect composite should beat ROI-means ridge"


@pytest.mark.skipif(not HAS, reason="Knutson data not present")
def test_affect_composite_robust_across_windows():
    from src.composite_model import window_robustness
    wr = window_robustness()
    assert all(v > 0 for v in wr.values()), f"affect composite should be positive in all windows: {wr}"
