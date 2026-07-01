"""Real-fMRI neuroforecasting test. Skipped if the Knutson data is absent (not committed).

No em dashes.
"""

import os
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "raw", "knutson", "knutson_stock.csv")
HAS = os.path.exists(DATA)


@pytest.mark.skipif(not HAS, reason="Knutson data not present")
def test_composite_beats_single_rois_out_of_sample():
    from src.neuralforecast import load, cross_experiment
    d = load()
    B = cross_experiment(d, "B")
    V = cross_experiment(d, "V")
    # the composite generalizes out-of-sample; it is sign-stable where single ROIs are not
    assert B["mean_r"] > 0.1, f"composite out-of-sample r should be positive, got {B['mean_r']}"
    assert B["perm_p"] < 0.05, f"composite should beat permutation null, got p={B['perm_p']}"
    assert B["mean_r"] > V["mean_r"], "composite should out-generalize NAcc alone"


@pytest.mark.skipif(not HAS, reason="Knutson data not present")
def test_composite_sign_stable_across_experiments():
    from src.neuralforecast import load, per_experiment_stability
    d = load()
    st = per_experiment_stability(d)
    # composite keeps the same (positive) sign in both experiments, even if weak in one
    assert st["experiment_1"]["B"] > 0 and st["experiment_2"]["B"] > 0
