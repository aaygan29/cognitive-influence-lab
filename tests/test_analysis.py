import numpy as np
import pytest

from spikeprint.analysis import auc, incremental_validity, predictive_validity


def test_auc_perfect_separation():
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0


def test_auc_reversed_is_zero():
    assert auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0


def test_auc_all_ties_is_chance():
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        auc([1, 2, 3], [1, 1, 1])


def test_predictive_validity_detects_real_signal():
    rng = np.random.default_rng(0)
    n = 400
    groups = rng.integers(0, 80, n)
    score = rng.normal(size=n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-1.5 * score))).astype(int)
    f = predictive_validity("H1", "synthetic", score, labels, groups=groups, n_boot=500, n_perm=500)
    assert f.value > 0.6
    assert f.passed
    assert f.ci95[0] > 0.5  # CI excludes chance


def test_predictive_validity_near_chance_for_noise():
    rng = np.random.default_rng(7)
    n = 300
    score = rng.normal(size=n)
    labels = rng.integers(0, 2, n)  # independent of score
    f = predictive_validity("H1-null", "synthetic", score, labels, n_boot=500, n_perm=500)
    assert abs(f.value - 0.5) < 0.12  # AUC near chance when there is no signal


def test_incremental_validity_csi_beats_uninformative_baseline():
    rng = np.random.default_rng(2)
    n = 400
    groups = rng.integers(0, 80, n)
    csi = rng.normal(size=n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-1.5 * csi))).astype(int)
    baseline = rng.normal(size=n)  # uninformative
    f = incremental_validity("H2", "synthetic", csi, baseline, labels, groups=groups, n_boot=500)
    assert f.value > 0  # CSI AUC exceeds baseline AUC
