import numpy as np
import pytest

from spikeprint.validate import (
    Finding,
    UnvalidatedClaimError,
    benjamini_hochberg,
    bootstrap_ci,
    decide_family,
    gate,
)


# --- Finding.decide (single-hypothesis primitive) ---
def test_finding_passes_when_ci_excludes_null():
    f = Finding("H1", 0.40, "choices13k", ci95=(0.20, 0.60), n=1000).decide()
    assert f.passed
    assert "VALIDATED" in f.render()


def test_finding_fails_when_ci_includes_null():
    f = Finding("H1", 0.05, "choices13k", ci95=(-0.10, 0.20), n=1000).decide()
    assert not f.passed
    assert "UNVALIDATED" in f.render()


def test_finding_unvalidated_without_ci_or_n():
    assert not Finding("H1", 0.4, "choices13k").decide().passed
    assert not Finding("H1", 0.4, "choices13k", ci95=(0.2, 0.6)).decide().passed  # no n


# --- bootstrap ---
def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci(list(range(100)), seed=0) == bootstrap_ci(list(range(100)), seed=0)


def test_bootstrap_ci_brackets_mean():
    data = np.random.default_rng(1).normal(5.0, 1.0, 500)
    lo, hi = bootstrap_ci(data, n_boot=2000, seed=0)
    assert lo < 5.0 < hi


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([], seed=0)


# --- Benjamini-Hochberg FDR ---
def test_bh_all_reject_when_all_significant():
    assert benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05).all()


def test_bh_partial_rejection():
    assert benjamini_hochberg([0.001, 0.9, 0.95], alpha=0.05).tolist() == [True, False, False]


def test_bh_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 1.5])


# --- registered family-level decision ---
def test_decide_family_requires_pvalues():
    with pytest.raises(ValueError):
        decide_family([Finding("H1", 0.4, "d", ci95=(0.2, 0.6), n=100)])


def test_decide_family_passes_only_with_ci_and_fdr():
    fs = [
        Finding("H1", 0.40, "d", ci95=(0.20, 0.60), n=100, p_value=0.001),
        Finding("H2", 0.00, "d", ci95=(-0.30, 0.30), n=100, p_value=0.90),  # CI includes null
    ]
    out = decide_family(fs)
    assert out[0].passed and not out[1].passed


# --- gate ---
def test_gate_blocks_unvalidated():
    with pytest.raises(UnvalidatedClaimError):
        gate([Finding("X", 1.0, "d").decide()])


def test_gate_passes_validated():
    gate([Finding("X", 1.0, "d", ci95=(0.5, 1.5), n=10).decide()])


# --- per-metric null (regression test for the mixed-null decide_family bug) ---
def test_decide_family_uses_per_finding_null_for_auc():
    # AUC Finding (null 0.5) whose CI [0.49, 0.55] includes chance -> must NOT pass, even though
    # that CI trivially excludes 0.0 (the old single-scalar-null rule would have passed it).
    auc_f = Finding(
        "H1a", 0.52, "narps", metric="AUC", ci95=(0.49, 0.55), n=500, p_value=0.001, null=0.5
    )
    assert not decide_family([auc_f])[0].passed


def test_decide_family_mixed_null_family():
    h1 = Finding("H1a", 0.58, "narps", metric="AUC", ci95=(0.54, 0.62), n=500, p_value=0.001, null=0.5)
    h2 = Finding("H2", 0.0, "narps", metric="dAUC", ci95=(-0.05, 0.05), n=500, p_value=0.90, null=0.0)
    out = decide_family([h1, h2])
    assert out[0].passed and not out[1].passed
