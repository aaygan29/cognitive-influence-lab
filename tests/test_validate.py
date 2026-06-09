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
