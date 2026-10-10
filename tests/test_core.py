"""Unit tests for the bellwether instrument.

Run with:  .venv/bin/python -m pytest -q
"""

from __future__ import annotations

import numpy as np

from bellwether.detect import (
    delong_paired_test,
    permutation_test_auc,
    roc_auc,
    stealth_floor,
)
from bellwether.dynamics import Adversary, CircuitParams, Stimulus, _transfer, simulate
from bellwether.estimator import directedness, fit_drift
from bellwether.ews import (
    kendall_tau,
    sliding_lag1_autocorr,
    sliding_restoring_rate,
    sliding_variance,
)
from bellwether.geometry import commitment_geometry
from bellwether.monitor import Monitor, MonitorConfig, records_from_trajectory
from bellwether.population import PopulationReadout


# --- dynamics ---------------------------------------------------------------
def test_transfer_is_finite_at_pole():
    p = CircuitParams()
    x_pole = p.b / p.a                      # a*x - b = 0 exactly
    val = _transfer(np.array([x_pole]), p)
    assert np.isfinite(val).all()
    assert abs(val[0] - 1.0 / p.d) < 1e-3   # limit is 1/d


def test_strong_evidence_biases_choice():
    tr = simulate(Stimulus(coherence=51.2), n_trials=100, duration=2.0, seed=1)
    choices = tr.choices()
    assert np.mean(choices == 1) > 0.9      # evidence favors pop 1


def test_ambiguous_is_near_chance():
    tr = simulate(Stimulus(coherence=0.0), n_trials=200, duration=2.0, seed=2)
    frac1 = np.mean(tr.choices() == 1)
    assert 0.3 < frac1 < 0.7                # no systematic bias


def test_dose_scales_with_amplitude():
    t = np.linspace(0, 2.5, 5000)
    assert Adversary(amplitude=0.02).dose(t) > Adversary(amplitude=0.01).dose(t) > 0.0


# --- early warning ----------------------------------------------------------
def test_variance_and_autocorr_rise_toward_instability():
    # AR(1) whose coefficient ramps from 0.5 to ~0.99 (restoring force vanishes).
    rng = np.random.default_rng(0)
    n = 4000
    a = np.linspace(0.5, 0.99, n)
    z = np.zeros(n)
    for k in range(1, n):
        z[k] = a[k] * z[k - 1] + rng.standard_normal() * 0.1
    _, var = sliding_variance(z, win=300, step=50)
    _, ac = sliding_lag1_autocorr(z, win=300, step=50)
    _, rate = sliding_restoring_rate(z, dt=1.0, win=300, step=50)
    assert var[-1] > var[0]
    assert ac[-1] > ac[0]
    assert rate[-1] < rate[0]               # restoring rate collapses


# --- drift estimator --------------------------------------------------------
def test_directedness_separates_ramp_from_walk():
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 1, 500) + rng.standard_normal(500) * 0.01
    walk = np.cumsum(rng.standard_normal(500)) * 0.05
    # Smoothing at the noise scale recovers the underlying path (as the monitor does).
    assert directedness(ramp, smooth=20) > 0.9
    assert directedness(walk, smooth=20) < 0.6
    # A clean ramp is maximally directed even without smoothing.
    assert directedness(np.linspace(0, 1, 500)) > 0.99


def test_drift_recovers_tilt_direction():
    dt = 1e-3
    n = 3000
    rng = np.random.default_rng(1)
    z = np.zeros(n)
    tilt = 0.4                              # constant positive drive
    for k in range(1, n):
        drift = tilt + 1.5 * z[k - 1] - 2.0 * z[k - 1] ** 3
        z[k] = z[k - 1] + dt * drift + rng.standard_normal() * 0.01
    fit = fit_drift(z, dt)
    assert fit.tilt > 0                     # sign recovered
    assert fit.scaled_tilt > 0


# --- geometry ---------------------------------------------------------------
def test_commitment_geometry_runs():
    ro = PopulationReadout(n_units=40, seed=0)
    tr = simulate(Stimulus(coherence=6.4), n_trials=120, duration=2.0, seed=3)
    geo = commitment_geometry(tr, ro)
    assert np.isfinite(geo.subspace_angle_deg)
    assert np.isfinite(geo.expansion_ratio)


# --- detect -----------------------------------------------------------------
def test_roc_auc_bounds():
    assert abs(roc_auc(np.array([3, 4, 5]), np.array([0, 1, 2])) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([1, 2, 3]), np.array([1, 2, 3])) - 0.5) < 1e-9


def test_stealth_floor_interpolates():
    doses = np.array([0.0, 0.01, 0.02, 0.03])
    aucs = np.array([0.5, 0.6, 0.8, 0.9])
    f = stealth_floor(doses, aucs, threshold=0.75)
    assert 0.01 < f < 0.02


def test_permutation_test_auc():
    rng = np.random.default_rng(0)
    pos = rng.normal(2, 1, 100)
    neg = rng.normal(0, 1, 100)
    auc, p = permutation_test_auc(pos, neg, n_perm=500)
    assert auc > 0.8 and p < 0.05
    # Same distribution -> not significant.
    _, p_null = permutation_test_auc(rng.normal(0, 1, 100), rng.normal(0, 1, 100), n_perm=500)
    assert p_null > 0.05


def test_delong_paired_prefers_better_detector():
    rng = np.random.default_rng(1)
    labels = np.r_[np.ones(100), np.zeros(100)]
    truth = np.r_[rng.normal(2, 1, 100), rng.normal(0, 1, 100)]
    good = truth + rng.normal(0, 0.3, 200)          # tracks the signal well
    poor = truth + rng.normal(0, 3.0, 200)          # noisy, weaker detector
    res = delong_paired_test(good, poor, labels)
    assert res["auc_a"] > res["auc_b"]
    assert res["n_pos"] == 100 and res["n_neg"] == 100


def test_kendall_tau_detects_trend():
    tau_up, p_up = kendall_tau(np.arange(30, dtype=float))
    assert tau_up > 0.95 and p_up < 0.01
    rng = np.random.default_rng(2)
    tau_flat, p_flat = kendall_tau(rng.standard_normal(30))
    assert abs(tau_flat) < 0.5


# --- monitor (integration) --------------------------------------------------
def test_monitor_separates_manipulation_from_honest_hard():
    ro = PopulationReadout(n_units=40, seed=0)
    cfg = MonitorConfig(dt=5e-4, onset=0.5, offset=2.3)
    honest = []
    for coh in (3.2, 6.4, 25.6, 51.2):
        tr = simulate(Stimulus(coherence=coh), n_trials=30, duration=2.5, dt=5e-4,
                      seed=200 + int(coh))
        honest += records_from_trajectory(tr, ro, reported_coherence=coh)
    # Ample honest ambiguous trials: these set the alarm threshold.
    null = simulate(Stimulus(coherence=0.0), n_trials=150, duration=2.5, dt=5e-4, seed=250)
    honest += records_from_trajectory(null, ro, 0.0)
    mon = Monitor(readout=ro, config=cfg).calibrate(honest)

    hard = records_from_trajectory(
        simulate(Stimulus(coherence=0.0), n_trials=100, duration=2.5, dt=5e-4, seed=500),
        ro, 0.0)
    manip = records_from_trajectory(
        simulate(Stimulus(coherence=0.0), Adversary(amplitude=0.012),
                 n_trials=100, duration=2.5, dt=5e-4, seed=501),
        ro, 0.0)

    s_hard = np.array([mon.score(r) for r in hard])
    s_manip = np.array([mon.score(r) for r in manip])
    assert roc_auc(s_manip, s_hard) > 0.8            # strong separation at this dose
    assert np.mean(s_hard > mon.alarm_threshold) < 0.12  # honest FPR controlled
