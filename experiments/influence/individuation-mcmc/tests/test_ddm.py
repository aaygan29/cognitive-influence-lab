"""Tests for the DDM likelihood and the learning-rule reconstruction."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from individuation_mcmc.ddm import wfpt_logpdf, simulate_ddm, simulate_subject
from individuation_mcmc.model import _rw_value_diff, pack_data


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def test_wfpt_normalizes_to_choice_probabilities():
    """The two boundary densities integrate to probabilities summing to 1."""
    v, a, w, t0 = 0.8, 1.2, 0.5, 0.2
    tgrid = np.linspace(t0 + 1e-3, 5.0, 4000)
    mass = 0.0
    for c in (0, 1):
        dens = np.exp(np.asarray(wfpt_logpdf(tgrid, np.full_like(tgrid, c), v, a, w, t0)))
        mass += _trapz(dens, tgrid)
    assert abs(mass - 1.0) < 0.02


def test_wfpt_matches_monte_carlo_choice_rate():
    v, a, w, t0 = 0.6, 1.0, 0.5, 0.15
    rt, ch = simulate_ddm(v, a, w, t0, n=20000, rng=np.random.default_rng(3))
    ok = ch >= 0
    emp_upper = float((ch[ok] == 1).mean())
    tgrid = np.linspace(t0 + 1e-3, 5.0, 4000)
    dens = np.exp(np.asarray(wfpt_logpdf(tgrid, np.ones_like(tgrid), v, a, w, t0)))
    ana_upper = _trapz(dens, tgrid)
    assert abs(ana_upper - emp_upper) < 0.03


def test_wfpt_invalid_rt_is_neg_inf():
    lp = float(np.asarray(wfpt_logpdf(0.1, 1.0, 0.5, 1.0, 0.5, 0.2)))
    assert lp == -np.inf


def test_rw_value_bounded_in_unit_interval():
    """Q1 - Q0 stays within [-1, 1] because each Q stays in [0, 1]."""
    rng = np.random.default_rng(0)
    choice = jnp.asarray((rng.random((3, 50)) < 0.5).astype(float))
    reward = jnp.asarray((rng.random((3, 50)) < 0.5).astype(float))
    qdiff = np.asarray(_rw_value_diff(jnp.array([0.1, 0.4, 0.9]), choice, reward))
    assert np.all(qdiff >= -1.0 - 1e-6) and np.all(qdiff <= 1.0 + 1e-6)


def test_pack_data_shapes():
    subs = [simulate_subject(dict(a=1.1, t0=0.2, w=0.5, beta=0.6),
                             dict(alpha=0.3, w_q=0.8), n_trials=40, seed=i)
            for i in range(4)]
    d = pack_data(subs)
    assert d["cue"].shape == (4, 40) and d["rt"].shape == (4, 40)
    assert d["min_rt"].shape == (4,)


def test_p_upper_matches_simulator():
    from individuation_mcmc.steering import p_upper
    from individuation_mcmc.ddm import simulate_ddm
    rt, ch = simulate_ddm(0.8, 1.2, 0.5, 0.15, n=15000, rng=np.random.default_rng(0))
    ok = ch >= 0
    assert abs(float(p_upper(0.8, 1.2, 0.5)) - float((ch[ok] == 1).mean())) < 0.03


def test_capture_dose_reaches_target():
    from individuation_mcmc.steering import p_upper, capture_dose
    dv = capture_dose(1.1, 0.5, 0.75)
    assert abs(float(p_upper(dv, 1.1, 0.5)) - 0.75) < 1e-2


def test_capture_dose_zero_when_already_biased():
    from individuation_mcmc.steering import capture_dose
    assert capture_dose(1.0, 0.8, 0.75) == 0.0
