"""Simulation-based inference on the full attractor circuit (ABC route).

The DDM experiment uses an analytic likelihood on a reduced model. This module
is the likelihood-free cross-check on the full nonlinear attractor circuit
(bellwether's reduced mean-field decision model). It does two things:

  * `abc_rejection` infers the circuit's own parameters (cross-inhibition
    `J_cross` and noise `sigma_noise`) from behavior, with no likelihood, only
    the simulator. If this recovers known parameters, likelihood-free inference
    works on the full circuit.
  * `attractor_capture_dose` measures a subject's manipulability directly in the
    full circuit: the covert drive amplitude that flips an ambiguous decision to
    a target probability. This is the ground-truth manipulability that the fast
    DDM-route capture dose is cross-checked against in run_sbi_crosscheck.py.

ABC (approximate Bayesian computation) is the original simulation-based
inference method: draw parameters from a prior, simulate, keep the draws whose
summary statistics land closest to the observed ones. It needs only the
simulator and numpy, so it adds no heavy dependency.
"""

from __future__ import annotations

import numpy as np

from bellwether.dynamics import Adversary, CircuitParams, Stimulus, simulate

COHERENCES = (0.0, 6.4)
DURATION = 2.5
DT = 5e-4


def attractor_behavior(J_cross, sigma_noise, *, coherences=COHERENCES, n=150, seed=0):
    """Choice and commitment time at each coherence for one circuit setting."""
    params = CircuitParams(J_cross=float(J_cross), sigma_noise=float(sigma_noise))
    out = {}
    for i, c in enumerate(coherences):
        tr = simulate(Stimulus(coherence=c), None, n_trials=n, duration=DURATION,
                      dt=DT, seed=seed + i * 17 + 1, params=params)
        z = tr.S1 - tr.S2
        choice = (z[-1, :] > 0).astype(int)
        ct = tr.commitment_times(rate_bound=15.0)
        out[c] = (choice, ct)
    return out


def summary(behavior):
    """Summary-statistic vector: choice rate and mean/std commit time per coherence."""
    feats = []
    for c in sorted(behavior):
        choice, ct = behavior[c]
        feats.append(float(np.mean(choice)))
        feats.append(float(np.nanmean(ct)))
        feats.append(float(np.nanstd(ct)))
    return np.array(feats)


def simulate_bank(*, n_sims=1500, n=150, prior=((0.02, 0.09), (0.005, 0.04)), seed=0):
    """Simulate a prior bank of (theta, summary) pairs for ABC.

    Build this once and reuse it to do ABC for many observed subjects, since the
    simulation is the expensive part.
    """
    rng = np.random.default_rng(seed)
    (jlo, jhi), (slo, shi) = prior
    thetas = np.column_stack([rng.uniform(jlo, jhi, n_sims),
                              rng.uniform(slo, shi, n_sims)])
    stats = np.array([summary(attractor_behavior(j, s, n=n, seed=1000 + i))
                      for i, (j, s) in enumerate(thetas)])
    return thetas, stats


def abc_from_bank(obs_summary, thetas, stats, *, keep_frac=0.02):
    """Rejection-ABC posterior for one observed summary, against a prior bank.

    Scores each bank draw by z-scored Euclidean distance (the z-scoring uses the
    bank's own spread so no statistic dominates by scale) and keeps the closest
    `keep_frac` as posterior samples.
    """
    sd = stats.std(axis=0)
    sd[sd == 0] = 1.0
    d = np.sqrt(np.sum(((stats - obs_summary) / sd) ** 2, axis=1))
    k = max(5, int(keep_frac * len(thetas)))
    keep = np.argsort(d)[:k]
    return thetas[keep], d[keep]


def abc_rejection(obs_summary, *, n_sims=1500, keep_frac=0.02, n=150,
                  prior=((0.02, 0.09), (0.005, 0.04)), seed=0):
    """Rejection ABC posterior over (J_cross, sigma_noise), simulating its own
    bank. For many subjects, prefer simulate_bank + abc_from_bank."""
    thetas, stats = simulate_bank(n_sims=n_sims, n=n, prior=prior, seed=seed)
    return abc_from_bank(obs_summary, thetas, stats, keep_frac=keep_frac)


def attractor_capture_dose(J_cross, sigma_noise, *, target=0.75, n=400,
                           amp_hi=0.03, seed=0):
    """Covert drive amplitude that flips an ambiguous (coherence 0) decision to
    `target` probability of the upper choice, measured in the full circuit."""
    params = CircuitParams(J_cross=float(J_cross), sigma_noise=float(sigma_noise))

    def p_upper(amp):
        tr = simulate(Stimulus(coherence=0.0),
                      Adversary(amplitude=amp, onset=0.5, offset=2.3),
                      n_trials=n, duration=DURATION, dt=DT, seed=seed, params=params)
        z = tr.S1 - tr.S2
        return float(np.mean(z[-1, :] > 0.0))

    if p_upper(0.0) >= target:
        return 0.0
    if p_upper(amp_hi) < target:
        return float("nan")
    lo, hi = 0.0, amp_hi
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if p_upper(mid) >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
