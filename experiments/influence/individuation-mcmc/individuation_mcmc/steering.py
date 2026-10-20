"""The bridge to persuasion-control: per-person capture dose and stealth floor.

`persuasion-control` measures, in a population-level circuit, how much covert
steering flips a control decision (the capture dose) and how detectable that
steering is (the stealth floor). Here we compute the *per-person* version from a
fitted drift-diffusion model, so the manipulability numbers become individual
rather than population.

The DDM is the reduced form of the same decision circuit. A covert persuasion
drive is an added drift ``dv`` toward the target option, applied at a decision
the person finds genuinely ambiguous (baseline drift 0, so the honest choice
probability is just the person's start bias ``w``). Two quantities follow:

  * capture dose ``dv*`` : the smallest added drift that lifts the person's
    probability of the target choice to a target level (default 0.75). This is
    analytic: for a diffusion with unit noise, drift ``v``, boundaries 0 and
    ``a``, start ``w*a``,

        P(upper) = (1 - exp(-2 v w a)) / (1 - exp(-2 v a)),   P(upper | v=0) = w.

  * stealth floor : the smallest added drift at which the steered trials become
    detectable (AUC >= 0.75) from the honest ones, using the honest fitted model
    as the detector (a steered trial is scored by how unlikely its (rt, choice)
    is under the person's honest model). This is Monte-Carlo.

The gap (stealth floor minus capture dose) is the person's covert-capture band,
the same quantity persuasion-control proves non-empty at the population level.
"""

from __future__ import annotations

import numpy as np

from .ddm import simulate_ddm, wfpt_logpdf


def p_upper(v, a, w):
    """Probability of hitting the upper boundary (unit-noise diffusion)."""
    v = np.asarray(v, dtype=float)
    small = np.abs(v) < 1e-9
    v_safe = np.where(small, 1.0, v)
    val = (1.0 - np.exp(-2.0 * v_safe * w * a)) / (1.0 - np.exp(-2.0 * v_safe * a))
    return np.where(small, w, val)


def capture_dose(a, w, target=0.75, hi=8.0):
    """Smallest added drift dv>=0 with P(upper | dv) >= target. NaN if target
    is below the honest baseline (already captured) or unreachable."""
    if w >= target:
        return 0.0
    lo, hi = 0.0, float(hi)
    if p_upper(hi, a, w) < target:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_upper(mid, a, w) >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def ml_fit_ddm(rt, choice, coherence):
    """Maximum-likelihood DDM fit to one subject's (rt, choice, coherence) data.

    Drift is linear in coherence, v = k * coherence, so at an ambiguous
    (coherence 0) decision the drift is 0 and the honest choice probability is
    the start bias w. Returns dict(a, w, t0, k). Uses the analytic WFPT
    likelihood and a bounded optimiser; this is the fast per-subject fit used
    for the DDM-route capture dose in the SBI cross-check.
    """
    from scipy.optimize import minimize
    rt = np.asarray(rt, float); choice = np.asarray(choice, float)
    coh = np.asarray(coherence, float)
    rt_min = float(rt.min())

    def unpack(x):
        a = 0.3 + 2.5 * _sigmoid(x[0])        # boundary in (0.3, 2.8)
        w = 0.1 + 0.8 * _sigmoid(x[1])        # start bias in (0.1, 0.9)
        t0 = 0.95 * rt_min * _sigmoid(x[2])   # below fastest RT
        k = x[3]
        return a, w, t0, k

    def negll(x):
        a, w, t0, k = unpack(x)
        lp = np.asarray(wfpt_logpdf(rt, choice, k * coh, a, w, t0))
        lp = np.where(np.isfinite(lp), lp, -1e4)
        return -float(np.sum(lp))

    best = None
    for seed_x in ([0.0, 0.0, 0.0, 0.1], [0.5, 0.2, -0.5, 0.05], [-0.5, -0.2, 0.5, 0.2]):
        r = minimize(negll, np.array(seed_x), method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    a, w, t0, k = unpack(best.x)
    return dict(a=float(a), w=float(w), t0=float(t0), k=float(k))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _roc_auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _lr_score(rt, choice, a, w, t0, probe=0.5):
    """Locally-most-powerful detector of a positive drift shift: the log
    likelihood ratio of a +probe vs -probe drift hypothesis for each trial.
    Trials pushed toward the upper boundary score higher; honest (zero-drift)
    trials score near zero on average."""
    lp_pos = np.asarray(wfpt_logpdf(rt, choice, probe, a, w, t0))
    lp_neg = np.asarray(wfpt_logpdf(rt, choice, -probe, a, w, t0))
    return lp_pos - lp_neg


def _windowed(scores, window, rng):
    """Mean score over random windows of ``window`` trials (a monitor that
    accumulates a short run of the person's decisions before deciding)."""
    n_win = len(scores) // window
    idx = rng.permutation(len(scores))[: n_win * window].reshape(n_win, window)
    return scores[idx].mean(axis=1)


def stealth_floor(a, w, t0, doses, *, n=4000, target_auc=0.75, seed=0, probe=0.5,
                  window=5):
    """Smallest dose at which steered trials are detectable (AUC>=target_auc).

    Detector: the +/-probe drift log-likelihood ratio of each trial's
    (rt, choice), averaged over a monitor window of ``window`` consecutive
    decisions from the person. Honest trials are drift 0; steered trials are
    drift = dose. With ``window=1`` this is a single-trial detector, which
    saturates well below 1 because one decision barely reveals a small drift;
    a realistic monitor accumulates several decisions.
    """
    rng = np.random.default_rng(seed)
    rt_h, ch_h = simulate_ddm(0.0, a, w, t0, n=n, rng=rng)
    ok_h = ch_h >= 0
    sh = _windowed(_lr_score(rt_h[ok_h], ch_h[ok_h].astype(float), a, w, t0, probe),
                   window, rng)
    aucs = []
    for dv in doses:
        rt_s, ch_s = simulate_ddm(dv, a, w, t0, n=n, rng=rng)
        ok_s = ch_s >= 0
        ss = _windowed(_lr_score(rt_s[ok_s], ch_s[ok_s].astype(float), a, w, t0, probe),
                       window, rng)
        aucs.append(_roc_auc(ss, sh))
    aucs = np.array(aucs)
    doses = np.asarray(doses)
    floor = float("nan")
    for i in range(1, len(doses)):
        if aucs[i] >= target_auc:
            if aucs[i] == aucs[i - 1]:
                floor = float(doses[i]); break
            frac = (target_auc - aucs[i - 1]) / (aucs[i] - aucs[i - 1])
            floor = float(doses[i - 1] + frac * (doses[i] - doses[i - 1])); break
    return floor, aucs
