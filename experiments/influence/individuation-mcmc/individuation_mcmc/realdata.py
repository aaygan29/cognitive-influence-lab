"""Load and model a real probabilistic reversal-learning dataset.

Data: Waltmann, Schlagenhauf & Deserno (2022), "Sufficient reliability of the
behavioral and computational readouts of a probabilistic reversal learning
task", Behav. Res. Methods 54:2993-3014 (CC-BY 4.0, https://osf.io/4ng3e/),
as redistributed in the University of Reading reversal-learning reliability
dataset (https://doi.org/10.17864/1947.000466, CC-BY 4.0). Each participant did
the task twice (test-retest), which is the individuation test we want: does a
fit to a person in session 1 predict the same person in session 2.

The raw columns are SubID, Session, correct, stay, pfb (probabilistic feedback
-1/+1), persev, RT (ms). We model the win-stay/lose-shift signal directly: the
per-trial choice is whether the participant repeated their previous choice
(``stay``), and a Rescorla-Wagner value tracks recent feedback and drives the
drift toward staying. This uses exactly the columns present and makes no claim
about option identity or reversal timing.

The dataset is NOT redistributed in this repository; point the loader at a local
copy you downloaded under its CC-BY licence.
"""

from __future__ import annotations

import numpy as np

from .ddm import wfpt_logpdf


def load_waltmann(path, *, rt_lo=0.15, rt_hi=5.0, min_trials=60):
    """Parse the RL_Data_Retest_R_wo.txt file into per (subject, session) trials.

    Returns a list of dicts with keys: sub, session, stay (0/1), rt (seconds),
    pfb (-1/+1). Trials with missing stay/pfb/RT or out-of-range RT are dropped;
    sessions with fewer than ``min_trials`` usable trials are discarded.
    """
    rows = {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        col = {c: i for i, c in enumerate(header)}
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            sub = parts[col["SubID"]]
            sess = parts[col["Session"]]
            key = (sub, sess)
            rows.setdefault(key, []).append(parts)

    out = []
    for (sub, sess), parts in rows.items():
        stay, rt, pfb = [], [], []
        for p in parts:
            try:
                s = float(p[col["stay"]]); r = float(p[col["RT"]]) / 1000.0
                f = float(p[col["pfb"]])
            except (ValueError, IndexError):
                continue
            if np.isnan(s) or np.isnan(r) or np.isnan(f):
                continue
            if not (rt_lo < r < rt_hi):
                continue
            stay.append(int(s)); rt.append(r); pfb.append(f)
        if len(stay) >= min_trials:
            out.append({"sub": sub, "session": sess,
                        "stay": np.array(stay), "rt": np.array(rt),
                        "pfb": np.array(pfb, dtype=float)})
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rl_value(alpha, pfb):
    """Win-stay/lose-shift value available to the drift on each trial: an RL
    trace of recent feedback (pfb in {-1,+1}), using feedback from trials < t."""
    Q = 0.0
    used = np.empty(len(pfb))
    for t in range(len(pfb)):
        used[t] = Q
        Q = Q + alpha * (pfb[t] - Q)
    return used


def _unpack(x, rt_min):
    a = 0.3 + 2.5 * _sigmoid(x[0])        # boundary (0.3, 2.8)
    w = 0.1 + 0.8 * _sigmoid(x[1])        # start bias (0.1, 0.9)
    t0 = 0.95 * rt_min * _sigmoid(x[2])   # below fastest RT
    v0 = x[3]                             # baseline drift toward stay
    wq = x[4]                             # value -> drift coupling
    alpha = _sigmoid(x[5])                # learning rate (0,1)
    return a, w, t0, v0, wq, alpha


def neg_loglik(x, stay, rt, pfb):
    a, w, t0, v0, wq, alpha = _unpack(x, float(rt.min()))
    q = _rl_value(alpha, pfb)
    drift = v0 + wq * q
    lp = np.asarray(wfpt_logpdf(rt, stay.astype(float), drift, a, w, t0))
    lp = np.where(np.isfinite(lp), lp, -1e4)
    return -float(np.sum(lp))


def fit_subject(trials, *, restarts=4, seed=0):
    """ML fit of the RL-DDM to one (subject, session). Returns params dict."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    stay, rt, pfb = trials["stay"], trials["rt"], trials["pfb"]
    best = None
    seeds = [np.array([0.0, 0.0, 0.0, 0.2, 0.3, 0.0])]
    for _ in range(restarts - 1):
        seeds.append(rng.normal(0, 0.6, 6))
    for x0 in seeds:
        r = minimize(neg_loglik, x0, args=(stay, rt, pfb), method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    a, w, t0, v0, wq, alpha = _unpack(best.x, float(rt.min()))
    return {"a": a, "w": w, "t0": t0, "v0": v0, "w_q": wq, "alpha": alpha,
            "negll": float(best.fun), "n": len(stay)}


def mean_loglik(params, trials):
    """Mean per-trial log-likelihood of ``trials`` under given params."""
    stay, rt, pfb = trials["stay"], trials["rt"], trials["pfb"]
    t0 = min(params["t0"], 0.95 * float(rt.min()))
    q = _rl_value(params["alpha"], pfb)
    drift = params["v0"] + params["w_q"] * q
    lp = np.asarray(wfpt_logpdf(rt, stay.astype(float), drift,
                                params["a"], params["w"], t0))
    lp = np.where(np.isfinite(lp), lp, -1e4)
    return float(np.mean(lp))
