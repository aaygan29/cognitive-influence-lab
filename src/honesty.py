"""Honesty layer: every reported number is born gated.

A prediction leaves this module only as one of two things: a calibrated estimate wrapped in a
distribution-free coverage guarantee and a strictly-proper score, or an explicit abstention when
the sample cannot support the claimed effect. Nothing here is decorative; each primitive maps to a
gate in ``GUIDELINES.md``.

Primitives
----------
- strictly-proper scores: log-score, Brier, CRPS (Gaussian + ensemble)
- calibration: reliability bins + expected calibration error
- split-conformal prediction: regression intervals and classification sets with finite-sample,
  distribution-free marginal coverage
- power: minimum detectable effect (MDES) for a correlation and for a standardized mean effect
- the gate: wrap a number with (value, CI, null p, score, provenance) and ABSTAIN when the
  claimed effect is below the MDES at this N

Pure numpy/scipy. No sklearn dependency, so the honesty layer has no heavy import surface.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

_EPS = 1e-12


# --------------------------------------------------------------- proper scoring rules

def log_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood (log loss) for binary outcomes. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between probability and outcome. Strictly proper. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), _EPS, None)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> float:
    """CRPS from an ensemble of predictive samples, shape (n_obs, n_samples).

    Uses the energy form CRPS = E|X - y| - 0.5 E|X - X'|, estimated per observation.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    term1 = np.mean(np.abs(s - y[:, None]), axis=1)
    # pairwise mean absolute difference per row, O(m log m) via sorted formula
    m = s.shape[1]
    ss = np.sort(s, axis=1)
    weights = (2 * np.arange(1, m + 1) - m - 1)
    term2 = (2.0 / (m * m)) * np.sum(weights[None, :] * ss, axis=1)
    return float(np.mean(term1 - 0.5 * term2))


# --------------------------------------------------------------- calibration

def reliability(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """Reliability curve + expected calibration error for binary predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, count = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            conf.append(np.nan); acc.append(np.nan); count.append(0)
            continue
        c = float(np.mean(p[m])); a = float(np.mean(y[m])); k = int(m.sum())
        conf.append(c); acc.append(a); count.append(k)
        ece += (k / len(y)) * abs(a - c)
    return {"bin_conf": conf, "bin_acc": acc, "bin_count": count, "ece": float(ece)}


# --------------------------------------------------------------- split conformal prediction

def conformal_regression_qhat(residuals_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal half-width: the (1-alpha)(1+1/n) empirical quantile of |residuals|.

    Guarantees marginal coverage >= 1-alpha for exchangeable data (Vovk; Angelopoulos-Bates).
    """
    r = np.abs(np.asarray(residuals_cal, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError("need calibration residuals")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, level, method="higher"))


def conformal_classification_qhat(
    cal_probs_true: np.ndarray, alpha: float = 0.1
) -> float:
    """LAC (least-ambiguous set) threshold from calibration scores s = 1 - p(true label)."""
    s = 1.0 - np.asarray(cal_probs_true, dtype=float)
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_sets(test_probs: np.ndarray, qhat: float) -> List[List[int]]:
    """Prediction sets: include every class whose prob >= 1 - qhat."""
    P = np.atleast_2d(np.asarray(test_probs, dtype=float))
    keep = P >= (1.0 - qhat)
    return [list(np.where(row)[0]) for row in keep]


def empirical_coverage_interval(y: np.ndarray, center: np.ndarray, half: float) -> float:
    y = np.asarray(y, dtype=float); center = np.asarray(center, dtype=float)
    return float(np.mean(np.abs(y - center) <= half))


# --------------------------------------------------------------- power / MDES

def mdes_correlation(n: int, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> float:
    """Minimum detectable Pearson r at sample size n (Fisher-z approximation)."""
    if n <= 3:
        return float("nan")
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    zb = stats.norm.ppf(power)
    z_r = (za + zb) / np.sqrt(n - 3)
    return float(np.tanh(z_r))


def mdes_standardized_mean(n: int, alpha: float = 0.05, power: float = 0.8,
                           two_sided: bool = True, paired: bool = True) -> float:
    """Minimum detectable Cohen's d. paired/one-sample uses sqrt(n); two-sample uses sqrt(n/2)."""
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    zb = stats.norm.ppf(power)
    denom = np.sqrt(n) if paired else np.sqrt(n / 2.0)
    return float((za + zb) / denom)


# --------------------------------------------------------------- inference helpers

def bootstrap_ci(values: np.ndarray, statistic: Callable = np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = np.array([statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(statistic(v)), "lo": float(lo), "hi": float(hi)}


def permutation_p(x: np.ndarray, y: np.ndarray, statistic: Callable, n_perm: int = 5000,
                  seed: int = 0) -> float:
    """Two-sided permutation p for a statistic(x, y) under label exchange of y."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    obs = statistic(x, y)
    count = 0
    for _ in range(n_perm):
        if statistic(x, rng.permutation(y)) >= abs(obs) or statistic(x, rng.permutation(y)) <= -abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


# --------------------------------------------------------------- the gate

@dataclass
class GatedNumber:
    """A reportable number, or an abstention. Never a bare float."""
    name: str
    value: Optional[float]
    ci: Optional[Sequence[float]] = None
    null_p: Optional[float] = None
    score: Optional[Dict[str, float]] = None
    provenance: Dict[str, object] = field(default_factory=dict)
    abstained: bool = False
    reason: Optional[str] = None
    mdes: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def gate_effect(name: str, effect: float, n: int, kind: str = "correlation",
                alpha: float = 0.05, power: float = 0.8,
                provenance: Optional[Dict[str, object]] = None, **extra) -> GatedNumber:
    """Emit a GatedNumber. Abstain when |effect| < MDES at this N (the study is underpowered
    to license a claim of this size), so a withheld number replaces a guessed one."""
    if kind == "correlation":
        mdes = mdes_correlation(n, alpha, power)
    elif kind == "d":
        mdes = mdes_standardized_mean(n, alpha, power)
    else:
        raise ValueError("kind must be 'correlation' or 'd'")
    prov = dict(provenance or {}); prov.update({"n": n, "alpha": alpha, "power": power, "kind": kind})
    if not np.isfinite(mdes) or abs(effect) < mdes:
        return GatedNumber(name=name, value=None, abstained=True, mdes=mdes, provenance=prov,
                           reason=(f"|effect|={abs(effect):.3f} < MDES={mdes:.3f} at n={n}; "
                                   "underpowered to license a claim of this size"))
    gn = GatedNumber(name=name, value=float(effect), mdes=mdes, provenance=prov)
    for k, v in extra.items():
        setattr(gn, k, v)
    return gn


# --------------------------------------------------------------- self-check (positive control)

def _selfcheck(seed: int = 0) -> Dict[str, object]:
    """Prove the machinery: conformal coverage ~ nominal, calibration low, MDES sane, gate abstains.

    This is a positive control, not a demo. If coverage drifts from nominal or the gate stops
    abstaining on an underpowered effect, the honesty layer is broken; fix it, do not relax it.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {"seed": seed}

    # 1) split-conformal regression coverage on a heteroscedastic problem
    n = 3000
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + rng.normal(0, 0.3 + 0.1 * np.abs(x))
    # trivial "model": predict sin(x); calibrate residuals on half, test on the other half
    mu = np.sin(x)
    resid = y - mu
    cal, test = slice(0, n // 2), slice(n // 2, n)
    alpha = 0.1
    qhat = conformal_regression_qhat(resid[cal], alpha=alpha)
    cov = empirical_coverage_interval(y[test], mu[test], qhat)
    out["conformal_regression"] = {"nominal": 1 - alpha, "empirical_coverage": cov,
                                   "half_width": qhat, "pass": abs(cov - (1 - alpha)) <= 0.03}

    # 2) calibration + proper scores on a well-specified binary model
    z = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-z))
    yb = (rng.uniform(size=n) < p).astype(float)
    rel = reliability(yb, p, n_bins=10)
    out["scores"] = {"log_score": log_score(yb, p), "brier": brier_score(yb, p), "ece": rel["ece"],
                     "pass": rel["ece"] < 0.03}

    # 3) MDES sanity + gate behavior
    mdes50 = mdes_correlation(50)
    mdes500 = mdes_correlation(500)
    weak = gate_effect("weak_r_at_n50", effect=0.15, n=50, kind="correlation")
    strong = gate_effect("strong_r_at_n500", effect=0.30, n=500, kind="correlation")
    out["power"] = {"mdes_r_n50": mdes50, "mdes_r_n500": mdes500,
                    "weak_abstained": weak.abstained, "strong_reported": (not strong.abstained),
                    "pass": (mdes50 > mdes500) and weak.abstained and (not strong.abstained)}

    out["all_pass"] = bool(out["conformal_regression"]["pass"] and out["scores"]["pass"]
                           and out["power"]["pass"])
    return out


if __name__ == "__main__":
    import json
    import os

    res = _selfcheck()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    path = os.path.join(root, "results", "e2_honesty_selfcheck.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print("\nall_pass:", res["all_pass"], "->", path)
