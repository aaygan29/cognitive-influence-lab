"""Critical-slowing-down early-warning statistics.

As a dynamical system approaches a bifurcation its restoring force weakens, so
perturbations decay more slowly. Two fingerprints of that follow directly:
rising variance and rising short-lag autocorrelation of the state. A third,
more direct one is the estimated AR(1) restoring rate, which falls toward zero.

These are generic (they do not know anything about the decision circuit) and
are computed on a sliding window of a single scalar signal.
"""

from __future__ import annotations

import numpy as np


def _windows(n: int, win: int, step: int):
    for start in range(0, n - win + 1, step):
        yield start, start + win


def sliding_variance(z: np.ndarray, win: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Windowed variance. Returns (center_index, variance)."""
    centers, vals = [], []
    for a, b in _windows(len(z), win, step):
        centers.append((a + b) // 2)
        vals.append(np.var(z[a:b]))
    return np.asarray(centers), np.asarray(vals)


def sliding_lag1_autocorr(z: np.ndarray, win: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Windowed lag-1 autocorrelation. Returns (center_index, autocorr)."""
    centers, vals = [], []
    for a, b in _windows(len(z), win, step):
        seg = z[a:b] - np.mean(z[a:b])
        denom = np.dot(seg, seg)
        ac = np.dot(seg[:-1], seg[1:]) / denom if denom > 0 else 0.0
        centers.append((a + b) // 2)
        vals.append(ac)
    return np.asarray(centers), np.asarray(vals)


def sliding_restoring_rate(
    z: np.ndarray, dt: float, win: int, step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed AR(1) restoring rate ``lambda`` from ``z[t+1] = c + a z[t]``.

    The restoring rate is ``-ln(a)/dt`` (1/s); it drops toward zero as the
    system approaches the bifurcation. Returns (center_index, rate).
    """
    centers, vals = [], []
    for a_, b_ in _windows(len(z), win, step):
        seg = z[a_:b_]
        x, y = seg[:-1], seg[1:]
        xc = x - x.mean()
        denom = np.dot(xc, xc)
        a = np.dot(xc, y - y.mean()) / denom if denom > 0 else 0.0
        a = min(max(a, 1e-6), 0.999999)
        centers.append((a_ + b_) // 2)
        vals.append(-np.log(a) / dt)
    return np.asarray(centers), np.asarray(vals)


def kendall_tau(values: np.ndarray) -> tuple[float, float]:
    """Kendall rank-correlation of a series against time, with a normal-approx p.

    This is the field-standard way to test an early-warning indicator for a
    monotone rising trend (it is rank based, so it does not assume the indicator
    grows linearly or that its noise is Gaussian). Returns (tau, two_sided_p).
    """
    from math import erf, sqrt

    y = np.asarray(values, dtype=float)
    n = y.size
    if n < 3:
        return 0.0, 1.0
    # Concordant minus discordant pairs against the (monotone) time index.
    diff = np.sign(y[None, :] - y[:, None])
    iu = np.triu_indices(n, k=1)
    s = float(diff[iu].sum())
    tau = s / (0.5 * n * (n - 1))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = s / sqrt(var_s) if var_s > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(tau), float(p)


def trend_slope(values: np.ndarray) -> float:
    """Least-squares slope of a 1-D series against its index (robust to scale)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    denom = np.dot(x, x)
    return float(np.dot(x, values - values.mean()) / denom) if denom > 0 else 0.0
