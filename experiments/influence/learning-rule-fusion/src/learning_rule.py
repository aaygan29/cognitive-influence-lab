"""Learning-rule vector: a mathematical representation of a learner's decision policy.

This module adds a new *vector* to the biosignal-fusion framework. Instead of a
raw biosignal family, it represents each learner in a gambling task by the
**dynamics of their decision rule**: the trajectory of logistic-GLM weights on
task features (gain, loss) over trials.

The idea follows Roy, Bak, Akrami, Brody & Pillow (2021, *Neuron*, PsyTrack) and
Ashwood et al. (2022, *Nat. Neurosci.*, GLM-HMM): a subject's choice behaviour is
a time-varying generalised-linear map from stimulus features to choice, and that
map *is* the learner. Here we use a transparent, dependency-free approximation of
PsyTrack's trajectory: overlapping-window logistic GLMs give a weight trajectory
w_t, summarised into a fixed-length **learning-rule vector**.

Two uses, both exercised in ``scripts/run_learning_rule_risk.py``:

1. As an *individuating* fusion vector. Like a digital-brain neural fingerprint,
   the learning-rule vector identifies the person and improves forecasting of
   that person's later choices (see the geometry-vs-amplitude asymmetry in the
   sibling ``digital-brain`` project: individual structure, not average
   magnitude, is what carries identity).
2. As a *vector of risk*. The loss-aversion coefficient is a control surface: a
   bounded perturbation of it shifts choices predictably (``perturb_loss_weight``).

Nothing here is a claim about brains. It is a claim about behaviour in a gambling
task, estimated from choices alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURES = ("bias", "gain", "loss")  # RT handled separately as a covariate


@dataclass
class LearningRule:
    """Per-subject summary of the decision-rule trajectory."""

    subject_id: str
    weights_mean: np.ndarray            # mean weight per feature over the trajectory
    weights_volatility: np.ndarray      # std of each weight across windows (drift)
    weights_trend: np.ndarray           # early->late slope per weight
    loss_aversion: float                # |w_loss| / |w_gain|
    trajectory: np.ndarray = field(repr=False)  # (n_windows, n_features)

    def vector(self) -> np.ndarray:
        """Fixed-length learning-rule vector used as a fusion family."""
        return np.concatenate(
            [self.weights_mean, self.weights_volatility, self.weights_trend,
             [self.loss_aversion]]
        )


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                  iters: int = 50) -> np.ndarray:
    """Newton-IRLS logistic regression. Returns weight vector (n_features,).

    Dependency-free so the learning-rule vector can be computed anywhere the
    fusion framework runs. L2 ridge keeps degenerate windows (all-accept) finite.
    """
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        grad = X.T @ (p - y) + l2 * w
        H = (X * W[:, None]).T @ X + l2 * np.eye(d)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def extract_learning_rule(
    subject_id: str,
    gain: np.ndarray,
    loss: np.ndarray,
    accept: np.ndarray,
    window: int = 48,
    step: int = 12,
) -> LearningRule:
    """Fit an overlapping-window logistic-GLM trajectory for one subject.

    Parameters
    ----------
    gain, loss : per-trial task features (raw magnitudes; z-scored internally).
    accept : per-trial binary choice (1 = accept the gamble).
    window, step : trajectory resolution (trials).
    """
    gain = np.asarray(gain, float)
    loss = np.asarray(loss, float)
    accept = np.asarray(accept, float)
    # z-score features across the subject's own trials so weights are comparable
    def z(a):
        s = a.std()
        return (a - a.mean()) / (s if s > 1e-8 else 1.0)
    Xg, Xl = z(gain), z(loss)
    n = len(accept)

    traj: List[np.ndarray] = []
    starts = list(range(0, max(1, n - window + 1), step)) or [0]
    for s0 in starts:
        sl = slice(s0, min(s0 + window, n))
        Xw = np.column_stack([np.ones(sl.stop - sl.start), Xg[sl], Xl[sl]])
        yw = accept[sl]
        if yw.sum() == 0 or yw.sum() == len(yw):
            # degenerate window: ridge still returns a finite, shrunk estimate
            w = _fit_logistic(Xw, yw, l2=5.0)
        else:
            w = _fit_logistic(Xw, yw, l2=1.0)
        traj.append(w)
    T = np.vstack(traj)  # (n_windows, 3)

    wmean = T.mean(axis=0)
    wvol = T.std(axis=0)
    if T.shape[0] >= 2:
        x = np.linspace(0, 1, T.shape[0])
        trend = np.array([np.polyfit(x, T[:, k], 1)[0] for k in range(T.shape[1])])
    else:
        trend = np.zeros(T.shape[1])
    la = abs(wmean[2]) / (abs(wmean[1]) + 1e-8)  # loss weight / gain weight
    return LearningRule(subject_id, wmean, wvol, trend, float(la), T)


def perturb_loss_weight(
    rule_weights: np.ndarray,
    gain: np.ndarray,
    loss: np.ndarray,
    dose: float,
) -> Tuple[np.ndarray, float]:
    """Apply a bounded adversarial dose to the loss-aversion coefficient.

    Returns (perturbed accept-probabilities, mean shift in accept rate). This is
    the 'vector of risk': the learning rule is a control surface, and a bounded
    dose on the loss weight moves choices in a predictable direction. Positive
    ``dose`` reduces loss aversion (pushes toward accepting), negative increases
    it (pushes toward rejecting).
    """
    w = rule_weights.copy()
    def z(a):
        s = a.std(); return (a - a.mean()) / (s if s > 1e-8 else 1.0)
    X = np.column_stack([np.ones(len(gain)), z(np.asarray(gain, float)),
                         z(np.asarray(loss, float))])
    p0 = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
    w[2] = w[2] + dose  # dose adds to the (typically negative) loss weight
    p1 = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
    return p1, float((p1 - p0).mean())
