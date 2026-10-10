"""A high-dimensional neural readout of the two-variable circuit.

The monitor is not allowed to observe the latent decision variable ``z``
directly. It sees only population activity: many noisy units whose rates are a
random nonlinear mixture of the two gating variables. This mirrors the real
situation, where a decision lives on a low-dimensional manifold embedded in a
much larger recorded population, and it is what lets the ``geometry`` module
reproduce the commitment-geometry phenomenon.

Recovery of the decision axis from this population is only possible up to an
affine transform (scale, sign, offset). Every downstream statistic is built to
be invariant to that affine gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import Trajectory


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


@dataclass
class PopulationReadout:
    """Random nonlinear embedding of (S1, S2) into ``n_units`` neurons."""

    n_units: int = 60
    gain: float = 6.0
    seed: int | None = 0

    def project(self, traj: Trajectory) -> np.ndarray:
        """Return population rates of shape (T, n_trials, n_units).

        Each unit's input is a random linear combination of (S1, S2) plus a
        random bias, passed through a softplus. Mixing weights are shared
        across trials (one population, many trials), which is what makes the
        decision axis consistent enough to estimate.
        """
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.n_units, 2))      # loadings on [S1, S2]
        bias = rng.standard_normal(self.n_units) * 0.5
        # (T, n_trials, 2)
        latent = np.stack([traj.S1, traj.S2], axis=-1)
        drive = latent @ W.T + bias                     # (T, n_trials, n_units)
        return _softplus(self.gain * drive)

    def decision_axis(self, rates: np.ndarray) -> np.ndarray:
        """Estimate the 1-D decision variable from population rates.

        ``rates`` is (T, n_units) for a single trial. We take the leading
        principal component of the centered trajectory, which for this embedding
        aligns with the ``z`` axis up to sign and scale. The returned signal is
        affine-normalized (zero mean, unit std) so it is gauge invariant.
        """
        X = rates - rates.mean(axis=0, keepdims=True)
        # Leading right singular vector is the top PC direction.
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        zhat = X @ Vt[0]
        s = zhat.std()
        if s > 0:
            zhat = zhat / s
        return zhat

    def align_sign(self, zhat: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Flip ``zhat`` so it correlates positively with a known reference."""
        if np.corrcoef(zhat, reference)[0, 1] < 0:
            return -zhat
        return zhat
