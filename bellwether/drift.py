"""Representational drift and latent-dynamics realignment (BCI longitudinal use).

Long-term neural recordings show that the mapping from latent state to recorded
activity drifts over time: the same behavior is carried by changing population
codes, and a decoder fixed on one session decays on the next. The published
remedy is to align the low-dimensional latent dynamics across sessions with a
geometric (orthogonal Procrustes) transformation, which restores a stable,
transferable decoder.

This module reproduces that problem and that fix inside the bellwether model, so
the directedness detector can be evaluated the way it would be deployed on real
recordings over time. Three deployments are compared:

  * per-trial estimation (the bellwether default): the decision axis is
    re-estimated on every trial, so it is inherently drift-robust but pays a
    re-estimation cost and needs enough within-trial signal;
  * fixed axis: the decision axis is learned once on a reference session and
    reused, which is cheap but decays as the code drifts;
  * fixed axis + alignment: the reference axis is carried into each later
    session through an orthogonal Procrustes alignment of honest-trial subspaces,
    which recovers the fixed-axis performance without re-estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import Trajectory
from .population import PopulationReadout


def drifted_readout(base: PopulationReadout, drift_level: float, dropout: float,
                    seed: int) -> PopulationReadout:
    """Return a readout whose latent-to-neural loadings have drifted.

    Drift is modeled as per-unit tuning change (additive Gaussian perturbation of
    the loadings) plus unit dropout (a fraction of units silenced), matching the
    two components reported for real representational drift. The returned object
    carries its own loadings via a stored override.
    """
    rng = np.random.default_rng(seed)
    base_rng = np.random.default_rng(base.seed)
    W0 = base_rng.standard_normal((base.n_units, 2))
    bias = base_rng.standard_normal(base.n_units) * 0.5
    # Drift interpolates the latent-to-neural code from the reference toward a
    # fully fresh code: drift_level 0 is identical, 1 is an entirely new code.
    W_fresh = rng.standard_normal((base.n_units, 2))
    W = (1.0 - drift_level) * W0 + drift_level * W_fresh
    if dropout > 0:
        drop = rng.random(base.n_units) < dropout
        W[drop] = 0.0
    ro = PopulationReadout(n_units=base.n_units, gain=base.gain, seed=base.seed)
    ro._W_override = W           # type: ignore[attr-defined]
    ro._bias_override = bias     # type: ignore[attr-defined]
    return ro


def project_fixed(rates: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project a single trial's (T, n_units) rates onto a fixed N-dim axis."""
    X = rates - rates.mean(axis=0, keepdims=True)
    z = X @ axis
    s = z.std()
    return z / s if s > 0 else z


def reference_axis(readout: PopulationReadout, traj: Trajectory,
                   rate_bound: float = 15.0) -> np.ndarray:
    """Learn one decision axis on a reference session: the choice-difference direction."""
    rates = readout.project(traj)                 # (T, n_trials, n_units)
    choices = traj.choices(rate_bound)
    post = int(0.85 * rates.shape[0])
    g1, g2 = choices == 1, choices == -1
    if g1.sum() < 2 or g2.sum() < 2:
        # Fall back to the top PC of the session-mean trajectory.
        m = rates.mean(1)
        m = m - m.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(m, full_matrices=False)
        axis = Vt[0]
    else:
        axis = rates[post, g1].mean(0) - rates[post, g2].mean(0)
    n = np.linalg.norm(axis)
    return axis / n if n > 0 else axis


def honest_subspace(readout: PopulationReadout, traj: Trajectory, dim: int = 6) -> np.ndarray:
    """Top-``dim`` PCA basis (n_units x dim) of pooled honest activity in a session."""
    rates = readout.project(traj)                 # (T, n_trials, n_units)
    X = rates.reshape(-1, rates.shape[-1])
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:dim].T                             # (n_units, dim)


def procrustes_align(V_ref: np.ndarray, V_new: np.ndarray) -> np.ndarray:
    """Orthogonal transform R (dim x dim) aligning ``V_new`` to ``V_ref``.

    Solves min_R || V_ref - V_new R ||_F over orthogonal R via the SVD of
    ``V_new^T V_ref`` (the standard orthogonal Procrustes solution).
    """
    M = V_new.T @ V_ref
    U, _, Wt = np.linalg.svd(M)
    return U @ Wt


def aligned_axis(ref_axis: np.ndarray, V_ref: np.ndarray, V_new: np.ndarray) -> np.ndarray:
    """Carry a reference decision axis into a drifted session via Procrustes.

    The reference axis is expressed in the reference subspace (coordinates
    ``c = V_ref^T a``), the two honest subspaces are aligned (``R``), and the axis
    is reconstructed in the drifted session as ``V_new R c``.
    """
    c = V_ref.T @ ref_axis
    R = procrustes_align(V_ref, V_new)
    a = V_new @ (R @ c)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a
