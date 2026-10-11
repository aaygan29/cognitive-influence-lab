"""Recovering the governing drift of the decision variable.

Given a noisy estimate ``zhat(t)`` of the decision variable, we fit its drift
field with a small odd-plus-constant polynomial:

    dz/dt  ~  c0 + c1 z + c3 z^3 + noise

The cubic ``c1 z + c3 z^3`` is the intrinsic bistable force of the circuit (two
wells separated by a barrier). The constant ``c0`` is a *tilt*: a uniform push
toward one option that the intrinsic dynamics do not explain. Legitimate
sensory evidence produces a tilt, and so does a covert adversarial drive. The
monitor's job is to decide whether the observed tilt is larger than the
presented evidence can account for.

The fit is a plain least-squares problem on finite differences, so it is fast,
has no hyperparameters to tune, and returns an analytic standard error for the
tilt. Because ``zhat`` is only known up to an affine gauge, the tilt is scaled
by the fitted cubic stiffness to make it comparable across trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftFit:
    coeffs: np.ndarray        # [c0, c1, c3] for basis [1, z, z^3]
    tilt: float               # c0, the unexplained uniform drift
    tilt_se: float            # standard error of c0
    stiffness: float          # |c1|, intrinsic restoring near z = 0
    residual_std: float       # std of the drift residual

    @property
    def scaled_tilt(self) -> float:
        """Gauge-comparable tilt.

        The decision axis handed to the fit is normalized to unit variance, which
        fixes the affine scale. Under ``z -> a z`` the linear stiffness ``c1`` is
        invariant while the tilt ``c0`` scales with ``a``, so once the scale is
        pinned by unit-variance normalization the raw tilt is already comparable
        across trials. We do not divide by the stiffness: near the bifurcation
        ``c1 -> 0`` (critical slowing down), and dividing by it would manufacture
        a large "tilt" on exactly the honest ambiguous trials we must not flag.
        """
        return float(self.tilt)


def fit_drift(zhat: np.ndarray, dt: float, smooth: int = 5) -> DriftFit:
    """Least-squares drift fit ``dz/dt = c0 + c1 z + c3 z^3``.

    ``smooth`` applies a short centered moving average to ``zhat`` before taking
    the finite difference, which tames differentiation noise without biasing the
    drift. Returns a :class:`DriftFit`.
    """
    z = np.asarray(zhat, dtype=float)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        z = np.convolve(z, kernel, mode="same")

    dz = np.gradient(z, dt)
    # Trim edges where the moving average and gradient are least reliable.
    pad = max(smooth, 2)
    z_in, dz_in = z[pad:-pad], dz[pad:-pad]

    design = np.stack([np.ones_like(z_in), z_in, z_in ** 3], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, dz_in, rcond=None)
    resid = dz_in - design @ coeffs
    resid_std = float(np.std(resid))

    # Covariance of the least-squares estimate for the standard error of c0.
    gram = design.T @ design
    try:
        cov = np.linalg.inv(gram) * (resid_std ** 2)
        tilt_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        tilt_se = float("inf")

    return DriftFit(
        coeffs=coeffs,
        tilt=float(coeffs[0]),
        tilt_se=tilt_se,
        stiffness=float(abs(coeffs[1])),
        residual_std=resid_std,
    )


def directedness(z: np.ndarray, smooth: int = 1) -> float:
    """Straightness of a 1-D trajectory: net displacement over path length.

    ``|z[-1] - z[0]| / sum(|diff(z)|)``, in [0, 1]. This is the trajectory
    tortuosity measure used in movement analysis, read in one dimension.

    It is exactly the signal that separates a covertly steered decision from an
    honest ambiguous one. A constant steering drive makes the decision variable
    move in one direction (directedness toward 1); an unforced ambiguous
    decision diffuses near the saddle (directedness toward 0, since a random walk
    of ``n`` steps covers net ~ sqrt(n) over path ~ n). It is invariant to the
    sign and scale of the recovered decision axis, so no affine-gauge or
    sign-alignment choice can leak the target of the manipulation. Because a
    diffusion's directedness depends on the number of steps, the window length
    must be held fixed across trials (the monitor does this).

    ``smooth`` applies a centered moving average first. High-frequency
    measurement noise inflates the path length without changing the net
    displacement, which would bias directedness down; smoothing at the noise
    timescale measures the path of the underlying trajectory instead.
    """
    z = np.asarray(z, dtype=float)
    if z.size < 2:
        return 0.0
    if smooth > 1 and z.size >= smooth:
        z = np.convolve(z, np.ones(smooth) / smooth, mode="valid")
    net = abs(z[-1] - z[0])
    path = float(np.sum(np.abs(np.diff(z))))
    return net / path if path > 0 else 0.0


def approach_slice(z: np.ndarray, onset_idx: int, horizon: int,
                   frac: float = 0.85) -> tuple[int, int]:
    """Indices spanning stimulus onset to estimated commitment.

    Commitment is when the decision axis first reaches ``frac`` of its total
    excursion from the onset value. Restricting analysis to this window keeps
    measurements on the approach to the decision and away from the settled
    attractor, where a plateau would dilute trajectory statistics.
    """
    end = min(horizon, len(z))
    base = z[onset_idx]
    exc = z[onset_idx:end] - base
    tot = float(np.max(np.abs(exc))) if exc.size else 0.0
    if tot <= 0:
        return onset_idx, end
    reached = np.flatnonzero(np.abs(exc) >= frac * tot)
    i1 = onset_idx + (int(reached[0]) if reached.size else exc.size)
    return onset_idx, max(i1, onset_idx + 2)


def mean_directedness(z: np.ndarray, onset_idx: int, horizon: int,
                      win: int = 100, step: int = 20, smooth: int = 3) -> float:
    """Average directedness over sliding windows of the approach phase.

    This is the per-trial geometry feature used throughout: it reads how directed
    the decision trajectory is on its way to commitment, which is high for a
    steered or strong-evidence decision and low for an honest ambiguous one.
    """
    i0, i1 = approach_slice(z, onset_idx, horizon)
    vals = [directedness(z[a:a + win], smooth=smooth)
            for a in range(i0, max(i1 - win + 1, i0 + 1), step) if a + win <= len(z)]
    return float(np.mean(vals)) if vals else directedness(z[i0:i1], smooth=smooth)


def steering_z(fit: DriftFit) -> float:
    """Signed significance of the tilt: ``c0 / se(c0)``.

    This is the raw, per-trial steering evidence before any calibration against
    the presented sensory evidence (that calibration lives in ``monitor``).
    """
    if not np.isfinite(fit.tilt_se) or fit.tilt_se <= 0:
        return 0.0
    return float(fit.tilt / fit.tilt_se)
