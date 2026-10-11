"""Decoded-precision margins (grounded in human uncertainty/PE coding).

Human intracranial recordings show that uncertainty and prediction error are
explicitly encoded by single neurons, and that precision (inverse uncertainty)
gates how strongly a mismatch propagates: when the internal model is imprecise,
bottom-up signals are released. The defensive counterpart is that the moment a
decision is most manipulable is the moment its precision is lowest, i.e. when
the circuit is most uncertain.

This module decodes an uncertainty proxy from population activity alone and uses
it to set an adaptive alarm margin: the monitor tolerates more apparent
directedness when the decision is genuinely uncertain (and therefore noisier),
and tightens when the decision is precise. This realizes an uncertainty-tracking,
ratcheting margin rather than a single global threshold, and it controls the
false-alarm rate evenly across the uncertainty range instead of over-alarming on
ambiguous decisions and under-alarming on easy ones.
"""

from __future__ import annotations

import numpy as np


def decode_uncertainty(zhat: np.ndarray, onset_idx: int, horizon: int,
                       commit_frac: float = 0.6) -> float:
    """Decode an uncertainty proxy: how slowly the decision resolves.

    An ambiguous decision lingers near the saddle and resolves late; a precise,
    strongly-driven one resolves quickly. We read uncertainty as the normalized
    time for the decision axis to reach ``commit_frac`` of its eventual excursion
    from the onset value. This reproduces, in-model, the empirical fact that
    uncertainty is decodable from neural activity, and it is monotonic in evidence
    strength (high for ambiguous, low for strong-evidence decisions).

    Note the coupling that the experiments quantify: because covert steering also
    speeds resolution, it lowers this decoded uncertainty, so the decoded value
    cannot by itself separate steering from genuine evidence. Its honest use is to
    even the false-alarm rate across the honest uncertainty range, not to replace
    the presented-evidence reference for detection.
    """
    end = min(horizon, len(zhat))
    base = zhat[onset_idx]
    exc = zhat[onset_idx:end] - base
    if exc.size < 4:
        return 1.0
    total = float(np.max(np.abs(exc)))
    if total <= 0:
        return 1.0
    reached = np.flatnonzero(np.abs(exc) >= commit_frac * total)
    latency = (reached[0] if reached.size else exc.size) / exc.size
    return float(latency)


def adaptive_threshold(base_threshold: float, uncertainty: float,
                       slope: float) -> float:
    """Ratcheting margin: widen the alarm threshold when decoded uncertainty is high.

    ``threshold(u) = base * (1 + slope * (u - u_ref))`` is applied by the caller
    with a reference uncertainty; here we provide the simple multiplicative form
    ``base * (1 + slope * u)`` and let the experiment center ``u``.
    """
    return base_threshold * (1.0 + slope * uncertainty)
