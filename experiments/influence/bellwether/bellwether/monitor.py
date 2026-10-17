"""The online early-warning monitor.

The monitor watches population activity during the approach to a decision and
asks a single question: is this choice being pushed harder than the presented
evidence can explain? It answers with a sequential statistic that can raise an
alarm before the behavioral commitment.

Design choices that keep it honest:

  * It never sees the latent decision variable, the injected drive, or the
    realized choice. It sees population rates and the *presented* evidence only.
  * It is calibrated on honest trials alone. The adversary's trials are never
    used to set any threshold, so there is no leakage from the thing being
    detected into the detector.
  * The steering statistic (directedness of the decision axis) is invariant to
    the sign and scale of the recovered axis, so no gauge choice can leak the
    target of the manipulation.

The core statistic per analysis window is the decision axis's directedness,
minus what the presented evidence predicts, standardized by the honest-trial
residual spread. A duration-robust running mean of that excess trips an alarm at
a calibrated threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import Trajectory
from .estimator import directedness
from .population import PopulationReadout


@dataclass
class TrialRecord:
    """Per-trial inputs to the monitor: population rates and presented evidence."""

    rates: np.ndarray            # (T, n_units)
    reported_coherence: float    # the evidence the task claims to present
    commit_time: float = float("nan")  # ground-truth commit time (eval only)


@dataclass
class MonitorConfig:
    dt: float = 5e-4
    onset: float = 0.5           # stimulus onset (s); analysis starts here
    offset: float = 2.5          # stimulus offset (s)
    win_s: float = 0.05          # analysis window length (s)
    step_s: float = 0.01         # window hop (s)
    commit_frac: float = 0.85    # fraction of excursion that marks commitment
    smooth: int = 3              # decision-axis smoothing (samples) at the noise scale
    target_fpr: float = 0.05     # honest-trial false-alarm rate for the threshold
    null_coherence_tol: float = 1.0  # |coherence| below this counts as "ambiguous"


@dataclass
class Monitor:
    readout: PopulationReadout
    config: MonitorConfig = field(default_factory=MonitorConfig)
    # Calibrated quantities (set by ``calibrate``).
    feat_intercept: float = 0.0
    feat_slope: float = 0.0
    feat_sigma: float = 1.0
    alarm_threshold: float = float("inf")
    _min_windows: int = 2
    _calibrated: bool = False

    # -- geometry helpers ---------------------------------------------------
    def _zhat(self, rates: np.ndarray) -> np.ndarray:
        return self.readout.decision_axis(rates)

    def _approach_slice(self, zhat: np.ndarray) -> tuple[int, int]:
        """Indices spanning stimulus onset to estimated commitment.

        Commitment is when the decision axis first reaches ``commit_frac`` of its
        total excursion from the onset value. Restricting to this window keeps
        the drift fit on the approach phase, where it is meaningful, and away
        from the settled attractor.
        """
        c = self.config
        i0 = int(round(c.onset / c.dt))
        i_off = min(int(round(c.offset / c.dt)), len(zhat) - 1)
        base = zhat[i0]
        excursion = zhat[i0:i_off] - base
        total = np.max(np.abs(excursion)) if excursion.size else 0.0
        if total <= 0:
            return i0, i_off
        reached = np.flatnonzero(np.abs(excursion) >= c.commit_frac * total)
        i1 = i0 + int(reached[0]) if reached.size else i_off
        return i0, max(i1, i0 + 2)

    def _window_features(self, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Directedness per sliding window over the approach phase.

        Returns (center_indices, directedness). Directedness in a fixed-length
        window is the scale- and sign-invariant signature of covert steering: a
        steered decision moves in one direction, an honest ambiguous one
        diffuses. The decision axis is estimated by PCA of the population, i.e.
        along the direction the choice actually varies, which is the direction
        the bifurcation acts on. Measuring there (rather than an arbitrary
        population axis) is what keeps the warning from being hidden in a
        direction orthogonal to the bifurcation manifold.
        """
        c = self.config
        zhat = self._zhat(rates)
        i0, i1 = self._approach_slice(zhat)
        win = max(int(round(c.win_s / c.dt)), 8)
        step = max(int(round(c.step_s / c.dt)), 1)
        smooth = max(c.smooth, 1)              # light smoothing at the noise scale
        centers, feats = [], []
        for a in range(i0, max(i1 - win + 1, i0 + 1), step):
            b = a + win
            if b > len(zhat):
                break
            centers.append((a + b) // 2)
            feats.append(directedness(zhat[a:b], smooth=smooth))
        if not centers:
            # Degenerate (very short approach): one window over the available span.
            return np.array([(i0 + i1) // 2]), np.array([directedness(zhat[i0:i1], smooth=smooth)])
        return np.asarray(centers), np.asarray(feats)

    # -- calibration --------------------------------------------------------
    def calibrate(self, honest: list[TrialRecord]) -> "Monitor":
        """Fit the evidence->directedness baseline and alarm threshold on honest data.

        We regress per-window directedness on ``|reported_coherence|`` (the
        directedness any presented evidence legitimately earns), then standardize
        residuals. The alarm threshold is the quantile of the honest per-trial
        peak running-mean residual that yields the configured false-alarm rate.
        """
        coh, feat = [], []
        per_trial_windows = []
        for rec in honest:
            centers, feats = self._window_features(rec.rates)
            per_trial_windows.append((rec, centers, feats))
            coh.extend([abs(rec.reported_coherence)] * len(feats))
            feat.extend(feats.tolist())
        coh = np.asarray(coh)
        feat = np.asarray(feat)

        A = np.stack([np.ones_like(coh), coh], axis=1)
        (b0, b1), *_ = np.linalg.lstsq(A, feat, rcond=None)
        resid = feat - A @ np.array([b0, b1])
        sigma = float(np.std(resid)) or 1.0
        self.feat_intercept, self.feat_slope, self.feat_sigma = float(b0), float(b1), sigma

        # Alarm threshold is set on honest AMBIGUOUS trials only. Those are the
        # condition a covert attack imitates and, because directedness residuals
        # are heteroscedastic (ambiguous decisions wander more than strong-evidence
        # ones), they are the hardest false-alarm case. Calibrating here gives an
        # honest false-alarm rate on exactly the distribution the attack hides in.
        peaks, is_null = [], []
        for rec, centers, feats in per_trial_windows:
            peaks.append(self._peak_running_mean(feats, abs(rec.reported_coherence)))
            is_null.append(abs(rec.reported_coherence) <= self.config.null_coherence_tol)
        peaks = np.asarray(peaks)
        null_peaks = peaks[np.asarray(is_null)]
        basis = null_peaks if null_peaks.size >= 20 else peaks
        q = 1.0 - self.config.target_fpr
        self.alarm_threshold = float(np.quantile(basis, q)) if basis.size else float("inf")
        self._calibrated = True
        return self

    # -- scoring ------------------------------------------------------------
    def _residuals(self, feats: np.ndarray, abs_coh: float) -> np.ndarray:
        expected = self.feat_intercept + self.feat_slope * abs_coh
        return (feats - expected) / self.feat_sigma

    def _running_mean(self, feats: np.ndarray, abs_coh: float) -> np.ndarray:
        """Cumulative mean of standardized residuals, window by window.

        Entries with fewer than ``min_windows`` samples are set to -inf so they
        cannot trip an alarm. A running mean (not a sum) is duration-robust: a
        consistent steering feat keeps it high, while an honest trial's residuals
        fluctuate around zero and average out no matter how long the approach is.
        """
        r = self._residuals(feats, abs_coh)
        if r.size == 0:
            return r
        cummean = np.cumsum(r) / np.arange(1, r.size + 1)
        cummean[: self._min_windows - 1] = -np.inf
        return cummean

    def _peak_running_mean(self, feats: np.ndarray, abs_coh: float) -> float:
        cm = self._running_mean(feats, abs_coh)
        finite = cm[np.isfinite(cm)]
        return float(finite.max()) if finite.size else 0.0

    def score(self, rec: TrialRecord) -> float:
        """Continuous detection score: peak running-mean steering residual."""
        _, feats = self._window_features(rec.rates)
        return self._peak_running_mean(feats, abs(rec.reported_coherence))

    def alarm(self, rec: TrialRecord) -> dict:
        """Online alarm for one trial.

        Fires at the first window where the running-mean steering residual
        exceeds the calibrated threshold. Returns ``alarmed`` (bool),
        ``alarm_time`` (s), ``lead_time`` (commit_time - alarm_time, s), ``peak``.
        """
        if not self._calibrated:
            raise RuntimeError("Monitor must be calibrated before alarm().")
        centers, feats = self._window_features(rec.rates)
        cm = self._running_mean(feats, abs(rec.reported_coherence))
        peak = float(cm[np.isfinite(cm)].max()) if np.isfinite(cm).any() else 0.0
        crossed = np.flatnonzero(cm > self.alarm_threshold)
        if crossed.size:
            idx = int(centers[crossed[0]])
            alarm_time = idx * self.config.dt
            lead = rec.commit_time - alarm_time if np.isfinite(rec.commit_time) else float("nan")
            return {"alarmed": True, "alarm_time": alarm_time, "lead_time": lead, "peak": peak}
        return {"alarmed": False, "alarm_time": float("nan"),
                "lead_time": float("nan"), "peak": peak}


def records_from_trajectory(
    traj: Trajectory,
    readout: PopulationReadout,
    reported_coherence: float,
    rate_bound: float = 15.0,
) -> list[TrialRecord]:
    """Turn a simulation batch into per-trial monitor inputs.

    ``reported_coherence`` is the evidence the task *claims* to present, which
    for a covert-manipulation trial is the ambiguous cover story, not the hidden
    drive. Commit times are attached for evaluation only.
    """
    rates = readout.project(traj)          # (T, n_trials, n_units)
    commit = traj.commitment_times(rate_bound)
    recs = []
    for j in range(rates.shape[1]):
        recs.append(TrialRecord(rates=rates[:, j, :],
                                reported_coherence=reported_coherence,
                                commit_time=float(commit[j])))
    return recs
