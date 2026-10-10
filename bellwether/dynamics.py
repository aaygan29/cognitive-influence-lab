"""The decision substrate: a reduced two-variable attractor circuit.

We use the standard reduced mean-field model of a mutual-inhibition decision
circuit (two populations of NMDA synaptic gating variables competing through
feedback inhibition). With ambiguous evidence the circuit sits near a symmetric
unstable point; noise and/or input tips it into one of two winner-take-all
attractors. That tipping is the bifurcation the whole project is about:

  * near the tipping point the circuit is maximally sensitive to small inputs,
    which is exactly what an adversary exploits, and
  * approaching the tipping point produces critical slowing down, which is
    exactly what the monitor listens for.

All constants are the conventional values for this reduced model. The decision
variable used throughout the package is ``z = S1 - S2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CircuitParams:
    """Parameters of the reduced two-variable decision circuit."""

    a: float = 270.0        # transfer slope (Hz/nA)
    b: float = 108.0        # transfer offset (Hz)
    d: float = 0.154        # transfer time constant (s)
    gamma: float = 0.641    # gating kinetics factor
    tau_s: float = 0.100    # NMDA gating time constant (s)
    tau_ampa: float = 0.002  # noise (AMPA) time constant (s)
    J_self: float = 0.2609  # recurrent self coupling (nA)
    J_cross: float = 0.0497  # cross inhibition coupling (nA)
    I0: float = 0.3255      # background input (nA)
    J_ext: float = 0.00052  # external input scaling (nA/Hz)
    sigma_noise: float = 0.02  # noise strength (nA)


def _transfer(x: np.ndarray, p: CircuitParams) -> np.ndarray:
    """Abbott-Chance f-I transfer function, numerically safe at the pole.

    ``H(x) = (a x - b) / (1 - exp(-d (a x - b)))`` with the removable
    singularity at ``a x - b = 0`` handled by its limit ``1/d``.
    """
    u = p.a * x - p.b
    small = np.abs(u * p.d) < 1e-4
    out = np.empty_like(u)
    # Away from the pole: direct formula.
    safe = np.where(small, 1.0, u)  # avoid div-by-zero in the masked entries
    out = safe / (1.0 - np.exp(-p.d * safe))
    # At the pole: H -> 1/d.
    out = np.where(small, 1.0 / p.d, out)
    return out


@dataclass
class Stimulus:
    """Sensory evidence presented to the circuit.

    ``coherence`` is the signed percent bias toward population 1 (positive) or
    population 2 (negative). ``mu0`` is the overall stimulus strength (Hz).
    ``onset`` and ``offset`` bound the stimulus window (s).
    """

    mu0: float = 30.0
    coherence: float = 0.0
    onset: float = 0.5
    offset: float = 2.5

    def inputs(self, t: np.ndarray, p: CircuitParams) -> tuple[np.ndarray, np.ndarray]:
        on = (t >= self.onset) & (t < self.offset)
        base = p.J_ext * self.mu0
        i1 = np.where(on, base * (1.0 + self.coherence / 100.0), 0.0)
        i2 = np.where(on, base * (1.0 - self.coherence / 100.0), 0.0)
        return i1, i2


@dataclass
class Adversary:
    """A covert steering drive added to population 1 (the target choice).

    The drive is subthreshold by construction: its amplitude is small relative
    to the sensory scale, so it does not look like evidence. ``amplitude`` is in
    nA, ``onset``/``offset`` bound the drive window (s). The "dose" is the time
    integral of the injected current (nA s), reported by :func:`dose`.
    """

    amplitude: float = 0.0
    onset: float = 0.5
    offset: float = 2.5
    noise: float = 0.0   # per-step jitter std (nA); a stealthy adversary hides
                         # the steering drift inside added diffusion.

    def drive(self, t: np.ndarray) -> np.ndarray:
        """Mean (deterministic) drive. Jitter is added per trial in ``simulate``."""
        on = (t >= self.onset) & (t < self.offset)
        return np.where(on, self.amplitude, 0.0)

    def active_mask(self, t: np.ndarray) -> np.ndarray:
        return (t >= self.onset) & (t < self.offset)

    def dose(self, t: np.ndarray) -> float:
        """Dose = time integral of the mean steering current (nA s).

        Jitter has zero mean and does not change the dose; it only changes how
        detectable a given dose is.
        """
        dt = float(t[1] - t[0])
        return float(np.sum(np.abs(self.drive(t))) * dt)


@dataclass
class Trajectory:
    """Output of a simulation batch (``n_trials`` independent runs)."""

    t: np.ndarray                 # (T,) time axis (s)
    S1: np.ndarray                # (T, n_trials) gating variable, pop 1
    S2: np.ndarray                # (T, n_trials) gating variable, pop 2
    r1: np.ndarray                # (T, n_trials) firing rate, pop 1 (Hz)
    r2: np.ndarray                # (T, n_trials) firing rate, pop 2 (Hz)
    dose: float                   # injected adversarial dose (nA s)
    meta: dict = field(default_factory=dict)

    @property
    def z(self) -> np.ndarray:
        """Decision variable ``z = S1 - S2`` of shape (T, n_trials)."""
        return self.S1 - self.S2

    def commitment_times(self, rate_bound: float = 15.0) -> np.ndarray:
        """First time (s) either population's rate crosses ``rate_bound``.

        Returns ``nan`` for trials that never commit within the window.
        """
        winner = np.maximum(self.r1, self.r2)          # (T, n_trials)
        crossed = winner >= rate_bound
        out = np.full(self.S1.shape[1], np.nan)
        for j in range(crossed.shape[1]):
            idx = np.flatnonzero(crossed[:, j])
            if idx.size:
                out[j] = self.t[idx[0]]
        return out

    def choices(self, rate_bound: float = 15.0) -> np.ndarray:
        """Chosen option per trial: +1 (pop 1), -1 (pop 2), 0 (no commit)."""
        ct = self.commitment_times(rate_bound)
        out = np.zeros(self.S1.shape[1], dtype=int)
        for j in range(out.size):
            if np.isnan(ct[j]):
                continue
            k = int(np.searchsorted(self.t, ct[j]))
            k = min(k, self.S1.shape[0] - 1)
            out[j] = 1 if self.r1[k, j] >= self.r2[k, j] else -1
        return out


def simulate(
    stim: Stimulus,
    adversary: Adversary | None = None,
    *,
    n_trials: int = 200,
    duration: float = 3.0,
    dt: float = 5e-4,
    params: CircuitParams | None = None,
    seed: int | None = None,
) -> Trajectory:
    """Integrate the circuit for ``n_trials`` independent noisy runs.

    Uses Euler-Maruyama. Noise enters as an Ornstein-Uhlenbeck current per
    population (filtered white noise with time constant ``tau_ampa``), the
    conventional way to drive this model. Vectorized across trials.
    """
    p = params or CircuitParams()
    adv = adversary or Adversary()
    rng = np.random.default_rng(seed)

    n_steps = int(round(duration / dt))
    t = np.arange(n_steps) * dt

    i1_ext, i2_ext = stim.inputs(t, p)
    adv_drive = adv.drive(t)
    adv_mask = adv.active_mask(t)

    S1 = np.full(n_trials, 0.06)
    S2 = np.full(n_trials, 0.06)
    In1 = np.zeros(n_trials)
    In2 = np.zeros(n_trials)

    S1_rec = np.empty((n_steps, n_trials))
    S2_rec = np.empty((n_steps, n_trials))
    r1_rec = np.empty((n_steps, n_trials))
    r2_rec = np.empty((n_steps, n_trials))

    # OU (filtered-white-noise) current, standard reduced-model convention:
    #   I += (-I) dt/tau_ampa + sqrt(dt/tau_ampa) * sigma_noise * N(0, 1)
    noise_amp = p.sigma_noise * np.sqrt(dt / p.tau_ampa)

    for k in range(n_steps):
        # Update OU noise currents.
        In1 += (-In1 * dt / p.tau_ampa) + noise_amp * rng.standard_normal(n_trials)
        In2 += (-In2 * dt / p.tau_ampa) + noise_amp * rng.standard_normal(n_trials)

        adv_k = adv_drive[k]
        if adv.noise > 0.0 and adv_mask[k]:
            adv_k = adv_k + adv.noise * rng.standard_normal(n_trials)
        x1 = p.J_self * S1 - p.J_cross * S2 + p.I0 + i1_ext[k] + adv_k + In1
        x2 = p.J_self * S2 - p.J_cross * S1 + p.I0 + i2_ext[k] + In2

        r1 = _transfer(x1, p)
        r2 = _transfer(x2, p)

        S1 += dt * (-S1 / p.tau_s + (1.0 - S1) * p.gamma * r1)
        S2 += dt * (-S2 / p.tau_s + (1.0 - S2) * p.gamma * r2)
        np.clip(S1, 0.0, 1.0, out=S1)
        np.clip(S2, 0.0, 1.0, out=S2)

        S1_rec[k] = S1
        S2_rec[k] = S2
        r1_rec[k] = r1
        r2_rec[k] = r2

    return Trajectory(
        t=t,
        S1=S1_rec,
        S2=S2_rec,
        r1=r1_rec,
        r2=r2_rec,
        dose=adv.dose(t),
        meta={
            "coherence": stim.coherence,
            "mu0": stim.mu0,
            "adv_amplitude": adv.amplitude,
            "seed": seed,
        },
    )
