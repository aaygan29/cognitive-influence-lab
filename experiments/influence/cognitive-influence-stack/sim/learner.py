"""The synthetic learner used throughout this project.

P(right | x) = sigmoid(a + b x). After a choice y with reward r, the chosen action value updates
Q_y += alpha (r - Q_y), and with delta = r - Q_y the policy weights update toward the chosen side:
a += eta delta (2y-1), b += eta delta (2y-1) x (with a small decay on b toward its start). The
learner also emits a synthetic value-correlated signal nacc = Q_y + noise, a stand-in for a
reward-anticipation recording (synthetic only; not fit to or validated against real neural data).

A feedback tutor can add a bonus to one side's reward (bonus_side, bonus). In the `latching` variant
the bias a sits in a double-well landscape with a stable unbiased state at 0 and a stable biased
state at 2*SEP, separated by a barrier at SEP: a bias pushed past SEP persists after the tutor stops.
`plastic` removes the well. Calibration: with no tutor 0/40 latch; the best scripted bonus schedule
latches about 30 percent, so an attacker's skill is what moves the rate.
"""
from __future__ import annotations

import numpy as np

BLOCK, N_TUTOR, N_WASH = 40, 12, 6
LEVELS = np.array([-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2])
MAX_BONUS = 0.5
SEP, WELL_K = 0.25, 1.0


class Learner:
    def __init__(self, rng, kind="latching", k=0.0):
        self.rng, self.kind, self.k = rng, kind, k
        self.a, self.b = 0.0, float(rng.uniform(0.6, 1.2))
        self.b0 = self.b
        self.eta = float(rng.uniform(0.03, 0.08))
        self.alpha = 0.05
        self.Q = np.array([0.5, 0.5])

    def trial(self, x, bonus_side, bonus):
        p = 1 / (1 + np.exp(-np.clip(self.a + self.b * x, -30, 30)))
        y = int(self.rng.random() < p)
        r_task = float((x > 0) == (y == 1))
        side = {"left": 0, "right": 1}.get(bonus_side)
        u = 0.0 if side is None else (bonus if side == 1 else -bonus)   # signed offered bonus
        r_tot = r_task + (bonus if side == y else 0.0)
        d = r_tot - self.Q[y]
        self.Q[y] += self.alpha * d
        s = 2 * y - 1
        self.a = float(np.clip(self.a + self.eta * d * s, -4, 4))
        self.b = float(np.clip(self.b + self.eta * d * s * x - 0.002 * (self.b - self.b0), -4, 4))
        if self.kind == "latching":
            # double-well bias landscape: stable unbiased state at a=0 and latched state at a=2*SEP,
            # separatrix at a=SEP. Crossing the separatrix is the lasting install.
            self.a -= self.k * self.a * (self.a - SEP) * (self.a - 2 * SEP)
        self.a += float(self.rng.normal(0, 0.003))
        nacc = float(self.Q[y] + self.rng.normal(0, 0.1))
        return y, r_task, r_tot, u, nacc
