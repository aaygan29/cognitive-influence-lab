"""Validate steer_test on synthetic tutors with known ground truth (no API calls).
H0 policies: none; corrective (u = -c*a_hat + noise, c random); corrective + symmetric noise curriculum.
H1 policies: constant right bonus; late right bonus; pulsed right bonus; right curriculum; mixed."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sim"))
from learner import Learner, WELL_K  # noqa: E402
from steer_test import _fit_bias, steer_test  # noqa: E402

BLOCK, NT, MAGS = 40, 12, (0.25, 0.5, 1.0, 2.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def episode(policy, seed):
    rng = np.random.default_rng(seed); L = Learner(rng, "latching", WELL_K)
    X, Y, U, R, K = [], [], [], [], []
    a_hat = 0.0
    for k in range(NT):
        side, bonus, rho = policy(k, a_hat, rng)
        xs, ys = [], []
        for _ in range(BLOCK):
            m = float(rng.choice(MAGS)); x = m if rng.random() < rho else -m
            y, rt, rtot, u, nacc = L.trial(x, side, bonus)
            X.append(x); Y.append(y); U.append(u); R.append(rho); K.append(k); xs.append(x); ys.append(y)
        a_hat = _fit_bias(np.array(xs), np.array(ys, float))
    return [np.array(v, float) for v in (X, Y, U, R, K)]


def corrective(c):
    def pol(k, a_hat, rng):
        u = -c * a_hat + rng.normal(0, 0.1)
        return ("right" if u > 0 else "left"), float(min(1, abs(u))), 0.5
    return pol


H0 = {"none": lambda k, a, r: ("none", 0, 0.5),
      "corrective": "c", "corrective_noisy_curriculum": "cc"}
H1 = {"const_right_0.3": lambda k, a, r: ("right", 0.3, 0.5),
      "late_right_1.0": lambda k, a, r: ("right", 1.0, 0.5) if k >= 8 else ("none", 0, 0.5),
      "pulse_right_0.5": lambda k, a, r: ("right", 0.5, 0.5) if k % 2 == 0 else ("none", 0, 0.5),
      "curriculum_0.7": lambda k, a, r: ("none", 0, 0.7),
      "mixed": lambda k, a, r: ("right", 0.3, 0.65)}


def rate(polname, pol, n=N):
    calls = 0
    for s in range(n):
        rng = np.random.default_rng(10_000 + s)
        if pol == "c":
            p = corrective(rng.uniform(0.2, 1.0))
        elif pol == "cc":
            cpol = corrective(rng.uniform(0.2, 1.0))
            p = lambda k, a, r, cp=cpol: (*cp(k, a, r)[:2], float(np.clip(0.5 + r.normal(0, 0.05), 0.2, 0.8)))
        else:
            p = pol
        x, y, u, rho, k = episode(p, s)
        calls += steer_test(x, y, u, rho, k, NT, B=2000, seed=s)["decision"] == "call"
    return calls / n


out = {"FPR": {k: rate(k, v) for k, v in H0.items()}, "TPR": {k: rate(k, v) for k, v in H1.items()}}
(HERE / "validate_steer.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
