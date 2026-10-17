"""E7 (extension): emotional states as recoverable displacements on the decision manifold.

The paper argues that a transient affective state (for example a threat-elevated or "angry/anxious"
state) is not a different person but the same phenotype displaced along the manifold: a state raises
or lowers specific affective coordinates. Here we test whether such a displacement is recoverable. An
entity makes decisions in a baseline state and in a threat-elevated state (its threat coordinate
omega/rho is shifted by a known amount); we recover the per-condition posture and ask whether the
recovered shift tracks the true shift. A negative control with zero true shift must recover no shift.

This closes the loop on the state/role extension: states are within-entity contrasts (recovered here),
roles are between-entity prior shifts (discussed in the paper). Simulation of the mechanism.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402
import phenotype as P  # noqa: E402


def _posture_threat(rng, la: float, th: float, n_trials: int) -> float:
    """Simulate decisions for a single-agent posture and recover the threat ratio omega/rho."""
    rho = 1.0
    theta = {"rho": rho, "lambda": la * rho, "omega": th * rho,
             "kappa": np.clip(rng.normal(0.6, 0.2), 0.05, None) * rho,
             "delta": np.clip(rng.normal(0.5, 0.2), 0.05, None) * rho,
             "tau": np.clip(rng.normal(1.3, 0.2), 0.1, None)}
    X = P.sample_contexts(rng, n_trials)
    y = P.simulate_agent_choices(rng, X, theta)
    beta = P.fit_agent(X, y)
    return P.recover_ratios(beta)["threat"]


def run_e7(n_entities: int = 60, n_trials: int = 1500, n_seeds: int = 10) -> Dict[str, object]:
    def condition(true_shift_dist):
        rec_shifts, true_shifts = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(9000 + s)
            for _ in range(n_entities):
                la = rng.uniform(1.4, 2.4)
                th_base = rng.uniform(0.5, 1.1)
                shift = true_shift_dist(rng)              # true threat displacement under state
                th_state = np.clip(th_base + shift, 0.05, None)
                r_base = _posture_threat(rng, la, th_base, n_trials)
                r_state = _posture_threat(rng, la, th_state, n_trials)
                rec_shifts.append(r_state - r_base)
                true_shifts.append(shift)
        rec = np.array(rec_shifts); tru = np.array(true_shifts)
        ci = H.bootstrap_ci(rec, statistic=np.mean, seed=0)
        r_track = float(stats.pearsonr(tru, rec)[0]) if tru.std() > 1e-6 else None
        return {"mean_recovered_shift": float(rec.mean()), "mean_recovered_ci95": [ci["lo"], ci["hi"]],
                "shift_tracking_r": r_track, "detects_shift": bool(ci["lo"] > 0)}

    pos = condition(lambda rng: rng.uniform(0.4, 1.0))     # genuine threat elevation
    neg = condition(lambda rng: 0.0)                        # no state shift
    return {
        "experiment": "E7_state_shift_recovery",
        "design": {"n_entities": n_entities, "n_trials": n_trials, "n_seeds": n_seeds},
        "framing": "states as recoverable within-entity manifold displacements; simulation of mechanism",
        "positive_control": pos,
        "negative_control": neg,
        "valid": bool(pos["detects_shift"] and (pos["shift_tracking_r"] or 0) > 0.3
                      and not neg["detects_shift"]),
    }


if __name__ == "__main__":
    import json

    res = run_e7()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "results", "e7_state_shift.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    p, n = res["positive_control"], res["negative_control"]
    print("E7 state-shift recovery")
    print(f"  positive (threat elevated): mean recovered shift={p['mean_recovered_shift']:+.3f} "
          f"CI[{p['mean_recovered_ci95'][0]:+.3f},{p['mean_recovered_ci95'][1]:+.3f}] "
          f"track r={p['shift_tracking_r']:.3f} detects={p['detects_shift']}")
    print(f"  negative (no shift):        mean recovered shift={n['mean_recovered_shift']:+.3f} "
          f"CI[{n['mean_recovered_ci95'][0]:+.3f},{n['mean_recovered_ci95'][1]:+.3f}] detects={n['detects_shift']}")
    print("  valid:", res["valid"], "->", path)
