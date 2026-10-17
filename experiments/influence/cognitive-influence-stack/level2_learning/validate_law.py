r"""Validate the installed-bias law in five independent ways. SIMULATION ONLY.

    theta_install(b, rho)  ~=  (eta/alpha) * e_bar(b)  +  g(rho)

This law is a statement about a synthetic learner. We do not fit it to, or test it on, human
behavioural or neural data: doing so is the capability this project exists to warn about, not to
build. The checks below use only learners we wrote, whose ground-truth learning rule we know.

Checks (each prints one number):
  1 superposition: the two channels add, up to a small saturating residual.
  2 reward coefficient: the reward channel scales with eta/alpha across a 4x range of the ratio.
  3 bound: the reward channel never exceeds (eta/alpha) sup|b| over 2000 random schedules.
  4 curriculum independence: g(rho) does not depend on the bonus.
  5 realism: how well the law predicts the full stochastic learner (honest: it under-predicts the
    curriculum channel; the defensive BOUND in check 3 is what is exact, not this prediction).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from learner import Learner  # noqa: E402

MAGS = np.array([0.25, 0.5, 1.0, 2.0])


def mf(bonus, rho, eta=0.05, al=0.05, T=800):
    a, b, Q, QR0 = 0.0, 0.9, np.array([0.5, 0.5]), 0.5
    for _ in range(T):
        da = db = 0.0; dQ = np.zeros(2); dQR0 = 0.0
        for xs, ws in ((1, rho), (-1, 1 - rho)):
            for m in MAGS:
                x = xs * m; w = ws / 4; p = 1 / (1 + np.exp(-(a + b * x)))
                for y, py in ((1, p), (0, 1 - p)):
                    rt = float((x > 0) == (y == 1)); r = rt + (bonus if y == 1 else 0.0)
                    d = r - Q[y]; s = 2 * y - 1
                    da += w * py * eta * d * s; db += w * py * eta * d * s * x; dQ[y] += w * py * al * d
                    if y == 1:
                        dQR0 += w * py * al * (rt - QR0)
        a += da; b += db; Q = Q + dQ; QR0 += dQR0
    return a, Q[1] - QR0


def mf_sched(bseq, rho, eta=0.05, al=0.05):
    a, b, Q = 0.0, 0.9, np.array([0.5, 0.5])
    for bt in bseq:
        da = db = 0.0; dQ = np.zeros(2)
        for xs, ws in ((1, rho), (-1, 1 - rho)):
            for m in MAGS:
                x = xs * m; w = ws / 4; p = 1 / (1 + np.exp(-(a + b * x)))
                for y, py in ((1, p), (0, 1 - p)):
                    r = float((x > 0) == (y == 1)) + (bt if y == 1 else 0.0); d = r - Q[y]; s = 2 * y - 1
                    da += w * py * eta * d * s; db += w * py * eta * d * s * x; dQ[y] += w * py * al * d
        a += da; b += db; Q = Q + dQ
    return a


def stoch(bonus, rho, seed, T=480):
    rng = np.random.default_rng(seed); L = Learner(rng, "plastic", 0.0)
    L.a, L.b, L.eta, L.alpha, L.Q = 0.0, 0.9, 0.05, 0.05, np.array([0.5, 0.5])
    side = "right" if bonus > 0 else "none"
    for _ in range(T):
        L.trial(float(np.random.default_rng(seed * 1000 + _).choice(MAGS)) * (1 if rng.random() < rho else -1), side, bonus)
    return L.a


def main():
    out = {"simulation_only": True}
    base, _ = mf(0, 0.5)
    r1 = max(abs((mf(bo, rho)[0] - base) - ((mf(bo, 0.5)[0] - base) + (mf(0, rho)[0] - base)))
             for bo in (0, 0.3, 0.6) for rho in (0.5, 0.65, 0.8))
    out["check1_superposition_max_residual"] = round(r1, 3)

    c2 = []
    for eta, al in ((0.05, 0.05), (0.08, 0.04), (0.03, 0.06), (0.10, 0.05)):
        a, e = mf(0.4, 0.5, eta, al); c2.append({"eta_over_alpha": round(eta / al, 2), "coeff": round(a / e, 3)})
    out["check2_reward_coeff_vs_ratio"] = c2

    rng = np.random.default_rng(0); viol = 0; worst = 0.0
    for _ in range(2000):
        kind = rng.integers(3); T = int(rng.integers(100, 600))
        if kind == 0:
            bseq = np.full(T, rng.uniform(0, 1))
        elif kind == 1:
            per = int(rng.integers(5, 80)); bseq = np.array([rng.uniform(0, 1) if (t // per) % 2 == 0 else 0.0 for t in range(T)])
        else:
            bseq = np.clip(np.cumsum(rng.normal(0, 0.1, T)) + rng.uniform(0, 0.5), 0, 1)
        direct = mf_sched(bseq, 0.5); bound = max(np.abs(bseq))
        viol += abs(direct) > bound + 1e-9; worst = max(worst, abs(direct) / (bound + 1e-12))
    out["check3_bound"] = {"violations": int(viol), "n": 2000, "tightest_ratio": round(float(worst), 3)}

    c4 = {}
    for rho in (0.65, 0.8):
        gs = [mf(bo, rho)[0] - mf(bo, 0.5)[0] for bo in (0, 0.3, 0.6)]
        c4[str(rho)] = {"g_at_bonus_0_0.3_0.6": [round(g, 3) for g in gs], "spread": round(max(gs) - min(gs), 3)}
    out["check4_curriculum_independent_of_bonus"] = c4

    c5 = []
    for bo, rho in ((0.4, 0.5), (0.0, 0.8), (0.4, 0.7)):
        emp = float(np.mean([stoch(bo, rho, s) for s in range(200)]))
        pred = mf(bo, 0.5)[0] + mf(0, rho)[0]
        c5.append({"bonus": bo, "rho": rho, "stochastic": round(emp, 3), "law": round(pred, 3), "diff": round(emp - pred, 3)})
    out["check5_vs_stochastic_learner"] = c5
    Path(__file__).with_name("validate_law.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
