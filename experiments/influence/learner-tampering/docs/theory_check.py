"""Numerical check of the two propositions in docs/THEORY.md, on the mean-field (expected-update)
learner so stochastic choice noise does not blur the bound.

P1 (value absorption): for a bonus schedule b_t on one side, the bias it installs relative to the
    no-bonus learner obeys |a_T - a_T^0| <= (eta/alpha) * TV(b), TV(b) = |b_0| + sum_t |b_t - b_{t-1}|,
    independent of total dose sum_t b_t.
P2 (fixed-point shift): a curriculum with right fraction rho moves the learner's fixed point to a bias
    of the sign of rho - 1/2 that persists while rho is held, i.e. it is not absorbed.
"""
import json
import numpy as np

ETA, ALPHA, MAGS = 0.05, 0.05, np.array([0.25, 0.5, 1.0, 2.0])


def mean_field(bonus, rho=0.5, T=None, a0=0.0, b0=0.9):
    T = len(bonus) if T is None else T
    a, b, Q = a0, b0, np.array([0.5, 0.5])
    for t in range(T):
        bt = bonus[t] if t < len(bonus) else 0.0
        da = db = 0.0; dQ = np.zeros(2); w_tot = 0.0
        for x_sign, w_side in ((1, rho), (-1, 1 - rho)):
            for m in MAGS:
                x = x_sign * m; w = w_side / len(MAGS)
                p = 1 / (1 + np.exp(-(a + b * x)))
                for y, py in ((1, p), (0, 1 - p)):
                    r = float((x > 0) == (y == 1)) + (bt if y == 1 else 0.0)
                    d = r - Q[y]; s = 2 * y - 1
                    da += w * py * ETA * d * s; db += w * py * ETA * d * s * x
                    dQ[y] += w * py * ALPHA * d
        a += da; b += db; Q = Q + dQ
    return a


def tv(bonus):
    return abs(bonus[0]) + float(np.sum(np.abs(np.diff(bonus))))


rng = np.random.default_rng(0)
T = 480
base = mean_field(np.zeros(T))
rows = []
for i in range(60):
    kind = i % 3
    if kind == 0:   # constant
        sched = np.full(T, rng.uniform(0.1, 1.0))
    elif kind == 1:  # block pulses
        on = rng.uniform(0.2, 1.0); period = int(rng.choice([20, 40, 80]))
        sched = np.array([on if (t // period) % 2 == 0 else 0.0 for t in range(T)])
    else:            # random walk
        sched = np.clip(np.cumsum(rng.normal(0, 0.05, T)) + 0.3, 0, 1)
    shift = mean_field(sched) - base
    rows.append({"kind": ["constant", "pulse", "walk"][kind], "dose": float(sched.sum()), "tv": tv(sched),
                 "shift": float(shift), "bound": ETA / ALPHA * tv(sched)})
viol = [r for r in rows if abs(r["shift"]) > r["bound"] + 1e-9]
corr_dose = float(np.corrcoef([r["dose"] for r in rows], [abs(r["shift"]) for r in rows])[0, 1])
corr_tv = float(np.corrcoef([r["tv"] for r in rows], [abs(r["shift"]) for r in rows])[0, 1])
cur = {str(rho): float(mean_field(np.zeros(T), rho=rho)) for rho in (0.5, 0.65, 0.8)}
cur_wash = {str(rho): float(mean_field(np.zeros(T + 240), rho=rho)) for rho in (0.8,)}
out = {"P1_violations": len(viol), "n": len(rows), "max_shift_over_bound": max(abs(r["shift"]) / r["bound"] for r in rows),
       "corr(|shift|, dose)": corr_dose, "corr(|shift|, TV)": corr_tv, "P2_bias_at_rho": cur, "rows": rows}
json.dump(out, open("docs/theory_check.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
