"""Level 1: a momentary belief under an influence controller, with a critical-slowing-down early
warning. Simulation only.

A bistable belief x sits in a double-well potential U(x; a) = x^4/4 - x^2/2 - a x, so there are two
stable states near x = -1 and x = +1 and a barrier between them. An influence controller raises the
tilt a, which lowers the barrier on the -1 side; past the fold a_c = 2/(3*sqrt(3)) ~= 0.385 the -1
state disappears and the belief flips to +1. This is the acute-persuasion level: push the ball over
the hill.

Critical slowing down (Scheffer et al., Nature 2009) predicts that as the barrier flattens, the
belief's fluctuations get larger and more autocorrelated BEFORE it flips. We test whether a monitor
that watches the rolling variance and lag-1 autocorrelation of x gives a positive lead time before
the crossing, and whether it stays quiet when the controller holds the belief safely away from the
fold (the calibration null). The monitor sees only x, as a real defender would.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

A_C = 2 / (3 * np.sqrt(3))   # fold of the double well


def _tau(y):
    """Kendall rank correlation of y against time; the standard early-warning trend statistic."""
    n = len(y)
    if n < 8:
        return 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            c += np.sign(y[j] - y[i])
    return c / (n * (n - 1) / 2)


def simulate(seed, approach=True, D=0.012, dt=0.01, T=8000, win=150, sub=40, trend_n=25, tau_crit=0.80):
    """Belief x in U(x;a)=x^4/4 - x^2/2 - a x. Monitor watches the trend (Kendall tau) of the rolling
    variance of x; warns when the trend over the last `trend_n` subsampled points exceeds tau_crit,
    the signature of approach to the fold (Scheffer 2009; trend method Dakos 2012)."""
    rng = np.random.default_rng(seed)
    x = -1.0
    ramp_to = 0.37 if approach else 0.20
    ramp_end = int(T * 0.5)
    xs = np.empty(T); avar = np.full(T, np.nan)
    buf, vseries = [], []
    cross_t = warn_t = None
    for t in range(T):
        a = ramp_to * min(1.0, t / ramp_end) if approach else ramp_to  # null: constant tilt, stationary
        x += (-(x ** 3 - x - a)) * dt + np.sqrt(2 * D * dt) * rng.normal()
        xs[t] = x
        buf.append(x)
        if len(buf) > win:
            buf.pop(0)
        if len(buf) == win:
            v = np.var(buf); avar[t] = v
            if t % sub == 0:
                vseries.append(v)
                if warn_t is None and len(vseries) >= trend_n:
                    if _tau(vseries[-trend_n:]) >= tau_crit:
                        warn_t = t
        if cross_t is None and x > 0.0:
            cross_t = t
    return {"x": xs, "var": avar, "cross_t": cross_t, "warn_t": warn_t, "a_target": ramp_to}


def _cumtau_warn(r, start_k=6, sub=40, tau_crit=0.6):
    v = r["var"]; idx = np.where(~np.isnan(v))[0]
    vs = v[idx][::sub]; ts = idx[::sub]
    for k in range(start_k, len(vs)):
        if _tau(vs[:k]) >= tau_crit:
            return ts[k]
    return None


def main():
    HERE = Path(__file__).parent
    N = 150
    table = []
    for tc in (0.4, 0.5, 0.6, 0.7):
        sens = crossed = 0; leads = []
        for s in range(N):
            r = simulate(s, approach=True)
            if r["cross_t"]:
                crossed += 1
                w = _cumtau_warn(r, tau_crit=tc)
                if w and w < r["cross_t"]:
                    sens += 1; leads.append((r["cross_t"] - w) * 0.01)
        fp = sum(_cumtau_warn(simulate(1000 + s, approach=False), tau_crit=tc) is not None for s in range(N))
        table.append({"tau_crit": tc, "sensitivity": round(sens / crossed, 3) if crossed else None,
                      "false_alarm_rate": round(fp / N, 3),
                      "median_lead_time": round(float(np.median(leads)), 2) if leads else None})
    # exemplar at the clean operating point (tau_crit 0.6)
    ex = None
    for s in range(N):
        r = simulate(s, approach=True)
        w = _cumtau_warn(r, tau_crit=0.6)
        if r["cross_t"] and w and w < r["cross_t"]:
            ex = {"x": r["x"].tolist(), "var": r["var"].tolist(), "cross_t": int(r["cross_t"]), "warn_t": int(w)}
            break
    out = {"a_c": round(float(A_C), 4), "n": N, "method": "Kendall-tau trend of rolling variance (Scheffer 2009; Dakos 2012)",
           "operating_points": table}
    (HERE / "belief_csd.json").write_text(json.dumps({"summary": out, "exemplar": ex}))
    print(json.dumps(out, indent=1))




if __name__ == "__main__":
    main()
