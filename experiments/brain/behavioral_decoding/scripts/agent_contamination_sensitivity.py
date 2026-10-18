"""Reviewer action #4: agent-contamination sensitivity for the aggregate arm.

AGEA (Ozudogru et al. 2026) shows a consumer LLM agent can complete online
behavioural experiments with human-range behaviour. The market/cohort arms here
(y_aggregate -- the Kiva arm) draw on online populations, so a fraction of
"participants" may be agents whose features look plausible but whose outcome does
not follow the human feature->outcome law. This treats that contamination
fraction f as a nuisance parameter and measures how the de-leaked Kiva decode
degrades with it.

Contamination model: inject agent rows until a fraction f of the dataset is
agents. Agent feature rows are bootstrapped from the real applicants (so marginals
and correlations look human, as AGEA agents do), but their label is drawn
independently at the base rate -- the agent's "outcome" carries no genuine
feature->outcome signal. The full (real + agent) set then goes through the same
subject-grouped... here loan-level StratifiedKFold decode. Both train and test
are contaminated, because agents are indistinguishable participants.

Reported: balanced accuracy vs f, and the contamination fraction at which the
decode is no longer distinguishable from chance (0.5).

    python scripts/agent_contamination_sensitivity.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "agent_contamination_sensitivity"
SEEDS = list(range(25))
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def load_kiva_clean():
    rows = list(csv.DictReader(open(ROOT / "data" / "raw" / "kiva_kenya.csv")))
    sectors = sorted({r["sector"] for r in rows})
    y = np.array([1 if r["status"] == "1" else 0 for r in rows])
    X = []
    for r in rows:
        X.append([float(r["loan_amount"] or 0),
                  1.0 if r["gender"].upper().startswith("F") else 0.0,
                  float(len(r["en"]))] + [1.0 if r["sector"] == s else 0.0 for s in sectors])
    return np.array(X), y


def contaminate(X, y, f, rng):
    """Add agent rows until fraction f of the result is agents."""
    n = len(y)
    if f <= 0:
        return X, y, np.zeros(n, bool)
    n_agent = int(round(f / (1 - f) * n))
    idx = rng.integers(0, n, size=n_agent)          # bootstrap real feature rows
    Xa = X[idx].copy()
    base = y.mean()
    ya = (rng.random(n_agent) < base).astype(int)    # label independent of features
    Xc = np.vstack([X, Xa])
    yc = np.concatenate([y, ya])
    is_agent = np.concatenate([np.zeros(n, bool), np.ones(n_agent, bool)])
    return Xc, yc, is_agent


def decode(X, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        out.append(balanced_accuracy_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(out))


def main():
    X, y = load_kiva_clean()
    curve = {}
    for f in FRACTIONS:
        vals = []
        for s in SEEDS:
            rng = np.random.default_rng(1000 + s)
            Xc, yc, _ = contaminate(X, y, f, rng)
            vals.append(decode(Xc, yc, s))
        m, sd = float(np.mean(vals)), float(np.std(vals))
        curve[f] = {"balacc_mean": round(m, 3), "balacc_sd": round(sd, 3)}
        print(f"f={f:.2f}  balanced_acc={m:.3f} +/- {sd:.3f}")

    clean = curve[0.0]["balacc_mean"]
    # contamination tolerance: smallest f whose mean is within 1 sd of chance 0.5
    tol = None
    for f in FRACTIONS:
        if curve[f]["balacc_mean"] - curve[f]["balacc_sd"] <= 0.5:
            tol = f; break
    # half-signal point: f where lift over chance halves
    lift0 = clean - 0.5
    half = None
    for f in FRACTIONS:
        if (curve[f]["balacc_mean"] - 0.5) <= lift0 / 2:
            half = f; break

    res = {"arm": "Kiva aggregate (de-leaked)", "clean_balacc": clean,
           "fractions": FRACTIONS, "curve": curve,
           "contamination_to_chance_f": tol, "half_signal_f": half,
           "model": "agent rows: bootstrapped real features, label ~ base rate (no feature->outcome link); "
                    "train+test contaminated; balanced accuracy, 25 seeds",
           "reference": "Ozudogru et al. 2026, AGEA (agents threaten online behavioral experiments)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sensitivity.json").write_text(json.dumps(res, indent=2))
    print("\nhalf-signal at f =", half, "| indistinguishable-from-chance at f =", tol)
    print(json.dumps(res["curve"], indent=2))


if __name__ == "__main__":
    main()
