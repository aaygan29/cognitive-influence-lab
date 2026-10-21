"""Real neural-to-behavior test: does the E+V-R signature, computed from real fMRI, predict real choice?

Data: the Knutson stock-investment neuroforecasting task (Kuhnen and Knutson 2005; Knutson lineage),
obtained from the author's neurobridge repository. Each row is a stock with the group-averaged BOLD
timeseries (16 TRs) in three regions of interest and the aggregate behavioral outcome Choice, the fraction
of participants who chose to buy. Two experiments (1 and 2) allow a genuine out-of-sample test: fit on one
experiment, predict the other.

This is the real-fMRI analogue of the text-proxy signature. The regions map onto the signature channels:
  E engagement = MPFC        (mpfcb)
  V value      = NAcc        (nacc8mmb)
  R resistance = anterior insula (desai_insb), the avoidance signal that opposes buying (Knutson 2007)
The signature is B = z(E) + z(V) - z(R), the SAME structure as the text instrument, now on real neural
activity. The behavioral outcome is real aggregate choice. This is the neural phenomenon predicting
behavior that the program is built to test.

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "raw", "knutson", "knutson_stock.csv")
RESULTS = os.path.join(ROOT, "results", "neuralforecast.json")

# a priori anticipation window (hemodynamic peak, TR ~ 2s): TRs 3..8. Fixed before looking at outcomes.
WINDOW = list(range(3, 9))
ROI = {"E": "mpfcb", "V": "nacc8mmb", "R": "desai_insb"}


def _roi_activation(df, base):
    cols = [f"{base}_TR_{t}" for t in WINDOW if f"{base}_TR_{t}" in df.columns]
    return df[cols].mean(axis=1).values


def load():
    df = pd.read_csv(DATA)
    out = {"Choice": df["Choice"].values.astype(float), "Experiment": df["Experiment"].values.astype(int)}
    for ch, base in ROI.items():
        out[ch] = _roi_activation(df, base)
    return out


def _z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def signature(d, idx=None):
    E, V, R = d["E"], d["V"], d["R"]
    if idx is not None:
        E, V, R = E[idx], V[idx], R[idx]
    return _z(E) + _z(V) - _z(R)


def cross_experiment(d, predictor="B"):
    """Fit predictor -> Choice on one experiment, predict the other. Report out-of-sample r and AUC."""
    exp = d["Experiment"]
    res = {}
    rs, aucs = [], []
    for train_e, test_e in [(1, 2), (2, 1)]:
        tr = exp == train_e
        te = exp == test_e
        if predictor == "B":
            xtr, xte = signature(d)[tr], signature(d)[te]
        else:
            xtr, xte = _z(d[predictor])[tr], _z(d[predictor])[te]
        ytr, yte = d["Choice"][tr], d["Choice"][te]
        reg = LinearRegression().fit(xtr.reshape(-1, 1), ytr)
        pred = reg.predict(xte.reshape(-1, 1))
        r = stats.pearsonr(pred, yte)[0]
        ybin = (yte > np.median(d["Choice"])).astype(int)
        auc = roc_auc_score(ybin, pred) if len(np.unique(ybin)) > 1 else np.nan
        rs.append(r); aucs.append(auc)
        res[f"E{train_e}->E{test_e}"] = {"pearson_r": float(r), "auc": float(auc)}
    res["mean_r"] = float(np.nanmean(rs))
    res["mean_auc"] = float(np.nanmean(aucs))
    # permutation null on the pooled out-of-sample r
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        yperm = rng.permutation(d["Choice"])
        rr = []
        for train_e, test_e in [(1, 2), (2, 1)]:
            tr = exp == train_e; te = exp == test_e
            xtr = signature(d)[tr] if predictor == "B" else _z(d[predictor])[tr]
            xte = signature(d)[te] if predictor == "B" else _z(d[predictor])[te]
            reg = LinearRegression().fit(xtr.reshape(-1, 1), yperm[tr])
            rr.append(stats.pearsonr(reg.predict(xte.reshape(-1, 1)), yperm[te])[0])
        null.append(np.nanmean(rr))
    null = np.array(null)
    res["perm_p"] = float((null >= res["mean_r"]).mean())
    res["null_mean_r"] = float(null.mean())
    return res


def per_experiment_stability(d):
    """In-sample correlation of each predictor with Choice, per experiment, to expose sign stability."""
    out = {}
    for e in (1, 2):
        m = d["Experiment"] == e
        row = {}
        for ch in ("E", "V", "R"):
            row[ch] = float(stats.pearsonr(d[ch][m], d["Choice"][m])[0])
        row["B"] = float(stats.pearsonr(signature(d)[m], d["Choice"][m])[0])
        out[f"experiment_{e}"] = row
    return out


def main():
    d = load()
    out = {
        "dataset": "Knutson stock neuroforecasting (Kuhnen and Knutson 2005), via neurobridge",
        "n_stocks": int(len(d["Choice"])),
        "window_TRs": WINDOW,
        "roi_map": ROI,
        "signature_B": cross_experiment(d, "B"),
        "NAcc_only": cross_experiment(d, "V"),
        "MPFC_only": cross_experiment(d, "E"),
        "insula_only": cross_experiment(d, "R"),
        "per_experiment_stability": per_experiment_stability(d),
        "caveat": "Effect is near-null in experiment 1 and strong in experiment 2; the composite is the "
                  "only predictor sign-stable across both, but the result is weak and experiment-dependent.",
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2)

    print(f"n stocks={out['n_stocks']}  window TRs={WINDOW}")
    for key in ["signature_B", "NAcc_only", "MPFC_only", "insula_only"]:
        r = out[key]
        print(f"  {key:12s} out-of-sample r={r['mean_r']:+.3f} auc={r['mean_auc']:.3f} "
              f"perm_p={r['perm_p']:.4f} (null r={r['null_mean_r']:+.3f})")
    print("wrote", RESULTS)
    return out


if __name__ == "__main__":
    main()
