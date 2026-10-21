"""An improved, literature-grounded integrated composite for neuroforecasting, tested honestly.

What the literature says (and why each change is made):
- For AGGREGATE forecasting, only anticipatory-affect regions generalize: nucleus accumbens (approach,
  positive) and anterior insula (avoidance, negative). Medial prefrontal cortex predicts INDIVIDUAL choice
  but does not generalize to aggregate outcomes (Genevsky et al. 2017; Tong et al. 2020). So the robust
  aggregate composite is approach minus avoidance, NAcc - AIns, and adding MPFC can hurt generalization.
- These predictors are replicable and generalize across tasks and samples (Mortazavi et al. 2025, N=230,
  triple dissociation of NAcc/MPFC approach vs AIns avoidance).
- For small, correlated samples, ridge shrinkage beats lasso, stepwise, and OLS; parsimony wins the
  bias-variance tradeoff (methods-comparison literature).

We therefore compare, out-of-sample across the two experiments, with permutation nulls:
  M0 NAcc alone                     (Genevsky classic single region)
  M1 E+V-R  = NAcc+MPFC-AIns        (the prior composite)
  M2 affect = NAcc-AIns             (literature-preferred aggregate composite, MPFC dropped)
  M3 ridge on all ROI x TR features (full multivariate, expected to overfit at n=274)
  M4 ridge on ROI window-means      (moderate multivariate)
  M5 shrinkage ensemble             (mean of M2 fixed composite and M4 ridge; bias-variance compromise)

The honest question is which one GENERALIZES best out-of-sample, not which fits best in-sample.

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "raw", "knutson", "knutson_stock.csv")
RESULTS = os.path.join(ROOT, "results", "composite_model.json")

WINDOW = list(range(3, 9))  # a priori anticipatory window, fixed
BILATERAL = {"NAcc": "nacc8mmb", "MPFC": "mpfcb", "AIns": "desai_insb"}
ALL_ROIS = ["nacc8mmb", "nacc8mml", "nacc8mmr", "mpfcb", "mpfcl", "mpfcr",
            "desai_insb", "desai_insl", "desai_insr"]


def _z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def load():
    df = pd.read_csv(DATA)
    out = {"Choice": df["Choice"].values.astype(float), "Experiment": df["Experiment"].values.astype(int)}
    # window-mean per bilateral ROI
    for name, base in BILATERAL.items():
        cols = [f"{base}_TR_{t}" for t in WINDOW if f"{base}_TR_{t}" in df.columns]
        out[name] = df[cols].mean(axis=1).values
    # full ROI x TR feature matrix (all subregions, all 16 TRs)
    full_cols = [c for c in df.columns if any(c.startswith(r + "_TR_") for r in ALL_ROIS)]
    out["FULL"] = df[full_cols].values
    # ROI window-means for all subregions (moderate feature set)
    mean_cols = {}
    for r in ALL_ROIS:
        cs = [f"{r}_TR_{t}" for t in WINDOW if f"{r}_TR_{t}" in df.columns]
        mean_cols[r] = df[cs].mean(axis=1).values
    out["ROIMEANS"] = np.column_stack([mean_cols[r] for r in ALL_ROIS])
    return out


def _fixed_pred(d, kind):
    if kind == "NAcc":
        return _z(d["NAcc"])
    if kind == "EVR":
        return _z(d["NAcc"]) + _z(d["MPFC"]) - _z(d["AIns"])
    if kind == "affect":
        return _z(d["NAcc"]) - _z(d["AIns"])
    raise ValueError(kind)


def _oos_scores(d, predict_fn):
    """Cross-experiment out-of-sample. predict_fn(train_mask, test_mask) -> predictions on test."""
    exp = d["Experiment"]; y = d["Choice"]
    rs, aucs = [], []
    for tr_e, te_e in [(1, 2), (2, 1)]:
        tr, te = exp == tr_e, exp == te_e
        pred = predict_fn(tr, te)
        rs.append(stats.pearsonr(pred, y[te])[0])
        ybin = (y[te] > np.median(y)).astype(int)
        aucs.append(roc_auc_score(ybin, pred) if len(np.unique(ybin)) > 1 else np.nan)
    return float(np.nanmean(rs)), float(np.nanmean(aucs))


def _make_predict_fn(d, model):
    y = d["Choice"]
    if model in ("NAcc", "EVR", "affect"):
        x = _fixed_pred(d, model)
        def f(tr, te):
            reg = LinearRegression().fit(x[tr].reshape(-1, 1), y[tr])
            return reg.predict(x[te].reshape(-1, 1))
        return f
    if model in ("ridge_full", "ridge_means"):
        X = d["FULL"] if model == "ridge_full" else d["ROIMEANS"]
        def f(tr, te):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            reg = Ridge(alpha=10.0).fit(Xtr, y[tr])
            return reg.predict(Xte)
        return f
    if model == "ensemble":
        xa = _fixed_pred(d, "affect")
        X = d["ROIMEANS"]
        def f(tr, te):
            r1 = LinearRegression().fit(xa[tr].reshape(-1, 1), y[tr]).predict(xa[te].reshape(-1, 1))
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            r2 = Ridge(alpha=10.0).fit((X[tr]-mu)/sd, y[tr]).predict((X[te]-mu)/sd)
            return 0.5 * _z(r1) + 0.5 * _z(r2)
        return f
    raise ValueError(model)


def _perm_p(d, model, observed_r, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    exp = d["Experiment"]
    base_y = d["Choice"].copy()
    null = []
    for _ in range(n_perm):
        d2 = dict(d); d2["Choice"] = rng.permutation(base_y)
        r, _ = _oos_scores(d2, _make_predict_fn(d2, model))
        null.append(r)
    null = np.array(null)
    return float((null >= observed_r).mean())


def window_robustness(seed=0):
    """Is the affect composite significant across a range of anticipatory windows, not just one?

    Reports the out-of-sample r of the affect composite (NAcc - AIns) for several fixed windows. All are
    reported; no window is selected after the fact. Robustness = the effect holds across windows.
    """
    df = pd.read_csv(DATA)
    y = df["Choice"].values.astype(float); exp = df["Experiment"].values.astype(int)
    windows = {"early_2_5": range(2, 6), "mid_3_8": range(3, 9), "late_5_10": range(5, 11),
               "wide_2_12": range(2, 13)}
    out = {}
    for name, w in windows.items():
        def roi(base):
            cs = [f"{base}_TR_{t}" for t in w if f"{base}_TR_{t}" in df.columns]
            return df[cs].mean(axis=1).values
        pred_full = _z(roi("nacc8mmb")) - _z(roi("desai_insb"))
        rs = []
        for tr_e, te_e in [(1, 2), (2, 1)]:
            tr, te = exp == tr_e, exp == te_e
            reg = LinearRegression().fit(pred_full[tr].reshape(-1, 1), y[tr])
            rs.append(stats.pearsonr(reg.predict(pred_full[te].reshape(-1, 1)), y[te])[0])
        out[name] = float(np.nanmean(rs))
    return out


def main():
    d = load()
    models = {
        "M0_NAcc": "NAcc",
        "M1_EVR": "EVR",
        "M2_affect_NAcc_minus_AIns": "affect",
        "M3_ridge_full": "ridge_full",
        "M4_ridge_ROImeans": "ridge_means",
        "M5_shrinkage_ensemble": "ensemble",
    }
    out = {"dataset": "Knutson neuroforecasting (Kuhnen and Knutson 2005)", "n": int(len(d["Choice"])),
           "window_TRs": WINDOW, "models": {}}
    for label, m in models.items():
        r, auc = _oos_scores(d, _make_predict_fn(d, m))
        p = _perm_p(d, m, r)
        out["models"][label] = {"oos_r": r, "oos_auc": auc, "perm_p": p}
    # pick the best generalizer
    best = max(out["models"].items(), key=lambda kv: kv[1]["oos_r"])
    out["best_generalizer"] = {"model": best[0], **best[1]}
    out["window_robustness_affect"] = window_robustness()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2)

    print(f"n={out['n']}  window={WINDOW}")
    for label, v in out["models"].items():
        print(f"  {label:28s} oos_r={v['oos_r']:+.3f}  auc={v['oos_auc']:.3f}  perm_p={v['perm_p']:.4f}")
    print(f"  BEST GENERALIZER: {out['best_generalizer']['model']} (oos_r={out['best_generalizer']['oos_r']:+.3f})")
    print("wrote", RESULTS)
    return out


if __name__ == "__main__":
    main()
