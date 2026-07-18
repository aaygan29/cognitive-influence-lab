"""
evaluate.py — the metrics that actually matter for a decision tool.

Accuracy alone is the polygraph's marketing mistake. We report:
  * AUC on the calibrated probability (ranking quality, independent of threshold)
  * Brier score + calibration curve (does p mean what it says?)
  * Confusion at the DECISION level over non-abstained cases, with the
    false-positive rate broken out (falsely calling a naive subject informed —
    the ethically costly error)
  * A coverage/accuracy trade-off: as we allow more abstention, accuracy on the
    cases we DO decide should rise. This quantifies the value of "knowing when
    you don't know."
  * Countermeasure handling: what fraction of countermeasure cases are correctly
    abstained rather than mis-cleared.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def calibration_table(p, y, n_bins=10):
    """Reliability curve: mean predicted vs observed frequency per probability bin."""
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin_lo": lo, "bin_hi": hi, "mean_pred": p[m].mean(),
                     "obs_freq": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def ranking_metrics(decisions: pd.DataFrame) -> dict:
    """AUC and Brier on ALL cases (probability quality, pre-abstention)."""
    y = decisions["concealed"].to_numpy()
    p = decisions["p_informed"].to_numpy()
    return {"auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "n": int(len(y))}


def decision_metrics(decisions: pd.DataFrame) -> dict:
    """Confusion and error rates over the cases we actually decided."""
    d = decisions
    decided = d[d.action.isin(["INFORMED", "NAIVE"])]
    n_total = len(d)
    n_decided = len(decided)
    coverage = n_decided / n_total if n_total else 0.0
    if n_decided == 0:
        return {"coverage": 0.0, "accuracy": float("nan"), "fpr": float("nan"),
                "fnr": float("nan"), "n_decided": 0, "n_total": n_total}
    pred_pos = (decided.action == "INFORMED").to_numpy()
    truth = decided.concealed.to_numpy().astype(bool)
    tp = int(np.sum(pred_pos & truth)); fp = int(np.sum(pred_pos & ~truth))
    tn = int(np.sum(~pred_pos & ~truth)); fn = int(np.sum(~pred_pos & truth))
    acc = (tp + tn) / n_decided
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")  # naive called informed
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")  # informed called naive
    return {"coverage": coverage, "accuracy": acc, "fpr": fpr, "fnr": fnr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_decided": n_decided, "n_total": n_total}


def coverage_accuracy_curve(decisions: pd.DataFrame, n_points=25):
    """Selective-prediction curve. Rank cases by confidence = |p-0.5| and, for
    each coverage level, report accuracy if we only answered the most confident
    fraction. Shows the accuracy 'purchased' by abstaining."""
    d = decisions.copy()
    d["conf"] = (d["p_informed"] - 0.5).abs()
    d = d.sort_values("conf", ascending=False).reset_index(drop=True)
    y = d["concealed"].to_numpy().astype(bool)
    pred = (d["p_informed"] >= 0.5).to_numpy()
    rows = []
    n = len(d)
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(1, int(round(frac * n)))
        acc = float(np.mean(pred[:k] == y[:k]))
        rows.append({"coverage": k / n, "accuracy": acc})
    return pd.DataFrame(rows)


def countermeasure_report(decisions: pd.DataFrame) -> dict:
    """How countermeasure cases are handled. The desired behavior is to ABSTAIN
    (not to be silently mis-cleared as NAIVE)."""
    if "used_countermeasure" not in decisions.columns:
        return {}
    cm = decisions[decisions.used_countermeasure == 1]
    if len(cm) == 0:
        return {"n_cm": 0}
    return {
        "n_cm": int(len(cm)),
        "frac_abstained": float((cm.action == "ABSTAIN").mean()),
        "frac_miscleared_naive": float((cm.action == "NAIVE").mean()),
        "frac_correct_informed": float((cm.action == "INFORMED").mean()),
    }
