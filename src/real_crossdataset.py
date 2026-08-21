"""Cross-dataset real-data test: does the decision phenotype transfer across labs and datasets?

We pool two independent mixed-gambles fMRI datasets, both public:
- NARPS ds001734 (Botvinik-Nezer 2019): 108 subjects.
- Tom, Fox, Trepel & Poldrack (2007) ds000005: 16 subjects, the original loss-aversion study.

Same paradigm, different labs, years apart. This tests external validity (does loss aversion
replicate on an independent dataset?) and cross-dataset generalization (does a phenotype/choice model
trained on one dataset predict the other's choices out-of-dataset?), which is the strongest real
form of the pooling claim (C4) and Gate-7 external validity.

Events only, already downloaded. respcat/response coding: NARPS strongly/weakly accept|reject;
ds000005 respcat 1=accept, 0=reject, -1=no response.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402
import real_narps as RN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS5 = os.path.join(ROOT, "data", "ds000005", "events")


def _fit(g, l, y, l2=1e-2):
    X = np.column_stack([np.ones_like(g), g, l])
    def obj(b):
        z = np.clip(X @ b, -30, 30); logp = -np.logaddexp(0, -z); logq = -np.logaddexp(0, z)
        nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * l2 * np.sum(b[1:] ** 2)
        p = 1 / (1 + np.exp(-z)); grad = X.T @ (p - y); grad[1:] += l2 * b[1:]
        return nll, grad
    return optimize.minimize(obj, np.zeros(3), jac=True, method="L-BFGS-B").x


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))


def load_ds000005() -> Dict[str, Dict[str, np.ndarray]]:
    subs: Dict[str, Dict[str, list]] = {}
    for path in sorted(glob.glob(os.path.join(DS5, "*_events.tsv"))):
        sub = os.path.basename(path).split("_")[0]
        d = subs.setdefault(sub, {"gain": [], "loss": [], "accept": []})
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            gi, li, ci = hdr.index("gain"), hdr.index("loss"), hdr.index("respcat")
            for line in fh:
                p = line.rstrip("\n").split("\t")
                try:
                    rc = int(float(p[ci]))
                except (ValueError, IndexError):
                    continue
                if rc not in (0, 1):
                    continue
                d["gain"].append(float(p[gi])); d["loss"].append(float(p[li])); d["accept"].append(rc)
    return {s: {k: np.asarray(v, float) for k, v in d.items()} for s, d in subs.items()}


def load_narps() -> Dict[str, Dict[str, np.ndarray]]:
    raw = RN.load_subject_trials()
    return {s: {"gain": d["gain"], "loss": d["loss"], "accept": d["accept"]} for s, d in raw.items()}


def _per_subject_lambda(data, min_trials=30):
    lams, aucs = [], []
    for s, d in data.items():
        g, l, y = d["gain"], d["loss"], d["accept"]
        if len(y) < min_trials or len(np.unique(y)) < 2:
            continue
        b = _fit(g, l, y)
        if abs(b[1]) < 1e-6:
            continue
        lam = -b[2] / b[1]
        if np.isfinite(lam) and lam > 0:
            lams.append(lam)
        # within-subject 5-fold held-out AUC
        idx = np.arange(len(y)); rng = np.random.default_rng(0); rng.shuffle(idx)
        folds = np.array_split(idx, 5); proba = np.full(len(y), np.nan)
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[k] for k in range(5) if k != f])
            if len(np.unique(y[tr])) < 2:
                proba[te] = y[tr].mean(); continue
            bb = _fit(g[tr], l[tr], y[tr])
            proba[te] = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones_like(g[te]), g[te], l[te]]) @ bb, -30, 30)))
        aucs.append(_auc(y, proba))
    return np.array(lams), np.array([a for a in aucs if np.isfinite(a)])


def _pooled_population_model(data) -> np.ndarray:
    """Fit one logistic on all trials pooled (the population affective valuation)."""
    G, L, Y = [], [], []
    for d in data.values():
        if len(d["accept"]) >= 30 and len(np.unique(d["accept"])) > 1:
            G.append(d["gain"]); L.append(d["loss"]); Y.append(d["accept"])
    g, l, y = np.concatenate(G), np.concatenate(L), np.concatenate(Y)
    return _fit(g, l, y)


def _out_of_dataset_auc(model: np.ndarray, target) -> float:
    """Apply a model trained on one dataset to predict the other dataset's choices."""
    G, L, Y = [], [], []
    for d in target.values():
        if len(d["accept"]) >= 30 and len(np.unique(d["accept"])) > 1:
            G.append(d["gain"]); L.append(d["loss"]); Y.append(d["accept"])
    g, l, y = np.concatenate(G), np.concatenate(L), np.concatenate(Y)
    p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones_like(g), g, l]) @ model, -30, 30)))
    return _auc(y, p)


def run() -> Dict[str, object]:
    narps, ds5 = load_narps(), load_ds000005()
    lam_n, auc_n = _per_subject_lambda(narps)
    lam_d, auc_d = _per_subject_lambda(ds5)
    # distribution overlap (do the two datasets' loss-aversion distributions agree?)
    ks = stats.ks_2samp(lam_n, lam_d)
    # cross-dataset transfer: train on one, predict the other out-of-dataset
    m_narps = _pooled_population_model(narps)
    m_ds5 = _pooled_population_model(ds5)
    auc_narps_to_ds5 = _out_of_dataset_auc(m_narps, ds5)
    auc_ds5_to_narps = _out_of_dataset_auc(m_ds5, narps)
    pooled = np.concatenate([lam_n, lam_d])
    return {
        "experiment": "REAL_crossdataset_narps_plus_tom2007",
        "datasets": {"NARPS_ds001734": {"n_subjects": int(len(lam_n)), "median_lambda": float(np.median(lam_n)),
                                        "median_heldout_auc": float(np.median(auc_n))},
                     "Tom2007_ds000005": {"n_subjects": int(len(lam_d)), "median_lambda": float(np.median(lam_d)),
                                          "median_heldout_auc": float(np.median(auc_d))}},
        "loss_aversion_distributions_agree": {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                                              "agree": bool(ks.pvalue > 0.05)},
        "cross_dataset_transfer": {
            "narps_model_predicts_ds000005_auc": auc_narps_to_ds5,
            "ds000005_model_predicts_narps_auc": auc_ds5_to_narps,
            "note": "out-of-dataset choice prediction; >0.5 means the affective valuation transfers across labs",
        },
        "pooled": {"n_subjects": int(len(pooled)), "median_lambda": float(np.median(pooled)),
                   "iqr": [float(np.percentile(pooled, 25)), float(np.percentile(pooled, 75))]},
        "provenance": {"narps": "ds001734", "tom2007": "ds000005 (PDDL)"},
        "_lambda_narps": [float(x) for x in lam_n],
        "_lambda_ds5": [float(x) for x in lam_d],
    }


if __name__ == "__main__":
    res = run()
    with open(os.path.join(ROOT, "results", "real_crossdataset.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    d = res["datasets"]
    print("CROSS-DATASET real-data test (NARPS ds001734 + Tom 2007 ds000005)")
    for name, e in d.items():
        print(f"  {name:22s} n={e['n_subjects']:3d}  median lambda={e['median_lambda']:.2f}  "
              f"held-out AUC={e['median_heldout_auc']:.3f}")
    a = res["loss_aversion_distributions_agree"]
    print(f"  loss-aversion distributions agree: KS p={a['ks_p']:.3f} ({a['agree']})")
    t = res["cross_dataset_transfer"]
    print(f"  TRANSFER: NARPS-model -> ds000005 AUC={t['narps_model_predicts_ds000005_auc']:.3f} | "
          f"ds000005-model -> NARPS AUC={t['ds000005_model_predicts_narps_auc']:.3f}")
    print(f"  pooled n={res['pooled']['n_subjects']}  median lambda={res['pooled']['median_lambda']:.2f}")
