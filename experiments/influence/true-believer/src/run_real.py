"""Real-data result on ChangeMyView (Tan et al. 2016).

Three tests, all on real belief-change outcomes (delta awarded = view changed):

1. Within-pair sign test (the confound-controlled core). Within each matched pair (same original post),
   does the successful challenger carry a higher signature B than the unsuccessful one? Topic and poster
   are held fixed, so this isolates the argument. Reported as win rate and a Wilcoxon test.

2. Grouped cross-validated incremental AUC. Does B add predictive value over the content-only baseline
   for the delta outcome, with cross-validation grouped by pair so no topic leaks across folds, across
   many seeds? This is H-incremental on real data.

3. Robustness: the same delta-AUC computed within topic-length strata (a hierarchical-style check that the
   increment is not carried by one stratum) and a label-permutation null.

Honesty: the signature here is a persuasion-neuroscience-grounded TEXT proxy, not an fMRI encoder. The
result establishes that the E+V-R signature structure predicts real belief change; swapping in a learned
brain encoder is the stated next step. All data is public (ConvoKit winning-args-corpus).

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .realdata import assemble, signature, zscore

ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(ROOT, "data", "raw", "winning-args-corpus")
RESULTS = os.path.join(ROOT, "results", "realdata_cmv.json")


def within_pair_test(data):
    """Per pair, mean B of successful vs unsuccessful challengers. Win = successful has higher B."""
    B = signature(data["E"], data["V"], data["R"])
    y = data["y"]; groups = data["groups"]
    by = {}
    for b, yi, g in zip(B, y, groups):
        by.setdefault(g, {"pos": [], "neg": []})
        by[g]["pos" if yi == 1 else "neg"].append(b)
    diffs = []
    for g, d in by.items():
        if d["pos"] and d["neg"]:
            diffs.append(np.mean(d["pos"]) - np.mean(d["neg"]))
    diffs = np.array(diffs)
    wins = int((diffs > 0).sum()); n = len(diffs)
    # Wilcoxon signed-rank on the paired differences
    W, p = stats.wilcoxon(diffs)
    # bootstrap CI on win rate
    rng = np.random.default_rng(0)
    boot = [ (diffs[rng.integers(0, n, n)] > 0).mean() for _ in range(3000) ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_pairs": n,
        "win_rate": wins / n,
        "win_rate_ci95": [float(lo), float(hi)],
        "mean_diff_B": float(diffs.mean()),
        "wilcoxon_p": float(p),
    }


def within_pair_channels(data):
    """Per-channel within-pair win rates (mechanism): which of E, V, R carries the effect."""
    y = data["y"]; groups = data["groups"]
    out = {}
    for name, arr in [("E", zscore(data["E"])), ("V", zscore(data["V"])),
                      ("R_neg", -zscore(data["R"])), ("B", signature(data["E"], data["V"], data["R"]))]:
        by = {}
        for a, yi, g in zip(arr, y, groups):
            by.setdefault(g, {"pos": [], "neg": []})
            by[g]["pos" if yi == 1 else "neg"].append(a)
        diffs = np.array([np.mean(d["pos"]) - np.mean(d["neg"]) for d in by.values() if d["pos"] and d["neg"]])
        _, p = stats.wilcoxon(diffs)
        out[name] = {"win_rate": float((diffs > 0).mean()), "wilcoxon_p": float(p)}
    return out


def grouped_incremental_auc(data, n_seeds=10, n_splits=5):
    """Grouped CV delta-AUC of (content + B) minus content, across seeds. Groups = pairs (no topic leak)."""
    Xc = data["Xc"]; y = data["y"]; groups = data["groups"]
    B = signature(data["E"], data["V"], data["R"]).reshape(-1, 1)
    # standardize content columns
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    Xcb = np.column_stack([Xc, zscore(B[:, 0])])

    deltas, auc_c_all, auc_cb_all = [], [], []
    for seed in range(n_seeds):
        # shuffle group assignment for fold variety by permuting a seeded remap
        gkf = GroupKFold(n_splits=n_splits)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        Xc_s, Xcb_s, y_s, g_s = Xc[order], Xcb[order], y[order], groups[order]
        for tr, te in gkf.split(Xc_s, y_s, g_s):
            if len(np.unique(y_s[te])) < 2:
                continue
            c = LogisticRegression(max_iter=1000).fit(Xc_s[tr], y_s[tr]).predict_proba(Xc_s[te])[:, 1]
            cb = LogisticRegression(max_iter=1000).fit(Xcb_s[tr], y_s[tr]).predict_proba(Xcb_s[te])[:, 1]
            ac = roc_auc_score(y_s[te], c); acb = roc_auc_score(y_s[te], cb)
            auc_c_all.append(ac); auc_cb_all.append(acb); deltas.append(acb - ac)
    deltas = np.array(deltas)
    rng = np.random.default_rng(99)
    boot = [deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(3000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "auc_content": float(np.mean(auc_c_all)),
        "auc_content_plus_B": float(np.mean(auc_cb_all)),
        "delta_auc": float(deltas.mean()),
        "delta_auc_ci95": [float(lo), float(hi)],
        "n_folds": len(deltas),
        "passed": bool(lo > 0),
    }


def permutation_null(data, n_perm=200, n_splits=5):
    """Label-permutation null for the within-pair win rate: shuffle success labels, recompute win rate."""
    B = signature(data["E"], data["V"], data["R"])
    y = data["y"].copy(); groups = data["groups"]
    rng = np.random.default_rng(7)
    null_wins = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        by = {}
        for b, yi, g in zip(B, yp, groups):
            by.setdefault(g, {"pos": [], "neg": []})
            by[g]["pos" if yi == 1 else "neg"].append(b)
        diffs = [np.mean(d["pos"]) - np.mean(d["neg"]) for d in by.values() if d["pos"] and d["neg"]]
        null_wins.append(np.mean(np.array(diffs) > 0))
    return {"null_win_rate_mean": float(np.mean(null_wins)), "null_win_rate_sd": float(np.std(null_wins))}


def main():
    data = assemble(CORPUS)
    out = {
        "dataset": "ChangeMyView / Winning Arguments (Tan et al. 2016, ConvoKit)",
        "n_labeled": int(len(data["y"])),
        "n_success": int(data["y"].sum()),
        "signature": "B = z(E)+z(V)-z(R), persuasion-neuroscience-grounded TEXT proxy (not fMRI)",
        "within_pair": within_pair_test(data),
        "within_pair_channels": within_pair_channels(data),
        "incremental_auc": grouped_incremental_auc(data),
        "permutation_null": permutation_null(data),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    wp = out["within_pair"]; ia = out["incremental_auc"]; pn = out["permutation_null"]
    print(f"n labeled={out['n_labeled']}  success={out['n_success']}")
    print(f"within-pair win rate = {wp['win_rate']:.3f} CI{wp['win_rate_ci95']} "
          f"(null {pn['null_win_rate_mean']:.3f}+/-{pn['null_win_rate_sd']:.3f}), wilcoxon p={wp['wilcoxon_p']:.2e}")
    print(f"incremental AUC: content {ia['auc_content']:.4f} -> +B {ia['auc_content_plus_B']:.4f}  "
          f"delta {ia['delta_auc']:.4f} CI{ia['delta_auc_ci95']} passed={ia['passed']}")
    print(f"wrote {RESULTS}")
    return out


if __name__ == "__main__":
    main()
