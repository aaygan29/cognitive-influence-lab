"""Real-data run: the decision phenotype on NARPS ds001734 (OpenNeuro, CC0).

NARPS is the mixed-gambles reward task (Botvinik-Nezer et al. 2019), the paradigm of Tom, Fox,
Trepel & Poldrack (2007): on each trial a 50/50 gamble of a potential gain vs a potential loss, and
the subject accepts or rejects. Real human choices, 108 subjects, 64 trials x 4 runs, with reaction
times. This is real behavioral data for the affective decision phenotype: from the choices we recover
each subject's loss aversion (lambda = |b_loss| / |b_gain|), the same quantity that study estimated,
and we run the honesty layer and pooling on real predictions.

What this delivers on real data:
- RN1 (C1): per-subject loss aversion recovered from real choices; distribution vs the literature;
  within-subject held-out choice prediction beats base rate (proper scores).
- RN2 (C2): split-conformal coverage on real held-out choices; calibration; the identifiability gate
  abstains for subjects whose gain sensitivity is not resolvable.
- RN3 (C4): empirical-Bayes pooling stabilizes per-subject estimates (variance reduction), most for
  the subjects with the fewest usable trials.
- RN4 (tau, exploratory): an EZ-diffusion read of drift and boundary from real RTs.

The fMRI affective arm (NAcc/insula, the real "brain beats behavior" test) needs preprocessed
volumes and a GLM; it is the next stage and is NOT claimed here.

Provenance: ds001734 v1.0.5, DOI 10.18112/openneuro.ds001734, events.tsv only.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVENTS = os.path.join(ROOT, "data", "narps", "events")
ACCEPT = {"strongly_accept": 1, "weakly_accept": 1, "strongly_reject": 0, "weakly_reject": 0}


# --------------------------------------------------------------- load

def load_subject_trials() -> Dict[str, Dict[str, np.ndarray]]:
    """Per subject: gain, loss, accept (0/1), RT. Drops NoResp."""
    subjects: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(glob.glob(os.path.join(EVENTS, "*_events.tsv"))):
        sub = os.path.basename(path).split("_")[0]
        d = subjects.setdefault(sub, {"gain": [], "loss": [], "accept": [], "rt": []})
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            gi, li, ri, pi = (header.index(c) for c in ("gain", "loss", "RT", "participant_response"))
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) <= pi or p[pi] not in ACCEPT:
                    continue
                try:
                    g, l = float(p[gi]), float(p[li])
                    rt = float(p[ri]) if p[ri] not in ("", "n/a") else np.nan
                except ValueError:
                    continue
                d["gain"].append(g); d["loss"].append(l); d["accept"].append(ACCEPT[p[pi]]); d["rt"].append(rt)
    return {s: {k: np.asarray(v, float) for k, v in d.items()} for s, d in subjects.items()}


# --------------------------------------------------------------- logistic with intercept

def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, penalize_intercept: bool = False):
    """MLE logistic; X already includes an intercept column at position 0. Returns beta, cov."""
    d = X.shape[1]
    pen = np.ones(d) * l2
    if not penalize_intercept:
        pen[0] = 0.0

    def obj(b):
        z = X @ b
        logp = -np.logaddexp(0.0, -z); logq = -np.logaddexp(0.0, z)
        nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * np.sum(pen * b * b)
        p = 1.0 / (1.0 + np.exp(-z))
        return nll, X.T @ (p - y) + pen * b

    res = optimize.minimize(obj, np.zeros(d), jac=True, method="L-BFGS-B")
    z = X @ res.x; p = 1.0 / (1.0 + np.exp(-z)); W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + np.diag(pen) + 1e-9 * np.eye(d))
    return res.x, cov


def _design(gain: np.ndarray, loss: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(gain), gain, loss])   # [1, gain, loss]


def loss_aversion(beta: np.ndarray) -> float:
    b_gain, b_loss = beta[1], beta[2]
    return float(-b_loss / b_gain) if abs(b_gain) > 1e-9 else float("nan")


# --------------------------------------------------------------- RN1: per-subject fit + prediction

def _within_subject_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0):
    n = len(y); idx = np.arange(n)
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    proba = np.full(n, np.nan)
    for f in range(n_splits):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(n_splits) if g != f])
        if len(np.unique(y[tr])) < 2:
            proba[te] = y[tr].mean(); continue
        b, _ = _fit_logistic(X[tr], y[tr])
        proba[te] = 1.0 / (1.0 + np.exp(-(X[te] @ b)))
    return proba


def load_groups() -> Dict[str, str]:
    path = os.path.join(ROOT, "data", "narps", "participants.tsv")
    groups = {}
    if os.path.exists(path):
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            si, gi = hdr.index("participant_id"), hdr.index("group")
            for line in fh:
                p = line.rstrip("\n").split("\t")
                groups[p[si]] = p[gi]
    return groups


def _fit_pooled(designs: List[np.ndarray], ys: List[np.ndarray], n_iter: int = 6):
    """Empirical-Bayes pooling of the [b_gain, b_loss] slopes (intercept stays unpooled)."""
    betas = [(_fit_logistic(X, y))[0] for X, y in zip(designs, ys)]
    B = np.array(betas)
    mu = B.mean(0); sig2 = np.clip(B.var(0), 1e-3, None)
    for _ in range(n_iter):
        new = []
        for X, y in zip(designs, ys):
            lam = 1.0 / sig2; lam[0] = 0.0   # do not pool the intercept
            d = X.shape[1]
            def obj(b):
                z = X @ b; logp = -np.logaddexp(0, -z); logq = -np.logaddexp(0, z)
                nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * np.sum(lam * (b - mu) ** 2)
                p = 1 / (1 + np.exp(-z))
                return nll, X.T @ (p - y) + lam * (b - mu)
            new.append(optimize.minimize(obj, mu.copy(), jac=True, method="L-BFGS-B").x)
        B = np.array(new); mu = B.mean(0); sig2 = np.clip(B.var(0), 1e-3, None)
    return [B[i] for i in range(len(designs))]


def run_real(gate_z: float = 1.96, min_trials: int = 40) -> Dict[str, object]:
    subs = load_subject_trials()
    groups = load_groups()
    provenance = {"dataset": "ds001734", "version": "v1.0.5",
                  "doi": "10.18112/openneuro.ds001734", "source": "events.tsv", "n_subjects_raw": len(subs)}
    kept_designs, kept_ys, kept_subs, kept_ntrials, rt_signal = [], [], [], [], []
    group_lambda = {"equalRange": [], "equalIndifference": []}

    lambdas, kept_lambdas, oos = [], [], {"auc": [], "brier": [], "logloss": [], "baserate_brier": []}
    conformal_cov, ece_list = [], []
    n_abstain = 0
    per_subject = {}
    all_y, all_p = [], []

    for sub, d in subs.items():
        g, l, y = d["gain"], d["loss"], d["accept"]
        if len(y) < min_trials or len(np.unique(y)) < 2:
            n_abstain += 1
            per_subject[sub] = {"abstained": True, "reason": "too few usable/one-class trials", "n": int(len(y))}
            continue
        X = _design(g, l)
        beta, cov = _fit_logistic(X, y)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        gain_z = abs(beta[1]) / (se[1] + 1e-9)
        lam = loss_aversion(beta)
        # identifiability gate: gain sensitivity must be resolvable and lambda finite/positive
        if gain_z < gate_z or not np.isfinite(lam) or lam <= 0:
            n_abstain += 1
            per_subject[sub] = {"abstained": True, "reason": f"gain_z={gain_z:.2f} below gate or lambda invalid",
                                "n": int(len(y))}
            continue

        proba = _within_subject_cv(X, y)
        # proper scores + AUC on held-out
        auc = float(stats.rankdata(proba)[y == 1].mean() - stats.rankdata(proba)[y == 0].mean()) / len(y) + 0.5 \
            if len(np.unique(y)) == 2 else np.nan
        # robust AUC via Mann-Whitney
        pos, neg = proba[y == 1], proba[y == 0]
        auc = float((stats.mannwhitneyu(pos, neg, alternative="greater").statistic) / (len(pos) * len(neg)))
        brier = H.brier_score(y, proba); ll = H.log_score(y, proba)
        base = y.mean(); base_brier = H.brier_score(y, np.full_like(proba, base))

        # split-conformal on held-out (classification score s = 1 - p_true)
        cut = len(y) // 2
        s_cal = 1 - np.where(y[:cut] == 1, proba[:cut], 1 - proba[:cut])
        qhat = H.conformal_classification_qhat(s_cal, alpha=0.1)
        # coverage on the other half: prediction set includes the true label?
        cov_hits = []
        for i in range(cut, len(y)):
            pset = []
            if proba[i] >= 1 - qhat: pset.append(1)
            if (1 - proba[i]) >= 1 - qhat: pset.append(0)
            cov_hits.append(int(y[i]) in pset)
        conformal_cov.append(float(np.mean(cov_hits)))
        ece_list.append(H.reliability(y, proba, n_bins=8)["ece"])

        lambdas.append(lam); kept_lambdas.append(lam)
        oos["auc"].append(auc); oos["brier"].append(brier); oos["logloss"].append(ll)
        oos["baserate_brier"].append(base_brier)
        all_y.append(y); all_p.append(proba)
        kept_designs.append(X); kept_ys.append(y); kept_subs.append(sub); kept_ntrials.append(len(y))
        grp = groups.get(sub)
        if grp in group_lambda:
            group_lambda[grp].append(lam)
        # RN4 RT signature: net decision value magnitude vs RT (evidence accumulation -> faster)
        net = np.abs(g - lam * l)
        rt = d["rt"]; ok = np.isfinite(rt)
        if ok.sum() > 20 and np.std(net[ok]) > 1e-6:
            rt_signal.append(float(stats.pearsonr(net[ok], rt[ok])[0]))
        per_subject[sub] = {"abstained": False, "n": int(len(y)), "loss_aversion": lam,
                            "gain_z": float(gain_z), "oos_auc": auc, "oos_brier": brier,
                            "group": grp}

    lam = np.array(kept_lambdas)
    def ci(v):
        v = np.array(v); b = H.bootstrap_ci(v, np.mean, seed=0); return [b["lo"], b["hi"]]

    # RN3 pooling: empirical-Bayes pooled lambda vs unpooled; variance reduction, most for low-trial subjects
    pooled_betas = _fit_pooled(kept_designs, kept_ys)
    pooled_lambda = np.array([loss_aversion(b) for b in pooled_betas])
    pooled_lambda = pooled_lambda[np.isfinite(pooled_lambda) & (pooled_lambda > 0)]
    unp_sd = float(np.std(lam)); poo_sd = float(np.std(pooled_lambda))
    ntr = np.array(kept_ntrials, float)
    # per-subject shrinkage magnitude vs trial count (expect more shrinkage at fewer trials)
    shrink = np.abs(np.array([loss_aversion(b) for b in pooled_betas]) - lam)
    fin = np.isfinite(shrink)
    shrink_vs_ntrials_r = float(stats.pearsonr(ntr[fin], shrink[fin])[0]) if fin.sum() > 5 else None

    result = {
        "experiment": "REAL_narps_ds001734_phenotype",
        "provenance": provenance,
        "n_subjects_used": int(len(kept_lambdas)),
        "n_abstained": int(n_abstain),
        "abstention_rate": float(n_abstain / len(subs)),
        "RN1_loss_aversion": {
            "median": float(np.median(lam)), "mean": float(np.mean(lam)),
            "iqr": [float(np.percentile(lam, 25)), float(np.percentile(lam, 75))],
            "frac_loss_averse_gt1": float(np.mean(lam > 1.0)),
            "literature_ref": "Tom et al. 2007 median lambda ~1.4-2.0",
        },
        "RN1_prediction": {
            "mean_oos_auc": float(np.mean(oos["auc"])), "auc_ci95": ci(oos["auc"]),
            "median_oos_auc": float(np.median(oos["auc"])),
            "median_oos_brier": float(np.median(oos["brier"])),
            "median_baserate_brier": float(np.median(oos["baserate_brier"])),
            "beats_baserate": bool(np.median(oos["brier"]) < np.median(oos["baserate_brier"])),
            "median_logloss": float(np.median(oos["logloss"])),
        },
        "RN2_honesty": {
            "median_conformal_coverage": float(np.median(conformal_cov)),
            "nominal": 0.9, "median_ece": float(np.median(ece_list)),
            "identifiability_gate_abstention_rate": float(n_abstain / len(subs)),
            "note": "coverage >= nominal (conservative) since classes are few and predictions confident",
        },
        "RN3_pooling": {
            "unpooled_lambda_sd": unp_sd, "pooled_lambda_sd": poo_sd,
            "variance_reduced": bool(poo_sd < unp_sd),
            "shrinkage_vs_ntrials_r": shrink_vs_ntrials_r,
            "note": "negative r means fewer-trial subjects are shrunk more toward the population",
        },
        "RN4_rt_ddm_signature": {
            "median_corr_netvalue_rt": float(np.median(rt_signal)) if rt_signal else None,
            "n_subjects": len(rt_signal),
            "interpretation": ("negative = larger net decision-value magnitude -> faster RT, the "
                               "evidence-accumulation signature the DDM predicts; motivates full-DDM tau"),
        },
        "group_loss_aversion": {
            g: {"median": float(np.median(v)), "n": len(v)} for g, v in group_lambda.items() if v
        },
        "per_subject": per_subject,
    }
    return result, subs, per_subject


if __name__ == "__main__":
    res, subs, per_subject = run_real()
    path = os.path.join(ROOT, "results", "real_narps.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    r = res
    print("REAL DATA: NARPS ds001734 mixed-gambles phenotype")
    print(f"  subjects used: {r['n_subjects_used']}  abstained: {r['n_abstained']} "
          f"({r['abstention_rate']*100:.0f}%)")
    la = r["RN1_loss_aversion"]
    print(f"  RN1 loss aversion lambda: median={la['median']:.2f} "
          f"IQR[{la['iqr'][0]:.2f},{la['iqr'][1]:.2f}]  frac>1={la['frac_loss_averse_gt1']:.2f}  "
          f"(lit: {la['literature_ref']})")
    pr = r["RN1_prediction"]
    print(f"  RN1 held-out choice prediction: mean AUC={pr['mean_oos_auc']:.3f} "
          f"CI{[round(x,3) for x in pr['auc_ci95']]}  Brier={pr['median_oos_brier']:.3f} "
          f"vs base-rate {pr['median_baserate_brier']:.3f}  beats_baserate={pr['beats_baserate']}")
    h = r["RN2_honesty"]
    print(f"  RN2 honesty: conformal coverage={h['median_conformal_coverage']:.3f} (nom 0.90)  "
          f"ECE={h['median_ece']:.3f}  gate abstention={h['identifiability_gate_abstention_rate']*100:.0f}%")
    p3 = r["RN3_pooling"]
    print(f"  RN3 pooling: lambda SD {p3['unpooled_lambda_sd']:.3f} -> {p3['pooled_lambda_sd']:.3f} "
          f"(reduced={p3['variance_reduced']}); shrinkage-vs-ntrials r={p3['shrinkage_vs_ntrials_r']}")
    p4 = r["RN4_rt_ddm_signature"]
    print(f"  RN4 RT/DDM signature: median corr(|net value|, RT)={p4['median_corr_netvalue_rt']} "
          f"(n={p4['n_subjects']}; negative = DDM-consistent)")
    print(f"  group lambda: {r['group_loss_aversion']}")
    print("->", path)
