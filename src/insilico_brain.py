"""An in-silico belief model: a 'fake brain' that scores content for how likely an audience is to believe
it, generalizes across domains, and can be optimized by altering content factors.

Design, grounded in the encoding-model literature (Wang et al. 2025; Walters et al. 2022; Kriegeskorte
et al. 2018) and the approach-avoidance neuroforecasting result validated here on real fMRI:

  content factors  ->  brain-state layer  ->  belief score
  (6 alterable levers)   approach - avoidance    P(believe)

The brain-state layer is structured to match the validated neural composite: approach (evidence, framing,
affect, social proof) minus avoidance (identity threat), which mirrors the NAcc - anterior-insula signal
that generalized out-of-sample on the Knutson data. The belief score is a logistic read-out fit to real
belief outcomes (ChangeMyView deltas, Persuasion-for-Good donations).

We report, honestly and per Kriegeskorte's point about naming the generalization level:
  1. cross-domain generalization (train one corpus, score the other),
  2. multi-factor variance (bootstrap each factor's belief-effect, per corpus, cross-corpus agreement),
  3. optimization (counterfactually alter content factors to raise the belief score, measured lift).

This is not a scanned brain. It is a brain-STRUCTURED behavioral encoder: the intermediate representation
is constrained to the validated approach-avoidance structure, and the output is trained on real belief
outcomes. A learned fMRI encoder (TRIBE, or the author's Mary/Qualia) would replace the text->factor step;
that is the stated next step.

No em dashes.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .realdata import extract_features, load_labeled, FACTOR_NAMES

ROOT = os.path.dirname(os.path.dirname(__file__))
CMV = os.path.join(ROOT, "data", "raw", "winning-args-corpus")
P4G = os.path.join(ROOT, "data", "raw", "persuasionforgood-corpus")
RESULTS = os.path.join(ROOT, "results", "insilico_brain.json")

# brain-grounded sign prior: approach factors should raise belief (+), avoidance should lower it (-)
APPROACH = ["evidence", "framing", "affect", "social_proof", "concession"]
AVOIDANCE = ["identity_threat"]


def _factor_matrix(rows):
    return np.array([[r["factors"][f] for f in FACTOR_NAMES] for r in rows], float)


def _z(M, mu=None, sd=None):
    if mu is None:
        mu, sd = M.mean(0), M.std(0) + 1e-9
    return (M - mu) / sd, mu, sd


def load_cmv():
    rows = list(load_labeled(CMV))
    X = _factor_matrix(rows)
    y = np.array([r["success"] for r in rows], int)
    return X, y


def load_p4g():
    convos = json.load(open(os.path.join(P4G, "conversations.json")))
    text_by = {}
    with open(os.path.join(P4G, "utterances.jsonl")) as f:
        for line in f:
            u = json.loads(line)
            if u["meta"].get("role") != 0:
                continue
            cid = u.get("conversation_id") or u.get("root")
            text_by.setdefault(cid, []).append(u.get("text", "") or "")
    rows, ys = [], []
    for cid, meta in convos.items():
        txt = " ".join(text_by.get(cid, []))
        if len(txt.split()) < 5 or meta.get("donation_ee") is None:
            continue
        rows.append(extract_features(txt))
        ys.append(int(meta["donation_ee"] > 0))
    X = np.array([[r["factors"][f] for f in FACTOR_NAMES] for r in rows], float)
    return X, np.array(ys, int)


def brain_state(Xz):
    """approach - avoidance, from standardized factors (the validated neural structure)."""
    idx = {f: i for i, f in enumerate(FACTOR_NAMES)}
    approach = Xz[:, [idx[f] for f in APPROACH]].mean(1)
    avoid = Xz[:, [idx[f] for f in AVOIDANCE]].mean(1)
    return approach - avoid


def cross_domain():
    """Train the belief read-out on one corpus, score the other. Report AUC both directions."""
    Xc, yc = load_cmv()
    Xp, yp = load_p4g()
    out = {}
    for name, (Xtr, ytr, Xte, yte) in {
        "CMV->P4G": (Xc, yc, Xp, yp),
        "P4G->CMV": (Xp, yp, Xc, yc),
    }.items():
        Xtrz, mu, sd = _z(Xtr)
        Xtez, _, _ = _z(Xte, mu, sd)
        clf = LogisticRegression(max_iter=2000).fit(Xtrz, ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xtez)[:, 1])
        # brain-structured score AUC (approach-avoidance), fit only intercept+slope
        s_tr = brain_state(Xtrz).reshape(-1, 1)
        s_te = brain_state(Xtez).reshape(-1, 1)
        clf_b = LogisticRegression(max_iter=2000).fit(s_tr, ytr)
        auc_b = roc_auc_score(yte, clf_b.predict_proba(s_te)[:, 1])
        out[name] = {"auc_full_factors": float(auc), "auc_brain_state": float(auc_b)}
    return out


def factor_variance(n_boot=800, seed=0):
    """Bootstrap each factor's belief-effect (standardized logistic coef) per corpus; cross-corpus sign."""
    res = {}
    coefs_by_corpus = {}
    for cname, loader in [("CMV", load_cmv), ("P4G", load_p4g)]:
        X, y = loader()
        Xz, _, _ = _z(X)
        rng = np.random.default_rng(seed)
        boots = {f: [] for f in FACTOR_NAMES}
        for _ in range(n_boot):
            ix = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ix])) < 2:
                continue
            cf = LogisticRegression(max_iter=1000).fit(Xz[ix], y[ix]).coef_[0]
            for i, f in enumerate(FACTOR_NAMES):
                boots[f].append(cf[i])
        corp = {}
        for f in FACTOR_NAMES:
            arr = np.array(boots[f])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            corp[f] = {"coef": float(arr.mean()), "ci95": [float(lo), float(hi)],
                       "significant": bool(lo > 0 or hi < 0)}
        coefs_by_corpus[cname] = corp
    # cross-corpus robustness: significant and same sign in both
    robust = {}
    for f in FACTOR_NAMES:
        a, b = coefs_by_corpus["CMV"][f], coefs_by_corpus["P4G"][f]
        robust[f] = {"robust": bool(a["significant"] and b["significant"] and
                                    np.sign(a["coef"]) == np.sign(b["coef"])),
                     "cmv_coef": a["coef"], "p4g_coef": b["coef"],
                     "brain_prior": "+" if f in APPROACH else "-",
                     "sign_matches_brain": bool((np.sign(a["coef"]) > 0) == (f in APPROACH))}
    res["per_corpus"] = coefs_by_corpus
    res["cross_corpus_robust"] = robust
    return res


def optimize_counterfactual(corpus="CMV", quantile=0.3, step=1.0, seed=0):
    """Alter content of low-belief items along the belief gradient; measure predicted belief lift.

    We move each low-scoring item's factors by `step` SD in the direction of the fitted coefficient,
    ONLY for factors whose cross-corpus effect is robust and matches the brain prior. This is a
    counterfactual sensitivity analysis, not a deployed generator.
    """
    loader = load_cmv if corpus == "CMV" else load_p4g
    X, y = loader()
    Xz, mu, sd = _z(X)
    clf = LogisticRegression(max_iter=2000).fit(Xz, y)
    p0 = clf.predict_proba(Xz)[:, 1]
    # robust, brain-consistent factors to move
    var = factor_variance(n_boot=300, seed=seed)["cross_corpus_robust"]
    move = {i: np.sign(clf.coef_[0][i]) for i, f in enumerate(FACTOR_NAMES)
            if var[f]["robust"] and var[f]["sign_matches_brain"]}
    low = p0 < np.quantile(p0, quantile)
    Xz_opt = Xz.copy()
    for i, sgn in move.items():
        Xz_opt[low, i] += step * sgn
    p1 = clf.predict_proba(Xz_opt)[:, 1]
    return {
        "corpus": corpus,
        "n_low_belief_items": int(low.sum()),
        "factors_moved": [FACTOR_NAMES[i] for i in move],
        "mean_belief_before": float(p0[low].mean()),
        "mean_belief_after": float(p1[low].mean()),
        "mean_lift": float((p1[low] - p0[low]).mean()),
    }


def main():
    out = {
        "model": "in-silico brain-structured belief encoder (content factors -> approach-avoidance -> belief)",
        "cross_domain": cross_domain(),
        "factor_variance": factor_variance(),
        "optimization_CMV": optimize_counterfactual("CMV"),
        "optimization_P4G": optimize_counterfactual("P4G"),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(out, open(RESULTS, "w"), indent=2)

    print("== cross-domain generalization (AUC) ==")
    for k, v in out["cross_domain"].items():
        print(f"  {k}: full-factors {v['auc_full_factors']:.3f} | brain-state {v['auc_brain_state']:.3f}")
    print("== robust belief factors (significant + same sign in both corpora) ==")
    for f, v in out["factor_variance"]["cross_corpus_robust"].items():
        flag = "ROBUST" if v["robust"] else "-"
        print(f"  {f:16s} CMV {v['cmv_coef']:+.3f}  P4G {v['p4g_coef']:+.3f}  brain={v['brain_prior']}  {flag}")
    print("== optimization (counterfactual belief lift on low-belief items) ==")
    for key in ["optimization_CMV", "optimization_P4G"]:
        o = out[key]
        print(f"  {o['corpus']}: {o['mean_belief_before']:.3f} -> {o['mean_belief_after']:.3f} "
              f"(lift {o['mean_lift']:+.3f}) moving {o['factors_moved']}")
    print("wrote", RESULTS)
    return out


if __name__ == "__main__":
    main()
