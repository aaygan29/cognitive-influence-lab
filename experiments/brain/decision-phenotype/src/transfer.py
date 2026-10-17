"""E3: affective-vs-behavioral aggregate forecasting, with positive and negative controls.

This is the within-estimator test of the neuroforecasting dissociation (Genevsky, Yoon & Knutson
2017; Tong et al. 2020): a small scanned sample stands in for a population, and we ask whether the
aggregate *affective* readout forecasts a per-stimulus market outcome out of sample better than the
aggregate *behavior* does. The forward generative model here has the dissociation built in on the
positive control (the market depends on the affective latent; behavior is a compressed, idiosyncratic
proxy of it), and absent on the negative control (the market is independent noise).

What this validates: the pipeline's OPERATING CHARACTERISTICS. It detects brain>behavior when the
generative structure contains it, and it does NOT manufacture the effect when the structure lacks it
(the negative control). It is a positive/negative control on synthetic data, in the same spirit as
the harness's run_demo. The empirical claim itself is tested on real fMRI/EEG (NARPS, DEAP), which is
the deployment step; this module is the honest instrument check that must pass first.

Leakage guard: folds hold out whole STIMULI (a forecast of an item the model has seen is not a
forecast). Scoring: out-of-sample R^2 and CRPS (strictly proper), with a stimulus-label permutation
null. Arms are equal-capacity where compared, and a content-only arm controls for the economic
baseline (Gate 4).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H          # noqa: E402
import phenotype as P        # noqa: E402


# --------------------------------------------------------------- generative model

def generate_transfer_data(n_subjects: int = 30, n_stim: int = 80, dissociation: bool = True,
                           seed: int = 0) -> Dict[str, np.ndarray]:
    """A shared set of stimuli viewed by many subjects.

    - affective latent a_j: a population-general valuation of stimulus j (same function for all
      subjects), the generalizable component.
    - neural readout: a_j plus subject-specific measurement noise; the aggregate (mean over subjects)
      is the "neural focus group" brain feature.
    - behavior: a choice whose probability is sigmoid of tau*(a_j + idiosyncratic integrative term),
      so the aggregate choice rate is a compressed, idiosyncratically-contaminated proxy of a_j.
    - market outcome m_j: driven by a_j (positive control) or independent noise (negative control).
    """
    rng = np.random.default_rng(seed)
    C = P.sample_contexts(rng, n_stim)                      # (n_stim, 5): Eg, El, sigma, amb, delay
    w_pop = np.array([1.0, 1.9, 0.6, 0.8, 0.5]) * P.FEATURE_SIGN   # population-mean valuation weights
    a = C @ w_pop                                          # affective latent per stimulus
    a = (a - a.mean()) / a.std()

    neural = np.empty((n_subjects, n_stim))
    choice = np.empty((n_subjects, n_stim))
    for i in range(n_subjects):
        neural[i] = a + rng.normal(0, 0.7, n_stim)         # noisy per-subject affective readout
        idiosyncratic = rng.normal(0, 0.9, n_stim)         # subject-specific integrative contribution
        tau = rng.uniform(0.8, 1.6)
        p = 1.0 / (1.0 + np.exp(-tau * (a + idiosyncratic)))
        choice[i] = (rng.uniform(size=n_stim) < p).astype(float)

    if dissociation:
        m = a + rng.normal(0, 0.4, n_stim)                 # market tracks the affective latent
    else:
        m = rng.normal(0, 1.0, n_stim)                     # market is independent noise

    return {"C": C, "a": a, "neural": neural, "choice": choice, "market": m,
            "brain_feat": neural.mean(axis=0)[:, None],    # aggregate readout, 1-dim
            "behavior_feat": choice.mean(axis=0)[:, None]} # aggregate choice rate, 1-dim


# --------------------------------------------------------------- ridge + stimulus-grouped CV

def _ridge_oos(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0,
               alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)) -> np.ndarray:
    """Out-of-fold predictions from ridge with a small internal alpha search. Rows are stimuli."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    pred = np.full(n, np.nan)
    for f in range(n_splits):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_splits) if g != f])
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        ytr_m = y[tr].mean()
        # pick alpha by a nested holdout inside the training fold
        best_a, best_err = alphas[0], np.inf
        inner = max(2, n_splits - 1)
        ii = np.array_split(tr, inner)
        for a_ in alphas:
            errs = []
            for j in range(inner):
                vte = ii[j]; vtr = np.concatenate([ii[q] for q in range(inner) if q != j])
                Xv = (X[vtr] - X[vtr].mean(0)) / (X[vtr].std(0) + 1e-9)
                b = np.linalg.solve(Xv.T @ Xv + a_ * np.eye(Xv.shape[1]), Xv.T @ (y[vtr] - y[vtr].mean()))
                Xvt = (X[vte] - X[vtr].mean(0)) / (X[vtr].std(0) + 1e-9)
                errs.append(np.mean((y[vte] - (Xvt @ b + y[vtr].mean())) ** 2))
            if np.mean(errs) < best_err:
                best_err, best_a = np.mean(errs), a_
        beta = np.linalg.solve(Xtr.T @ Xtr + best_a * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ytr_m))
        pred[te] = Xte @ beta + ytr_m
    return pred


def _oos_scores(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    resid = y - pred
    ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma = np.full_like(pred, np.std(resid) + 1e-9)
    return {"r2": r2, "pearson_r": float(stats.pearsonr(y, pred)[0]),
            "crps": H.crps_gaussian(y, pred, sigma)}


def _perm_p_r2(X: np.ndarray, y: np.ndarray, n_perm: int = 500, seed: int = 0) -> Tuple[float, float]:
    obs = _oos_scores(y, _ridge_oos(X, y, seed=seed))["r2"]
    rng = np.random.default_rng(seed + 1)
    null = np.array([_oos_scores(y, _ridge_oos(X, rng.permutation(y), seed=seed))["r2"] for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


# --------------------------------------------------------------- E3

def _arms(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {"content_only": data["C"], "behavior_only": data["behavior_feat"],
            "brain_only": data["brain_feat"],
            "brain_plus_content": np.hstack([data["brain_feat"], data["C"]]),
            "all": np.hstack([data["brain_feat"], data["behavior_feat"], data["C"]])}


def run_condition(dissociation: bool, n_subjects: int, n_stim: int, n_seeds: int) -> Dict[str, object]:
    arms = ["content_only", "behavior_only", "brain_only", "brain_plus_content", "all"]
    per_arm = {a: {"r2": [], "crps": []} for a in arms}
    brain_minus_behavior = []
    brain_minus_content = []
    perm_p_brain = []
    for s in range(n_seeds):
        data = generate_transfer_data(n_subjects, n_stim, dissociation=dissociation, seed=s)
        A = _arms(data); y = data["market"]
        sc = {}
        for a in arms:
            pred = _ridge_oos(A[a], y, seed=s)
            sco = _oos_scores(y, pred)
            per_arm[a]["r2"].append(sco["r2"]); per_arm[a]["crps"].append(sco["crps"])
            sc[a] = sco
        brain_minus_behavior.append(sc["brain_only"]["r2"] - sc["behavior_only"]["r2"])
        brain_minus_content.append(sc["brain_plus_content"]["r2"] - sc["content_only"]["r2"])
        _, p = _perm_p_r2(data["brain_feat"], y, n_perm=300, seed=s)
        perm_p_brain.append(p)

    def summ(v):
        v = np.array(v); ci = H.bootstrap_ci(v, statistic=np.mean, seed=0)
        return {"mean": float(v.mean()), "ci95": [ci["lo"], ci["hi"]]}

    out = {
        "dissociation": dissociation,
        "arms_r2": {a: summ(per_arm[a]["r2"]) for a in arms},
        "arms_crps": {a: float(np.mean(per_arm[a]["crps"])) for a in arms},
        "brain_minus_behavior_r2": summ(brain_minus_behavior),
        "brain_minus_content_r2": summ(brain_minus_content),
        "brain_permutation_p_median": float(np.median(perm_p_brain)),
    }
    bmb = out["brain_minus_behavior_r2"]
    out["brain_beats_behavior"] = bool(bmb["ci95"][0] > 0)
    out["brain_significant_vs_null"] = bool(np.median(perm_p_brain) < 0.05)
    return out


def run_e3(n_subjects: int = 30, n_stim: int = 80, n_seeds: int = 20) -> Dict[str, object]:
    pos = run_condition(True, n_subjects, n_stim, n_seeds)
    neg = run_condition(False, n_subjects, n_stim, n_seeds)
    # pipeline is valid iff it detects the effect on the positive control AND stays null on the negative
    valid = bool(pos["brain_beats_behavior"] and pos["brain_significant_vs_null"]
                 and (not neg["brain_beats_behavior"]) and (not neg["brain_significant_vs_null"]))
    return {
        "experiment": "E3_affective_vs_behavioral_transfer",
        "design": {"n_subjects": n_subjects, "n_stim": n_stim, "n_seeds": n_seeds,
                   "cv": "stimulus-grouped", "scoring": "oos R2 + CRPS + permutation null"},
        "framing": ("synthetic positive/negative control on the pipeline's operating characteristics; "
                    "real NARPS (fMRI) + DEAP (EEG) is the empirical deployment"),
        "positive_control": pos,
        "negative_control": neg,
        "pipeline_valid": valid,
    }


if __name__ == "__main__":
    import json

    res = run_e3()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    path = os.path.join(root, "results", "e3_transfer.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)

    for name, key in [("POSITIVE control (market tracks affect)", "positive_control"),
                      ("NEGATIVE control (market is noise)", "negative_control")]:
        c = res[key]
        print(f"\n{name}")
        for a, v in c["arms_r2"].items():
            print(f"  {a:<19s} oosR2={v['mean']:+.3f}  95%CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
        bmb = c["brain_minus_behavior_r2"]
        print(f"  brain-behavior ΔR2={bmb['mean']:+.3f} CI[{bmb['ci95'][0]:+.3f},{bmb['ci95'][1]:+.3f}]"
              f" | brain perm p(med)={c['brain_permutation_p_median']:.3f}"
              f" | brain>behavior={c['brain_beats_behavior']} sig={c['brain_significant_vs_null']}")
    print("\npipeline_valid (detects on positive, null on negative):", res["pipeline_valid"], "->", path)
