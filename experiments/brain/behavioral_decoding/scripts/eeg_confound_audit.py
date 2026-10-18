"""EEG neuroprivacy confound audit (ASZED-153).

Reviewer asked: is the schizophrenia-vs-control decode a diagnosis signal, or a
demographic/acquisition confound? Patients are ~10 yr older, have longer
recordings, and the 4-phase cohort kept 52/76 patients but 24/77 controls.

Field-standard controls (Snoek, Miletić & Scholte 2019, NeuroImage 184:741-760,
"How to control for confounds in decoding analyses of neuroimaging data",
doi:10.1016/j.neuroimage.2018.09.074):
  - a CONFOUND-ONLY baseline (decode the label from nuisances alone), and
  - CONFOUND REGRESSION done INSIDE every CV fold (their key result: residualizing
    on the full data induces negative bias; per-fold removes it).
Caveat carried forward (NeurIPS 2026 residualization audits): residualization is
not a guarantee; we therefore report the confound-only baseline alongside it and
positive controls showing the features do carry the confound.

Runs subject-level (features averaged over each subject's epochs), StratifiedKFold,
class-balanced logistic regression, repeated over seeds. Pure numpy/sklearn; reads
the already-extracted features so no raw EEG recompute is needed.

    python scripts/eeg_confound_audit.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "results" / "iaseai27" / "aszed_features.npz"
SHEET = ROOT / "data" / "raw" / "aszed" / "ASZED_SpreadSheet.csv"
HEAD = ROOT / "results" / "iaseai27" / "aszed_headers.csv"
OUT = ROOT / "results" / "eeg_confound_audit"
SEEDS = list(range(20))
N_PERM = 1000


def load_subject_level():
    d = np.load(FEAT, allow_pickle=True)
    subj = d["subject"]; y = d["y"]
    X = np.hstack([d["X1"], d["X2"], d["X3"], d["X4"]])  # 4 phases concatenated
    subjects = sorted(set(subj.tolist()))
    Xs, ys = [], []
    for s in subjects:
        m = subj == s
        Xs.append(X[m].mean(axis=0))
        ys.append(int(y[m][0]))
    return subjects, np.array(Xs), np.array(ys)


def load_confounds(subjects):
    meta = {}
    with open(SHEET) as f:
        for r in csv.DictReader(f):
            meta[r["sn"]] = r
    dur = {}
    with open(HEAD) as f:
        for r in csv.DictReader(f):
            try:
                dur.setdefault(r["subject"], []).append(float(r["duration_s"]))
            except ValueError:
                pass
    rows = []
    for s in subjects:
        m = meta.get(s, {})
        age = float(m.get("age") or "nan")
        sex = 1.0 if (m.get("gender") or "").upper().startswith("M") else 0.0
        mindur = min(dur.get(s, [np.nan]))
        rows.append([age, sex, mindur])
    C = np.array(rows, dtype=float)
    # impute any nan with column mean
    for j in range(C.shape[1]):
        col = C[:, j]
        col[np.isnan(col)] = np.nanmean(col)
    return C


def clf():
    return LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)


def decode(X, y, seed, residualize_on=None):
    """Balanced accuracy over folds. If residualize_on=C, confound-regress X on C
    inside each training fold (Snoek per-fold) before scaling+decoding."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if residualize_on is not None:
            C = residualize_on
            cr = LinearRegression().fit(C[tr], Xtr)
            Xtr = Xtr - cr.predict(C[tr])
            Xte = Xte - cr.predict(C[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = clf().fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], model.predict(Xte)))
    return float(np.mean(accs))


def arm(X, y, residualize_on=None):
    vals = [decode(X, y, s, residualize_on) for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


def perm_null(X, y, n=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    obs = decode(X, y, seed)
    null = []
    for _ in range(n):
        yp = rng.permutation(y)
        null.append(decode(X, yp, seed))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (n + 1)
    return obs, float(null.mean()), float(np.percentile(null, 95)), float(p)


def main():
    subjects, X, y = load_subject_level()
    C = load_confounds(subjects)
    print(f"subjects={len(subjects)} patients={int(y.sum())} controls={int((y==0).sum())} "
          f"features={X.shape[1]}")

    res = {"n_subjects": len(subjects), "n_patient": int(y.sum()), "n_control": int((y == 0).sum()),
           "n_features": X.shape[1], "reference": "Snoek Miletić Scholte 2019 NeuroImage 184:741-760"}

    # --- positive controls: do the features carry the confounds? ---
    age = C[:, 0]
    age_r2 = float(np.mean(cross_val_score(Ridge(), StandardScaler().fit_transform(X), age,
                                           cv=StratifiedKFold(5, shuffle=True, random_state=0)
                                           if False else 5, scoring="r2")))
    sex_acc, _ = arm(X, C[:, 1].astype(int))
    res["positive_controls"] = {"age_from_EEG_r2": round(age_r2, 3),
                                "sex_from_EEG_balacc": round(sex_acc, 3)}

    # --- main arms ---
    raw_m, raw_s = arm(X, y)
    conf_m, conf_s = arm(C, y)                       # confound-only baseline
    comb_m, comb_s = arm(np.hstack([X, C]), y)       # EEG + confounds
    resid_m, resid_s = arm(X, y, residualize_on=C)   # per-fold confound regression
    res["arms"] = {
        "eeg_raw": [round(raw_m, 3), round(raw_s, 3)],
        "confound_only_age_sex_dur": [round(conf_m, 3), round(conf_s, 3)],
        "eeg_plus_confounds": [round(comb_m, 3), round(comb_s, 3)],
        "eeg_residualized_perfold": [round(resid_m, 3), round(resid_s, 3)],
    }

    # --- permutation null for the raw EEG decode ---
    obs, null_mean, null_p95, p = perm_null(X, y)
    res["permutation"] = {"observed": round(obs, 3), "null_mean": round(null_mean, 3),
                          "null_p95": round(null_p95, 3), "p_value": round(p, 4), "n_perm": N_PERM}

    # --- synthetic sanity: a pure-age feature must collapse under residualization ---
    rng = np.random.default_rng(0)
    Xsyn = (age[:, None] - age.mean()) + rng.normal(0, 0.5, size=(len(age), 5))
    syn_raw, _ = arm(Xsyn, y)
    syn_resid, _ = arm(Xsyn, y, residualize_on=C)
    res["synthetic_check_pure_age_feature"] = {"raw": round(syn_raw, 3),
                                               "residualized": round(syn_resid, 3),
                                               "expect": "raw high, residualized ~0.5"}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
