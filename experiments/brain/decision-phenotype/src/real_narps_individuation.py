"""Individuation: is the decision phenotype a stable individual trait? (NARPS ds001734)

The AIM account says the integrative component of choice INDIVIDUATES. If the recovered phenotype is
a real trait, a subject's phenotype estimated from one scanning run should identify that same subject
in held-out runs, better than chance, the way connectome fingerprinting identifies people across
sessions (Finn et al. 2015). NARPS gives four runs per subject, so this is a clean, leakage-safe test
using the behavioral events already on disk (no new downloads).

Per (subject, run) we build a phenotype vector: the affective valuation slopes [b_gain, b_loss],
the choice bias (intercept), the acceptance rate, mean reaction time, and the RT-vs-value slope (the
DDM evidence-accumulation signature). We fingerprint run-01 (database) against the mean of runs 02-04
(target): for each target subject the nearest database subject is the identification. Standardization
uses database statistics only, so no target information leaks. Chance is 1/N. A permutation null
(shuffling subject identity) calibrates the accuracy, and differential identifiability I_diff
(within- minus between-subject similarity) quantifies how separable the phenotypes are.

Honesty note: this is real cross-run identification of real people with a permutation null, not a
within-run circularity. If identification is only modest, that is reported as-is.
"""
from __future__ import annotations

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
FEATURES = ["b_gain", "b_loss", "intercept", "accept_rate", "mean_rt", "rt_value_slope"]


def _load_by_run() -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """subject -> run -> {gain, loss, accept, rt}."""
    import glob
    out: Dict[str, Dict[str, Dict[str, list]]] = {}
    for path in sorted(glob.glob(os.path.join(RN.EVENTS, "*_events.tsv"))):
        base = os.path.basename(path)
        sub = base.split("_")[0]
        run = base.split("run-")[1][:2]
        d = out.setdefault(sub, {}).setdefault(run, {"gain": [], "loss": [], "accept": [], "rt": []})
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            gi, li, ri, pi = (hdr.index(c) for c in ("gain", "loss", "RT", "participant_response"))
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) <= pi or p[pi] not in RN.ACCEPT:
                    continue
                try:
                    d["gain"].append(float(p[gi])); d["loss"].append(float(p[li]))
                    d["accept"].append(RN.ACCEPT[p[pi]])
                    d["rt"].append(float(p[ri]) if p[ri] not in ("", "n/a") else np.nan)
                except ValueError:
                    continue
    return {s: {r: {k: np.asarray(v, float) for k, v in rr.items()} for r, rr in runs.items()}
            for s, runs in out.items()}


def _phenotype_vector(d: Dict[str, np.ndarray]) -> np.ndarray:
    g, l, y, rt = d["gain"], d["loss"], d["accept"], d["rt"]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones_like(g), g, l])
    # simple penalized logistic
    def obj(b):
        z = np.clip(X @ b, -30, 30); logp = -np.logaddexp(0, -z); logq = -np.logaddexp(0, z)
        nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * 1e-2 * np.sum(b[1:] ** 2)
        p = 1 / (1 + np.exp(-z)); grad = X.T @ (p - y); grad[1:] += 1e-2 * b[1:]
        return nll, grad
    b = optimize.minimize(obj, np.zeros(3), jac=True, method="L-BFGS-B").x
    ok = np.isfinite(rt)
    net = np.abs(g - 1.8 * l)  # net value magnitude at a canonical lambda
    rt_slope = stats.pearsonr(net[ok], rt[ok])[0] if ok.sum() > 10 and np.std(net[ok]) > 0 else 0.0
    mean_rt = float(np.nanmean(rt)) if ok.any() else 0.0
    return np.array([b[1], b[2], b[0], float(np.mean(y)), mean_rt, rt_slope])


def _build_vectors(by_run):
    """Return db (run-01) and target (mean of runs 02-04) matrices over common subjects."""
    db, tg, subs = [], [], []
    for sub, runs in by_run.items():
        if "01" not in runs:
            continue
        v1 = _phenotype_vector(runs["01"])
        others = [_phenotype_vector(runs[r]) for r in ("02", "03", "04") if r in runs]
        others = [o for o in others if o is not None]
        if v1 is None or not others:
            continue
        db.append(v1); tg.append(np.mean(others, axis=0)); subs.append(sub)
    return np.array(db), np.array(tg), subs


def _identify(db: np.ndarray, tg: np.ndarray) -> Tuple[float, np.ndarray]:
    """Nearest-neighbour identification accuracy, standardizing by DB stats (leakage-safe)."""
    mu, sd = db.mean(0), db.std(0) + 1e-9
    D = (db - mu) / sd
    T = (tg - mu) / sd
    n = len(D)
    hits = 0
    matches = np.empty(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(D - T[i], axis=1)   # target i vs all db
        j = int(np.argmin(d)); matches[i] = j
        hits += (j == i)
    return hits / n, matches


def _idiff(db: np.ndarray, tg: np.ndarray) -> float:
    mu, sd = db.mean(0), db.std(0) + 1e-9
    D = (db - mu) / sd; T = (tg - mu) / sd
    # similarity = -distance; within = diagonal, between = off-diagonal
    dm = np.linalg.norm(D[:, None, :] - T[None, :, :], axis=2)
    sim = -dm
    within = np.mean(np.diag(sim))
    between = np.mean(sim[~np.eye(len(D), dtype=bool)])
    return float(within - between)


def structure_analysis(db: np.ndarray, tg: np.ndarray) -> Dict[str, object]:
    """Geometry of the phenotype manifold across the population (STRUCTURE lens).

    Uses the per-subject phenotype (mean of database and target vectors). Reports the coupling among
    affective coordinates, and the effective dimensionality via the participation ratio of the
    covariance eigenvalues -- low effective dimension supports the low-dimensional-manifold assumption.
    """
    P = 0.5 * (db + tg)
    Z = (P - P.mean(0)) / (P.std(0) + 1e-9)
    C = np.corrcoef(Z, rowvar=False)
    eig = np.linalg.eigvalsh(np.cov(Z, rowvar=False))
    eig = np.clip(eig, 0, None)
    part_ratio = float((eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12))  # effective dimensionality
    # variance explained by top-2 PCs
    ev = np.sort(eig)[::-1]; top2 = float(ev[:2].sum() / (ev.sum() + 1e-12))
    gl = float(C[FEATURES.index("b_gain"), FEATURES.index("b_loss")])
    return {
        "n_features": len(FEATURES),
        "gain_loss_coordinate_corr": gl,
        "participation_ratio_effective_dim": part_ratio,
        "top2_pc_variance_explained": top2,
        "interpretation": ("effective dimensionality below the feature count indicates the population "
                           "phenotype occupies a low-dimensional structure, as the manifold model assumes"),
    }


def run() -> Dict[str, object]:
    by_run = _load_by_run()
    db, tg, subs = _build_vectors(by_run)
    n = len(subs)
    structure = structure_analysis(db, tg)
    acc, _ = _identify(db, tg)
    idiff = _idiff(db, tg)
    # permutation null: shuffle target identity
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        perm = rng.permutation(n)
        a, _ = _identify(db, tg[perm])
        null.append(a)
    null = np.array(null)
    p = float((np.sum(null >= acc) + 1) / (len(null) + 1))
    chance = 1.0 / n
    gated = H.gate_effect("individuation_accuracy_above_chance", effect=acc - chance,
                          n=n, kind="d")  # descriptive gate on the accuracy lift
    return {
        "experiment": "REAL_narps_individuation",
        "n_subjects": n,
        "features": FEATURES,
        "identification_accuracy": float(acc),
        "chance": float(chance),
        "fold_over_chance": float(acc / chance),
        "permutation_p": p,
        "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
        "I_diff": idiff,
        "significant": bool(p < 0.05),
        "structure": structure,
        "_null_distribution": [float(x) for x in null[:1000]],
        "_coords_gain_loss": (0.5 * (db + tg))[:, :2].tolist(),
        "design": "database=run-01, target=mean(runs 02-04), NN identification, DB-standardized (leakage-safe)",
        "provenance": {"dataset": "ds001734", "source": "events.tsv, 4 runs"},
    }


if __name__ == "__main__":
    res = run()
    with open(os.path.join(ROOT, "results", "real_narps_individuation.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("REAL individuation (NARPS, cross-run fingerprinting of the decision phenotype)")
    print(f"  n={res['n_subjects']}  identification accuracy={res['identification_accuracy']:.3f} "
          f"(chance {res['chance']:.3f}, {res['fold_over_chance']:.1f}x)")
    print(f"  permutation p={res['permutation_p']:.4f}  null mean={res['null_mean']:.3f} "
          f"(95th {res['null_p95']:.3f})  I_diff={res['I_diff']:.3f}  significant={res['significant']}")
    s = res["structure"]
    print(f"  STRUCTURE: gain-loss coord corr={s['gain_loss_coordinate_corr']:+.3f}  "
          f"effective dim={s['participation_ratio_effective_dim']:.2f}/{s['n_features']}  "
          f"top-2 PC var={s['top2_pc_variance_explained']:.2f}")
