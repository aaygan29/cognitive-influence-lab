"""E5/E6: inverse inference and the attribution-scale ladder (C5, C6).

Forward gives a decision from a posture; inverse recovers the latent affective posture that best
explains observed decisions. The scientific question (C6) is where along the aggregate-to-individual
axis this inference survives. We evaluate nested attribution scopes S1..S6: an aggregate scope pools
many decision-makers (the population-general affective component dominates and is well determined),
an individual scope is one person with few cleanly-attributable decisions (idiosyncratic integrative
variance and sparsity take over). The prediction from AIM and neuroforecasting is a dose-response:
identifiability falls and abstention rises as scope narrows.

The honesty gate (C5) is the safety property: at scopes where the posture is not identifiable, the
gate ABSTAINS rather than emitting a confident latent. We report identifiability across all entities
(the raw dose-response) and, separately, recovery on the NON-abstained subset, to show the gate keeps
what it reports calibrated.

Ethics: the latent is a computational affective posture with a neural interpretation from the forward
map, never a claim of measured neural firing in a named individual. This is a simulation of the
mechanism; the real US-president decision corpus (two blind coders + Cohen's kappa) is the deployment.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phenotype as P  # noqa: E402

# nested attribution scopes: (label, K contributors, D cleanly-attributable decisions)
SCOPES: List[Tuple[str, int, int]] = [
    ("S1_exec+agencies", 150, 400),
    ("S2_WH+Pentagon", 60, 300),
    ("S3_WhiteHouse", 25, 200),
    ("S4_Pres+cabinet", 10, 140),
    ("S5_Pres+advisors", 4, 90),
    ("S6_President", 1, 50),
]


def _entity_postures(rng: np.random.Generator, n_entities: int) -> Dict[str, np.ndarray]:
    """True affective posture per entity: loss-aversion and threat ratios, drawn with spread."""
    return {"loss_aversion": rng.uniform(1.2, 2.8, n_entities),
            "threat": rng.uniform(0.4, 1.4, n_entities)}


def _member_thetas(rng, la: float, th: float, K: int) -> List[Dict[str, float]]:
    """K decision-makers clustered around the entity's posture (tight within-entity affective)."""
    out = []
    for _ in range(K):
        rho = np.clip(rng.normal(1.0, 0.10), 0.1, None)
        out.append({
            "rho": rho,
            "lambda": np.clip(rng.normal(la, 0.15), 0.05, None) * rho,
            "omega": np.clip(rng.normal(th, 0.12), 0.05, None) * rho,
            "kappa": np.clip(rng.normal(0.6, 0.3), 0.05, None) * rho,   # idiosyncratic integrative
            "delta": np.clip(rng.normal(0.5, 0.3), 0.05, None) * rho,
            "tau": np.clip(rng.normal(1.3, 0.3), 0.1, None),
        })
    return out


def _aggregate_decisions(rng, members: List[Dict[str, float]], D: int) -> Tuple[np.ndarray, np.ndarray]:
    """D events; each decision is the aggregate (mean-drift sign) over the entity's members.

    With many members the mean drift concentrates on the entity's shared affective drift (clean
    signal); with one member it is a single noisy agent (the individual scope)."""
    X = P.sample_contexts(rng, D)
    drifts = np.zeros(D)
    for m in members:
        drifts += P.drift(X, m)
    drifts /= len(members)
    p = 1.0 / (1.0 + np.exp(-drifts))
    y = (rng.uniform(size=D) < p).astype(float)
    return X, y


def infer_posture(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Logistic inverse fit; returns recovered ratios and the analytic SE of gain sensitivity.

    Identifiability hinges on gain sensitivity b_Eg being resolvable: if it is not significantly
    nonzero the ratios are undetermined, and the gate should abstain.
    """
    beta = P.fit_agent(X, y, l2=1e-3)
    # analytic covariance at the fit: inv(X^T W X), W = p(1-p)
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    W = p * (1 - p)
    XtWX = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
    cov = np.linalg.inv(XtWX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ratios = P.recover_ratios(beta)
    gain_z = float(abs(beta[0]) / (se[0] + 1e-9))   # |b_Eg| / SE(b_Eg)
    return {"ratios": ratios, "gain_z": gain_z}


def run_scale_ladder(n_entities: int = 50, n_seeds: int = 8, gate_z: float = 1.96) -> Dict[str, object]:
    params = ["loss_aversion", "threat"]
    per_scope = {}
    for label, K, D in SCOPES:
        r_all = {k: [] for k in params}          # recovery across all entities, per seed
        r_kept = {k: [] for k in params}         # recovery on non-abstained subset
        abst = []
        for s in range(n_seeds):
            rng = np.random.default_rng(7000 + s)
            postures = _entity_postures(rng, n_entities)
            rec = {k: np.full(n_entities, np.nan) for k in params}
            keep = np.zeros(n_entities, dtype=bool)
            for e in range(n_entities):
                members = _member_thetas(rng, postures["loss_aversion"][e],
                                         postures["threat"][e], K)
                X, y = _aggregate_decisions(rng, members, D)
                inf = infer_posture(X, y)
                for k in params:
                    rec[k][e] = inf["ratios"][k]
                keep[e] = inf["gain_z"] >= gate_z   # gate: abstain when gain sensitivity unresolved
            abst.append(float(1.0 - keep.mean()))
            for k in params:
                # recovery across all entities
                r_all[k].append(float(stats.pearsonr(postures[k], rec[k])[0]))
                if keep.sum() >= 5:
                    r_kept[k].append(float(stats.pearsonr(postures[k][keep], rec[k][keep])[0]))
        entry = {"K": K, "D": D, "abstention_rate": float(np.mean(abst))}
        for k in params:
            entry[f"identifiability_r_all_{k}"] = float(np.mean(r_all[k]))
            entry[f"identifiability_r_kept_{k}"] = float(np.mean(r_kept[k])) if r_kept[k] else None
        per_scope[label] = entry

    # dose-response checks: identifiability falls and abstention rises S1 -> S6
    labels = [s[0] for s in SCOPES]
    id_curve = [np.mean([per_scope[l][f"identifiability_r_all_{k}"] for k in params]) for l in labels]
    ab_curve = [per_scope[l]["abstention_rate"] for l in labels]
    id_monotone = bool(id_curve[0] > id_curve[-1] and np.all(np.diff(id_curve) <= 0.08))  # broadly decreasing
    ab_monotone = bool(ab_curve[-1] > ab_curve[0] and np.all(np.diff(ab_curve) >= -0.08))  # broadly rising

    return {
        "experiment": "E6_attribution_scale_ladder",
        "design": {"n_entities": n_entities, "n_seeds": n_seeds, "gate_z": gate_z, "scopes": labels},
        "framing": ("simulation of the inverse-identifiability dose-response across attribution scopes; "
                    "latent = affective posture with a neural interpretation, never measured firing; "
                    "real president decision corpus (2 blind coders + kappa) is the deployment"),
        "per_scope": per_scope,
        "identifiability_curve": id_curve,
        "abstention_curve": ab_curve,
        "identifiability_decreases_toward_individual": id_monotone,
        "abstention_increases_toward_individual": ab_monotone,
        "dose_response_holds": bool(id_monotone and ab_monotone),
    }


if __name__ == "__main__":
    import json

    res = run_scale_ladder()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    path = os.path.join(root, "results", "e6_scale_ladder.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)

    print("E6 attribution-scale ladder (aggregate S1 -> individual S6)")
    print("  %-20s %5s %5s  %8s  %8s  %10s" % ("scope", "K", "D", "id_r(all)", "id_r(kept)", "abstain"))
    for label, K, D in SCOPES:
        e = res["per_scope"][label]
        idall = np.mean([e[f"identifiability_r_all_{k}"] for k in ["loss_aversion", "threat"]])
        kept = [e[f"identifiability_r_kept_{k}"] for k in ["loss_aversion", "threat"] if e[f"identifiability_r_kept_{k}"] is not None]
        idkept = np.mean(kept) if kept else float("nan")
        print(f"  {label:<20s} {K:5d} {D:5d}  {idall:8.3f}  {idkept:8.3f}  {e['abstention_rate']:10.3f}")
    print("\ndose-response holds (id falls, abstention rises toward individual):", res["dose_response_holds"], "->", path)
