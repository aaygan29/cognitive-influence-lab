"""Multi-seed robustness sweeps for the simulation gates.

Addresses the council finding that the recovery and manipulability gates were
each a single synthetic cohort. Re-runs them across several seeds and reports
the median and range of the headline numbers, so a reader can see the
correlations are not single-cohort flukes.

Writes results/recovery_sweep.json and results/manip_sweep.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PARAMS = ["a", "w", "beta", "alpha", "w_q", "t0"]


def recovery_sweep(seeds, n_subj=14, n_trials=300, warmup=400, samples=400, chains=2):
    import jax
    import numpyro
    numpyro.set_host_device_count(chains)
    from numpyro.infer import MCMC, NUTS
    from individuation_mcmc.ddm import simulate_subject
    from individuation_mcmc.model import hierarchical_model, pack_data
    from run_recovery import draw_population

    per_seed = []
    for s in seeds:
        pop, truth = draw_population(n_subj, s)
        subjects = [simulate_subject(th, ph, n_trials=n_trials, seed=1000 + s * 50 + i)
                    for i, (th, ph) in enumerate(pop)]
        data = pack_data(subjects)
        mcmc = MCMC(NUTS(hierarchical_model, target_accept_prob=0.9),
                    num_warmup=warmup, num_samples=samples, num_chains=chains,
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(s), pooled=False, **data)
        post = mcmc.get_samples()
        row = {}
        for p in PARAMS:
            draws = np.asarray(post[p]); mean = draws.mean(0); t = truth[p]
            lo = np.percentile(draws, 5, 0); hi = np.percentile(draws, 95, 0)
            row[p] = {"r": float(np.corrcoef(mean, t)[0, 1]),
                      "cov90": float(np.mean((t >= lo) & (t <= hi)))}
        per_seed.append({"seed": s, **row})
        print(f"[recovery] seed {s}: " +
              " ".join(f"{p} r={row[p]['r']:.2f}" for p in PARAMS))
    agg = {}
    for p in PARAMS:
        rs = [d[p]["r"] for d in per_seed]; cv = [d[p]["cov90"] for d in per_seed]
        agg[p] = {"r_median": float(np.median(rs)), "r_min": float(np.min(rs)),
                  "r_max": float(np.max(rs)), "cov90_mean": float(np.mean(cv))}
    return {"seeds": list(seeds), "per_seed": per_seed, "aggregate": agg}


def manip_sweep(seeds, bank=1000, cohort=14, n=150, cap_n=350, post_draws=8):
    from individuation_mcmc.sbi_attractor import (abc_from_bank, attractor_behavior,
        attractor_capture_dose, simulate_bank, summary)
    from scipy import stats

    per_seed = []
    for s in seeds:
        thetas, st = simulate_bank(n_sims=bank, n=n, seed=s)
        rng = np.random.default_rng(s)
        tc, sc = [], []
        for i in range(cohort):
            jt = rng.uniform(0.035, 0.075); sg = rng.uniform(0.012, 0.03)
            obs = summary(attractor_behavior(jt, sg, n=n, seed=400 + s * 30 + i))
            post, _ = abc_from_bank(obs, thetas, st, keep_frac=0.03)
            truth = attractor_capture_dose(jt, sg, n=cap_n, seed=700 + s * 30 + i)
            jm, sm = post.mean(0)
            est = attractor_capture_dose(jm, sm, n=cap_n, seed=800 + s * 30 + i)
            if not (np.isnan(truth) or np.isnan(est)):
                tc.append(truth); sc.append(est)
        rho, p = stats.spearmanr(tc, sc)
        per_seed.append({"seed": s, "n": len(tc), "spearman": float(rho),
                         "pearson": float(np.corrcoef(tc, sc)[0, 1]), "p": float(p)})
        print(f"[manip] seed {s}: n={len(tc)} spearman={rho:.2f} p={p:.4f}")
    rhos = [d["spearman"] for d in per_seed]
    return {"seeds": list(seeds), "per_seed": per_seed,
            "spearman_median": float(np.median(rhos)),
            "spearman_min": float(np.min(rhos)), "spearman_max": float(np.max(rhos))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["recovery", "manip", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    RESULTS.mkdir(exist_ok=True)
    if args.which in ("recovery", "both"):
        out = recovery_sweep(seeds)
        (RESULTS / "recovery_sweep.json").write_text(json.dumps(out, indent=2))
        print("recovery aggregate:", json.dumps(out["aggregate"], indent=2))
    if args.which in ("manip", "both"):
        out = manip_sweep(seeds)
        (RESULTS / "manip_sweep.json").write_text(json.dumps(out, indent=2))
        print("manip sweep:", json.dumps({k: out[k] for k in
              ("spearman_median", "spearman_min", "spearman_max")}, indent=2))
