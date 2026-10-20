"""Synthetic parameter recovery: can NUTS recover known theta_i and phi_i?

This is the first validation gate. We draw a population of subjects with known
state vectors and learning rules, simulate their choices and reaction times, fit
the hierarchical model, and check (a) how well posterior means track the true
per-subject values and (b) whether the 90% posterior intervals have ~90%
coverage. A model whose posteriors are miscalibrated cannot be trusted as an
individuation tool, so this runs before any downstream claim.

Writes results/recovery.json and figures/recovery.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

PARAMS = ["a", "w", "beta", "alpha", "w_q", "t0"]


def draw_population(n_subj, seed):
    rng = np.random.default_rng(seed)
    subs = []
    truth = {p: [] for p in PARAMS}
    for _ in range(n_subj):
        a = float(np.exp(rng.normal(np.log(1.2), 0.25)))
        w = float(np.clip(rng.normal(0.5, 0.08), 0.2, 0.8))
        beta = float(rng.normal(1.1, 0.35))
        alpha = float(np.clip(rng.beta(2, 3), 0.02, 0.95))
        w_q = float(rng.normal(0.6, 0.3))
        t0 = float(np.clip(rng.normal(0.22, 0.05), 0.08, 0.4))
        truth["a"].append(a); truth["w"].append(w); truth["beta"].append(beta)
        truth["alpha"].append(alpha); truth["w_q"].append(w_q); truth["t0"].append(t0)
        subs.append((dict(a=a, t0=t0, w=w, beta=beta), dict(alpha=alpha, w_q=w_q)))
    return subs, {p: np.array(v) for p, v in truth.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS
    numpyro.set_host_device_count(args.chains)

    from individuation_mcmc.ddm import simulate_subject
    from individuation_mcmc.model import hierarchical_model, pack_data

    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

    pop, truth = draw_population(args.subjects, args.seed)
    subjects = [simulate_subject(th, ph, n_trials=args.trials, seed=1000 + i)
                for i, (th, ph) in enumerate(pop)]
    data = pack_data(subjects)

    kernel = NUTS(hierarchical_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(args.seed), pooled=False, **data)
    post = mcmc.get_samples()

    recovery = {}
    for p in PARAMS:
        draws = np.asarray(post[p])             # (S, J)
        mean = draws.mean(axis=0)
        lo = np.percentile(draws, 5, axis=0)
        hi = np.percentile(draws, 95, axis=0)
        t = truth[p]
        r = float(np.corrcoef(mean, t)[0, 1]) if np.std(mean) > 0 else float("nan")
        coverage = float(np.mean((t >= lo) & (t <= hi)))
        rmse = float(np.sqrt(np.mean((mean - t) ** 2)))
        recovery[p] = {"recovery_corr": r, "coverage_90": coverage, "rmse": rmse,
                       "true": t.tolist(), "post_mean": mean.tolist()}

    # Convergence diagnostics on the group means.
    import arviz as az
    idata = az.from_numpyro(mcmc)
    group_vars = [v for v in ["mu_log_a", "mu_beta", "mu_logit_alpha", "mu_wq"]
                  if v in idata.posterior]
    rhat = az.rhat(idata, var_names=group_vars)
    max_rhat = float(max(float(rhat[v].max()) for v in group_vars))

    summary = {
        "n_subjects": args.subjects, "n_trials": args.trials,
        "nuts": {"warmup": args.warmup, "samples": args.samples, "chains": args.chains},
        "max_rhat_group_means": max_rhat,
        "recovery": {p: {k: recovery[p][k] for k in
                         ("recovery_corr", "coverage_90", "rmse")} for p in PARAMS},
    }
    (RESULTS / "recovery.json").write_text(json.dumps(
        {**summary, "detail": recovery}, indent=2))
    print(json.dumps(summary, indent=2))

    _figure(recovery)


def _figure(recovery) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, p in zip(axes.ravel(), PARAMS):
        t = np.array(recovery[p]["true"]); m = np.array(recovery[p]["post_mean"])
        lo = hi = None
        ax.scatter(t, m, s=28, color="C0")
        lim = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"{p}  (r={recovery[p]['recovery_corr']:.2f}, "
                     f"cov={recovery[p]['coverage_90']:.2f})", fontsize=9)
        ax.set_xlabel("true"); ax.set_ylabel("posterior mean")
    fig.suptitle("Per-subject parameter recovery (truth vs posterior mean)")
    fig.tight_layout()
    fig.savefig(FIGS / "recovery.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
