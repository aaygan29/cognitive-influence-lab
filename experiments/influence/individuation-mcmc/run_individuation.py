"""Individuation gate: does a per-person fit beat a group fit out-of-sample?

A posterior over per-subject parameters is only an individuation tool if it
predicts that subject's held-out behavior better than a single pooled model, and
better than the same individual parameters assigned to the wrong subject. This
script fits the individual and pooled models on each subject's training trials
and compares three predictors on held-out test trials:

  * individual : each subject scored under their own posterior-mean parameters;
  * pooled     : every subject scored under the shared group parameters;
  * shuffled   : each subject scored under another subject's individual
                 parameters (identity-shuffle control).

The metric is mean held-out log-likelihood per trial (higher is better). For
individuation to hold: individual > pooled and individual > shuffled.

Writes results/individuation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _fit_means(data, pooled, key, warmup, samples, chains):
    import jax
    from numpyro.infer import MCMC, NUTS
    from individuation_mcmc.model import hierarchical_model
    kernel = NUTS(hierarchical_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=chains, progress_bar=False)
    mcmc.run(key, pooled=pooled, **data)
    post = mcmc.get_samples()
    return {p: np.asarray(post[p]).mean(axis=0) for p in
            ("a", "w", "beta", "alpha", "w_q", "t0")}


def _test_loglik(subjects, means, test_sl, assign):
    """Per-subject mean test log-lik, scoring subject i with params assign[i]."""
    import jax.numpy as jnp
    from individuation_mcmc.ddm import wfpt_logpdf
    from individuation_mcmc.model import _rw_value_diff
    out = []
    for i, s in enumerate(subjects):
        j = assign[i]
        a, w, beta = means["a"][j], means["w"][j], means["beta"][j]
        alpha, wq, t0 = means["alpha"][j], means["w_q"][j], means["t0"][j]
        # t0 must be below this subject's own fastest test RT to stay valid.
        t0 = float(min(t0, 0.95 * s["rt"][test_sl].min()))
        choice = jnp.asarray(s["choice"][None, :].astype(float))
        reward = jnp.asarray(s["reward"][None, :].astype(float))
        cue = s["cue"]
        qdiff = np.asarray(_rw_value_diff(jnp.asarray([alpha]), choice, reward))[0]
        drift = wq * qdiff + beta * cue
        lp = np.asarray(wfpt_logpdf(s["rt"], s["choice"].astype(float),
                                    drift, a, w, t0))
        out.append(float(np.mean(lp[test_sl])))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=14)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import numpyro
    numpyro.set_host_device_count(args.chains)
    from individuation_mcmc.ddm import simulate_subject
    from individuation_mcmc.model import pack_data
    from run_recovery import draw_population

    RESULTS.mkdir(exist_ok=True)

    pop, _ = draw_population(args.subjects, args.seed)
    subjects = [simulate_subject(th, ph, n_trials=args.trials, seed=2000 + i)
                for i, (th, ph) in enumerate(pop)]

    T = args.trials
    cut = int(0.7 * T)
    train_sl = slice(0, cut)
    test_sl = slice(cut, T)
    train = pack_data(subjects, sl=train_sl)

    k1, k2 = jax.random.split(jax.random.PRNGKey(args.seed))
    ind = _fit_means(train, False, k1, args.warmup, args.samples, args.chains)
    grp = _fit_means(train, True, k2, args.warmup, args.samples, args.chains)

    rng = np.random.default_rng(args.seed)
    shuffle = rng.permutation(args.subjects)
    # ensure no subject keeps its own index
    for i in range(args.subjects):
        if shuffle[i] == i:
            shuffle[i] = (i + 1) % args.subjects

    ll_ind = _test_loglik(subjects, ind, test_sl, np.arange(args.subjects))
    ll_grp = _test_loglik(subjects, grp, test_sl, np.arange(args.subjects))
    ll_shuf = _test_loglik(subjects, ind, test_sl, shuffle)

    def ci(x):
        return {"mean": float(np.mean(x)), "sem": float(np.std(x) / np.sqrt(len(x)))}

    summary = {
        "n_subjects": args.subjects, "n_trials": args.trials,
        "train_trials": cut, "test_trials": T - cut,
        "held_out_loglik_per_trial": {
            "individual": ci(ll_ind), "pooled": ci(ll_grp), "shuffled": ci(ll_shuf)},
        "individual_minus_pooled": {
            "mean": float(np.mean(ll_ind - ll_grp)),
            "n_subjects_better": int(np.sum(ll_ind > ll_grp))},
        "individual_minus_shuffled": {
            "mean": float(np.mean(ll_ind - ll_shuf)),
            "n_subjects_better": int(np.sum(ll_ind > ll_shuf))},
    }
    (RESULTS / "individuation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
