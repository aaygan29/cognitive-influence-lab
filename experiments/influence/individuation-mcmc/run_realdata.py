"""Real-data test: individuation on a probabilistic reversal-learning dataset.

What is falsifiable on real human data is individuation across sessions: does a
model fit to a person in session 1 predict that same person in session 2 better
than a pooled group model and better than another person's fit? This is the
test-retest design of the Waltmann et al. (2022) dataset, so it is the right
real-data check. Manipulability (capture dose) is NOT validatable here: you
cannot covertly steer a real participant to measure a ground-truth capture dose,
so any capture dose from these fits is an unvalidated model readout and is not
computed in this script.

Two results:
  * Test-retest reliability: Pearson correlation of each fitted parameter
    between the two sessions, across subjects.
  * Cross-session individuation: for each subject, session-2 mean log-likelihood
    under their own session-1 fit vs a pooled group fit vs another subject's
    session-1 fit (identity-shuffle).

Usage: run_realdata.py /path/to/RL_Data_Retest_R_wo.txt
Writes results/realdata.json and figures/realdata.png. The dataset itself is not
stored in the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from individuation_mcmc.realdata import fit_subject, load_waltmann, mean_loglik

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
PARAMS = ["a", "w", "t0", "v0", "w_q", "alpha"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_realdata.py /path/to/RL_Data_Retest_R_wo.txt")
    data_path = sys.argv[1]
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

    data = load_waltmann(data_path)
    by = {(d["sub"], d["session"]): d for d in data}
    subs = sorted({s for s, _ in by})
    subs = [s for s in subs if (s, "1") in by and (s, "2") in by]
    print(f"subjects with both sessions: {len(subs)}")

    fit1, fit2 = {}, {}
    for i, s in enumerate(subs):
        fit1[s] = fit_subject(by[(s, "1")], seed=i)
        fit2[s] = fit_subject(by[(s, "2")], seed=1000 + i)
        if (i + 1) % 25 == 0:
            print(f"  fitted {i+1}/{len(subs)}")

    # Test-retest reliability of each parameter.
    reliability = {}
    for p in PARAMS:
        x = np.array([fit1[s][p] for s in subs])
        y = np.array([fit2[s][p] for s in subs])
        reliability[p] = float(np.corrcoef(x, y)[0, 1])

    # Cross-session individuation: predict session 2 from session-1 fits.
    pooled = {p: float(np.mean([fit1[s][p] for s in subs])) for p in PARAMS}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(subs))
    for i in range(len(subs)):
        if perm[i] == i:
            perm[i] = (i + 1) % len(subs)

    ll_ind, ll_grp, ll_shuf = [], [], []
    for i, s in enumerate(subs):
        tgt = by[(s, "2")]
        ll_ind.append(mean_loglik(fit1[s], tgt))
        ll_grp.append(mean_loglik(pooled, tgt))
        ll_shuf.append(mean_loglik(fit1[subs[perm[i]]], tgt))
    ll_ind = np.array(ll_ind); ll_grp = np.array(ll_grp); ll_shuf = np.array(ll_shuf)

    def ci(x):
        return {"mean": float(np.mean(x)), "sem": float(np.std(x) / np.sqrt(len(x)))}

    summary = {
        "dataset": "Waltmann et al. 2022 probabilistic reversal learning (CC-BY), "
                   "Reading reliability redistribution",
        "n_subjects_both_sessions": len(subs),
        "test_retest_reliability_pearson": reliability,
        "cross_session_individuation": {
            "session2_loglik_per_trial": {
                "own_session1_fit": ci(ll_ind),
                "pooled_group_fit": ci(ll_grp),
                "other_subject_fit_shuffled": ci(ll_shuf)},
            "individual_minus_pooled": {
                "mean": float(np.mean(ll_ind - ll_grp)),
                "n_subjects_better": int(np.sum(ll_ind > ll_grp))},
            "individual_minus_shuffled": {
                "mean": float(np.mean(ll_ind - ll_shuf)),
                "n_subjects_better": int(np.sum(ll_ind > ll_shuf))},
        },
    }
    (RESULTS / "realdata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"reliability": reliability,
                      "individuation": summary["cross_session_individuation"]
                      ["individual_minus_shuffled"]}, indent=2))

    _figure(subs, fit1, fit2, ll_ind, ll_grp, ll_shuf, reliability)


def _figure(subs, fit1, fit2, ll_ind, ll_grp, ll_shuf, reliability) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    p = "alpha"
    x = [fit1[s][p] for s in subs]; y = [fit2[s][p] for s in subs]
    ax[0].scatter(x, y, s=20, alpha=0.6, color="C0")
    lim = [min(min(x), min(y)), max(max(x), max(y))]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("session 1 learning rate"); ax[0].set_ylabel("session 2 learning rate")
    ax[0].set_title(f"Test-retest of learning rate (r={reliability['alpha']:.2f})")
    means = [ll_ind.mean(), ll_grp.mean(), ll_shuf.mean()]
    sems = [np.std(v) / np.sqrt(len(v)) for v in (ll_ind, ll_grp, ll_shuf)]
    ax[1].bar(["own\nsession-1", "pooled\ngroup", "other\nsubject"], means,
              yerr=sems, color=["C0", "C1", "C3"])
    ax[1].set_ylabel("session-2 log-lik / trial")
    ax[1].set_title("Cross-session individuation")
    fig.tight_layout()
    fig.savefig(FIGS / "realdata.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
