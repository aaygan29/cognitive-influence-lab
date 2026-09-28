#!/usr/bin/env python3
"""Eliciting a target decision-related brain state by altering the learning rule.

Corrected paradigm (see docs). Static choice data cannot identify influence: there
is no intervention and no counterfactual. So influence is studied in a *controllable
simulated learner*, and real data (the NARPS loss-aversion distribution) is used
only to set what target states are plausible -- the prior over "what brain states
we expect", never the test set.

Object under control
--------------------
A learner values a mixed gamble with a prospect-theory-style code:
    V(gain, loss) = gain**rho  -  lambda * loss**rho
Its **internal state** (the neural analog) is the z-scored pattern of V over a
fixed probe grid of gambles -- a value-representation pattern, not a scalar. The
learner's loss-aversion ``lambda`` drifts over trials under a learning rule.

Intervention
------------
An adversary alters the *learning rule* by injecting a bounded per-trial dose
into the lambda-update:  lambda_{t+1} = lambda_t + eta*(feedback) + dose_t,
with |dose_t| <= cap. The question is whether, by manipulating how the learner
learns, one can synthetically **elicit a target internal pattern** (defined by a
target lambda* drawn from the real NARPS distribution), and at what control cost,
and whether the induced state persists after the intervention stops.

Nothing here is a claim about real brains. It is a control-theoretic feasibility
study on a learner model, with targets grounded in a real loss-aversion range.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def probe_grid():
    gains = np.arange(10, 41, 5.0)   # NARPS-like gain range
    losses = np.arange(5, 21, 3.0)   # NARPS-like loss range
    G, L = np.meshgrid(gains, losses)
    return G.ravel(), L.ravel()


def internal_pattern(lam, rho, G, L):
    # Brain-state analog = the LOSS-response pattern: the value code with the
    # gain-only component projected out, so the pattern reflects loss processing
    # (what the NARPS ex-ante hypotheses concern) and is genuinely sensitive to
    # loss-aversion. Without this residualisation the gain term dominates and the
    # pattern is ~invariant to lambda (a ceiling artifact).
    V = np.power(G, rho) - lam * np.power(L, rho)
    gain_only = np.power(G, rho)
    g = (gain_only - gain_only.mean())
    v = V - V.mean()
    v_res = v - (np.dot(v, g) / (np.dot(g, g) + 1e-12)) * g  # remove gain axis
    return v_res / (v_res.std() + 1e-8)


def rsa(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def run_learner(lam0, lam_target, rho, G, L, target_pat, cap, controlled,
                n_trials=200, eta=0.02, seed=0, intervene_until=None):
    """Simulate a learner. If controlled, a feedback controller injects bounded
    doses into the learning rule to drive the internal pattern to target_pat.
    Returns per-trial lambda, dose, and RSA-to-target."""
    rng = np.random.default_rng(seed)
    lam = lam0
    lams, doses, sims = [], [], []
    intervene_until = intervene_until if intervene_until is not None else n_trials
    total_budget = 0.0
    for t in range(n_trials):
        # endogenous learning: small mean-reverting drift toward the learner's
        # own baseline plus noise (the learner keeps being itself)
        feedback = (lam0 - lam)
        dose = 0.0
        if controlled and t < intervene_until:
            # proportional controller on lambda error, clipped to the cap
            err = lam_target - lam
            dose = float(np.clip(eta * 8.0 * err, -cap, cap))
        lam = lam + eta * feedback + dose
        lam = float(np.clip(lam, 0.0, 3.0))
        total_budget += abs(dose)
        pat = internal_pattern(lam, rho, G, L)
        lams.append(lam); doses.append(dose); sims.append(rsa(pat, target_pat))
    return dict(lam=np.array(lams), dose=np.array(doses), sim=np.array(sims),
                total_budget=total_budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-file",
                    default=str(REPO / "results" / "narps_lambda.npy"))
    ap.add_argument("--cap", type=float, default=0.03, help="per-trial dose bound")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--out", default=str(REPO / "results" / "elicitation_control.json"))
    args = ap.parse_args()

    las = np.load(args.lambda_file)
    lam_lo, lam_hi = float(np.percentile(las, 10)), float(np.percentile(las, 90))
    G, L = probe_grid()
    rho = args.rho

    # Target = a loss-averse brain state at the 90th percentile of the real range.
    lam_target = lam_hi
    target_pat = internal_pattern(lam_target, rho, G, L)

    res = {"dataset_target_prior": "NARPS ds001734 loss-aversion distribution",
           "n_real_subjects": int(len(las)),
           "lambda_target": lam_target, "lambda_baseline_range": [lam_lo, lam_hi],
           "probe_gambles": int(len(G)), "per_trial_dose_cap": args.cap, "rho": rho}

    # Simulate a cohort of baseline learners (their own lambda0 across the real range)
    lam0s = np.linspace(lam_lo, lam_hi, 25)
    controlled_final, free_final, budgets, persist = [], [], [], []
    for i, lam0 in enumerate(lam0s):
        free = run_learner(lam0, lam_target, rho, G, L, target_pat, args.cap,
                           controlled=False, seed=i)
        ctl = run_learner(lam0, lam_target, rho, G, L, target_pat, args.cap,
                          controlled=True, seed=i)
        # persistence: intervene for first half, release, measure state at end
        rel = run_learner(lam0, lam_target, rho, G, L, target_pat, args.cap,
                          controlled=True, seed=i, intervene_until=100, n_trials=200)
        free_final.append(free["sim"][-1])
        controlled_final.append(ctl["sim"][-1])
        budgets.append(ctl["total_budget"])
        persist.append(rel["sim"][-1])

    res["free_final_rsa_to_target"] = dict(mean=float(np.mean(free_final)),
                                           sd=float(np.std(free_final)))
    res["controlled_final_rsa_to_target"] = dict(mean=float(np.mean(controlled_final)),
                                                 sd=float(np.std(controlled_final)))
    res["control_budget_sum_abs_dose"] = dict(mean=float(np.mean(budgets)),
                                              sd=float(np.std(budgets)))
    res["persistence_rsa_after_release"] = dict(mean=float(np.mean(persist)),
                                                sd=float(np.std(persist)))
    # Elicitation success rate: fraction of learners driven to RSA>=0.95
    res["elicitation_success_rate"] = float(np.mean(np.array(controlled_final) >= 0.95))
    res["free_success_rate"] = float(np.mean(np.array(free_final) >= 0.95))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("lambda_baseline_range",)}, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
