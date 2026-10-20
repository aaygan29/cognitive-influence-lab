"""Per-person manipulability computed in the full circuit via SBI.

Gate 4 showed the DDM-reduction capture dose does not transfer to the full
attractor circuit. This is the fix it points to: for each subject, infer the
full-circuit parameters from behavior with ABC, then compute the capture dose
*in the attractor itself* from the inferred parameters, propagating posterior
uncertainty. No reduction is used for the manipulability step.

The validation is the loop the reduction failed: does the SBI-inferred capture
dose track the ground-truth capture dose (the one computed from each subject's
true parameters), and does the true value fall inside the posterior interval?

Writes results/sbi_manipulability.json and figures/sbi_manipulability.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from individuation_mcmc.sbi_attractor import (abc_from_bank, attractor_behavior,
    attractor_capture_dose, simulate_bank, summary)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def _spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=1200)
    ap.add_argument("--cohort", type=int, default=14)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--cap-n", type=int, default=350)
    ap.add_argument("--post-draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    thetas, stats = simulate_bank(n_sims=args.bank, n=args.n, seed=args.seed)

    per = []
    for i in range(args.cohort):
        jt = rng.uniform(0.035, 0.075); st = rng.uniform(0.012, 0.03)
        obs = summary(attractor_behavior(jt, st, n=args.n, seed=400 + i))
        post, _ = abc_from_bank(obs, thetas, stats, keep_frac=0.03)

        # Ground-truth capture dose from the true parameters.
        truth_cd = attractor_capture_dose(jt, st, n=args.cap_n, seed=700 + i)
        # Point estimate at the posterior mean.
        jm, sm = post.mean(axis=0)
        sbi_cd = attractor_capture_dose(jm, sm, n=args.cap_n, seed=800 + i)
        # Posterior interval by propagating a subsample of posterior draws.
        idx = rng.choice(len(post), size=min(args.post_draws, len(post)), replace=False)
        draws = [attractor_capture_dose(post[k, 0], post[k, 1], n=args.cap_n,
                                        seed=900 + i * 50 + k) for k in idx]
        draws = np.array([d for d in draws if not np.isnan(d)])
        ci = ([float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]
              if draws.size >= 3 else None)
        per.append({
            "subject": i,
            "true_J_cross": jt, "true_sigma": st,
            "post_mean_J_cross": float(jm), "post_mean_sigma": float(sm),
            "truth_capture_dose": truth_cd,
            "sbi_capture_dose": sbi_cd,
            "sbi_capture_ci": ci,
        })

    ok = [p for p in per if not (np.isnan(p["truth_capture_dose"])
                                 or np.isnan(p["sbi_capture_dose"]))]
    tc = np.array([p["truth_capture_dose"] for p in ok])
    sc = np.array([p["sbi_capture_dose"] for p in ok])
    covered = [p for p in ok if p["sbi_capture_ci"] and
               p["sbi_capture_ci"][0] <= p["truth_capture_dose"] <= p["sbi_capture_ci"][1]]
    summary_out = {
        "bank_size": args.bank, "cohort": args.cohort, "n_evaluable": len(ok),
        "spearman_sbi_vs_truth": _spearman(sc, tc),
        "pearson_sbi_vs_truth": float(np.corrcoef(sc, tc)[0, 1]),
        "median_abs_error_capture_dose": float(np.median(np.abs(sc - tc))),
        "ci_coverage_90": len(covered) / len(ok) if ok else None,
        "per_subject": per,
    }
    (RESULTS / "sbi_manipulability.json").write_text(json.dumps(summary_out, indent=2))
    print(json.dumps({k: summary_out[k] for k in
                      ("n_evaluable", "spearman_sbi_vs_truth", "pearson_sbi_vs_truth",
                       "median_abs_error_capture_dose", "ci_coverage_90")}, indent=2))

    _figure(ok)


def _figure(per) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tc = np.array([p["truth_capture_dose"] for p in per])
    sc = np.array([p["sbi_capture_dose"] for p in per])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for p in per:
        if p["sbi_capture_ci"]:
            ax.plot([p["truth_capture_dose"]] * 2, p["sbi_capture_ci"],
                    color="C0", alpha=0.4, lw=1)
    ax.scatter(tc, sc, color="C0", zorder=3)
    lim = [min(tc.min(), sc.min()), max(tc.max(), sc.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("ground-truth capture dose (nA)")
    ax.set_ylabel("SBI-inferred capture dose (nA)")
    ax.set_title("Full-circuit manipulability via SBI\n(inferred vs true, 90% CI)")
    fig.tight_layout()
    fig.savefig(FIGS / "sbi_manipulability.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
