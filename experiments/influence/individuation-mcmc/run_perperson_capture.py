"""Per-person capture dose and stealth floor from fitted posteriors.

Fits the hierarchical model to the synthetic cohort, then for each subject uses
the posterior over (a, w, t0) to compute:

  * the capture dose (posterior median and 90% interval), analytic across many
    posterior draws;
  * the stealth floor (at the posterior mean), Monte-Carlo;
  * the per-person covert gap = stealth floor - capture dose.

This is the individuation -> manipulability bridge: the same person-specific
posterior that predicts choices also yields how much covert steering would flip
one of their decisions, and whether that steering is detectable. We also report
how capture dose varies across people and tracks the fitted boundary, so the
manipulability is shown to be individual, not a shared constant.

Writes results/perperson_capture.json and figures/perperson_capture.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

DOSES = np.linspace(0.0, 4.0, 17)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=14)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import numpyro
    numpyro.set_host_device_count(args.chains)
    from numpyro.infer import MCMC, NUTS
    from individuation_mcmc.ddm import simulate_subject
    from individuation_mcmc.model import hierarchical_model, pack_data
    from individuation_mcmc.steering import capture_dose, stealth_floor
    from run_recovery import draw_population

    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

    pop, truth = draw_population(args.subjects, args.seed)
    subjects = [simulate_subject(th, ph, n_trials=args.trials, seed=3000 + i)
                for i, (th, ph) in enumerate(pop)]
    data = pack_data(subjects)

    mcmc = MCMC(NUTS(hierarchical_model, target_accept_prob=0.9),
                num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(args.seed), pooled=False, **data)
    post = mcmc.get_samples()
    a_s = np.asarray(post["a"]); w_s = np.asarray(post["w"]); t0_s = np.asarray(post["t0"])

    windows = [1, 5, 10]
    per = []
    for j in range(args.subjects):
        # Capture dose across posterior draws (subsample for speed).
        idx = np.linspace(0, a_s.shape[0] - 1, 200).astype(int)
        cds = np.array([capture_dose(a_s[s, j], w_s[s, j], args.target)
                        for s in idx])
        cds = cds[~np.isnan(cds)]
        a_m, w_m, t0_m = a_s[:, j].mean(), w_s[:, j].mean(), t0_s[:, j].mean()
        cd_med = float(np.median(cds)) if cds.size else float("nan")
        floors, gaps = {}, {}
        for win in windows:
            fl, _ = stealth_floor(a_m, w_m, t0_m, DOSES, n=6000,
                                  seed=args.seed + j, window=win)
            floors[win] = fl
            gaps[win] = (fl - cd_med) if (not np.isnan(fl) and not np.isnan(cd_med)) \
                else float("nan")
        per.append({
            "subject": j,
            "honest_baseline_approve": float(w_m),
            "capture_dose_median": cd_med,
            "capture_dose_90ci": [float(np.percentile(cds, 5)),
                                  float(np.percentile(cds, 95))] if cds.size else None,
            "stealth_floor_by_window": {str(k): v for k, v in floors.items()},
            "gap_by_window": {str(k): v for k, v in gaps.items()},
            "fitted_a": float(a_m),
        })

    caps = np.array([p["capture_dose_median"] for p in per], dtype=float)
    a_fit = np.array([p["fitted_a"] for p in per], dtype=float)
    gap_by_win = {}
    for win in windows:
        g = np.array([p["gap_by_window"][str(win)] for p in per], dtype=float)
        fin = g[~np.isnan(g)]
        gap_by_win[str(win)] = {
            "gap_median": float(np.nanmedian(g)) if fin.size else None,
            "gap_positive_fraction": float(np.mean(fin > 0)) if fin.size else None,
            "n_floor_unreached": int(np.sum(np.isnan(g))),
        }
    summary = {
        "n_subjects": args.subjects, "target_capture": args.target,
        "monitor_windows": windows,
        "capture_dose": {"median_across_subjects": float(np.nanmedian(caps)),
                         "min": float(np.nanmin(caps)), "max": float(np.nanmax(caps)),
                         "spread_ratio_max_over_min": float(np.nanmax(caps) / np.nanmin(caps))},
        "corr_capture_vs_fitted_boundary": float(np.corrcoef(caps, a_fit)[0, 1]),
        "gap_by_window": gap_by_win,
        "per_subject": per,
    }
    (RESULTS / "perperson_capture.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("capture_dose", "corr_capture_vs_fitted_boundary",
                       "gap_by_window")}, indent=2))

    _figure(per)


def _figure(per) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = [p for p in per if not np.isnan(p["capture_dose_median"])]
    order = sorted(range(len(per)), key=lambda i: per[i]["capture_dose_median"])
    caps = [per[i]["capture_dose_median"] for i in order]
    los = [per[i]["capture_dose_90ci"][0] for i in order]
    his = [per[i]["capture_dose_90ci"][1] for i in order]
    floors = [per[i]["stealth_floor_by_window"].get("5", float("nan")) for i in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].errorbar(x, caps, yerr=[np.array(caps) - np.array(los),
                                  np.array(his) - np.array(caps)],
                   fmt="o", color="C0", label="capture dose (90% CI)")
    ax[0].plot(x, floors, "s", color="C3", label="stealth floor (window=5)")
    ax[0].set_xlabel("subject (sorted by capture dose)")
    ax[0].set_ylabel("added drift (persuasion dose)")
    ax[0].set_title("Per-person capture dose and stealth floor")
    ax[0].legend(fontsize=8)
    a_fit = np.array([p["fitted_a"] for p in per])
    capv = np.array([p["capture_dose_median"] for p in per])
    ax[1].scatter(a_fit, capv, color="C0")
    ax[1].set_xlabel("fitted boundary a")
    ax[1].set_ylabel("capture dose")
    ax[1].set_title(f"Manipulability tracks the fitted model "
                    f"(r={np.corrcoef(capv, a_fit)[0,1]:.2f})")
    fig.tight_layout()
    fig.savefig(FIGS / "perperson_capture.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
