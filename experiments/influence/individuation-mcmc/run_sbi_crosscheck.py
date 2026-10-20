"""SBI cross-check: validate the DDM-reduction route against the full circuit.

Two parts:

  A. Likelihood-free recovery. Simulate a prior bank from the full attractor
     circuit and run rejection ABC to infer the circuit's own parameters
     (J_cross, sigma_noise) for held-out subjects. If ABC recovers them, the
     full nonlinear circuit is identifiable from behavior without a likelihood.

  B. Manipulability cross-check. For a cohort of circuit settings, compare the
     capture dose measured directly in the full attractor circuit (ground truth)
     with the capture dose predicted by the fast DDM-reduction route (an
     ML-fitted DDM on the same behavior). A high rank correlation means the
     cheap analytic route ranks who is manipulable the same way the expensive
     nonlinear circuit does.

Writes results/sbi_crosscheck.json and figures/sbi_crosscheck.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from individuation_mcmc.sbi_attractor import (COHERENCES, abc_from_bank,
    attractor_behavior, attractor_capture_dose, simulate_bank, summary)
from individuation_mcmc.steering import capture_dose, ml_fit_ddm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def _spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=1200)
    ap.add_argument("--recovery-subjects", type=int, default=6)
    ap.add_argument("--cohort", type=int, default=14)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Part A: ABC recovery against a shared prior bank. ----
    thetas, stats = simulate_bank(n_sims=args.bank, n=args.n, seed=args.seed)
    true, est = [], []
    for i in range(args.recovery_subjects):
        jt = rng.uniform(0.03, 0.08); st = rng.uniform(0.01, 0.035)
        obs = summary(attractor_behavior(jt, st, n=args.n, seed=50 + i))
        post, _ = abc_from_bank(obs, thetas, stats, keep_frac=0.03)
        true.append((jt, st)); est.append(post.mean(axis=0))
    true = np.array(true); est = np.array(est)
    rec = {
        "J_cross": {"recovery_corr": _spearman(true[:, 0], est[:, 0]),
                    "rmse": float(np.sqrt(np.mean((true[:, 0] - est[:, 0]) ** 2)))},
        "sigma_noise": {"recovery_corr": _spearman(true[:, 1], est[:, 1]),
                        "rmse": float(np.sqrt(np.mean((true[:, 1] - est[:, 1]) ** 2)))},
    }

    # ---- Part B: capture-dose cross-check. ----
    att_cd, ddm_cd = [], []
    for i in range(args.cohort):
        j = rng.uniform(0.035, 0.075); s = rng.uniform(0.012, 0.03)
        acd = attractor_capture_dose(j, s, n=400, seed=200 + i)
        beh = attractor_behavior(j, s, n=250, seed=300 + i)
        rt = np.concatenate([beh[c][1] for c in COHERENCES])
        ch = np.concatenate([beh[c][0] for c in COHERENCES])
        coh = np.concatenate([np.full(len(beh[c][0]), c) for c in COHERENCES])
        ok = ~np.isnan(rt)
        fit = ml_fit_ddm(rt[ok], ch[ok], coh[ok])
        dcd = capture_dose(fit["a"], fit["w"], 0.75)
        if not (np.isnan(acd) or np.isnan(dcd)):
            att_cd.append(acd); ddm_cd.append(dcd)
    att_cd = np.array(att_cd); ddm_cd = np.array(ddm_cd)
    spear = _spearman(att_cd, ddm_cd)
    pear = float(np.corrcoef(att_cd, ddm_cd)[0, 1])

    summary_out = {
        "bank_size": args.bank, "n_trials_per_cond": args.n,
        "abc_recovery": rec,
        "capture_dose_crosscheck": {
            "n_subjects": int(len(att_cd)),
            "spearman_attractor_vs_ddm": spear,
            "pearson_attractor_vs_ddm": pear,
            "attractor_capture_dose_nA": att_cd.tolist(),
            "ddm_capture_dose": ddm_cd.tolist(),
        },
    }
    (RESULTS / "sbi_crosscheck.json").write_text(json.dumps(summary_out, indent=2))
    print(json.dumps({"abc_recovery": rec,
                      "capture_dose_spearman": spear,
                      "capture_dose_pearson": pear,
                      "n": int(len(att_cd))}, indent=2))

    _figure(true, est, att_cd, ddm_cd, spear)


def _figure(true, est, att_cd, ddm_cd, spear) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].scatter(true[:, 0], est[:, 0], color="C0")
    lim = [true[:, 0].min(), true[:, 0].max()]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("true J_cross"); ax[0].set_ylabel("ABC posterior mean")
    ax[0].set_title("A. Likelihood-free recovery of circuit J_cross")
    ax[1].scatter(att_cd, ddm_cd, color="C2")
    ax[1].set_xlabel("full-attractor capture dose (nA)")
    ax[1].set_ylabel("DDM-route capture dose")
    ax[1].set_title(f"B. Cross-check (Spearman r={spear:.2f})")
    fig.tight_layout()
    fig.savefig(FIGS / "sbi_crosscheck.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
