"""Longitudinal detection under representational drift, with and without alignment.

Builds a reference session and several later sessions whose latent-to-neural code
has drifted, then measures detection (manip vs honest-hard) across sessions under
three deployments: per-trial axis estimation, a fixed reference axis, and a fixed
axis carried forward by orthogonal-Procrustes alignment of honest subspaces.

Outputs results/drift.json and figures/drift_overview.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import configs as C
from bellwether.detect import roc_auc
from bellwether.drift import (
    aligned_axis,
    drifted_readout,
    honest_subspace,
    project_fixed,
    reference_axis,
)
from bellwether.dynamics import Adversary, Stimulus, simulate
from bellwether.estimator import mean_directedness

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
DRIFT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
DIM = 6
N = 150
SMOOTH = 3


I0 = int(0.5 / C.DT)
HORIZON = int(2.3 / C.DT)


def trial_directedness(rates, axis=None):
    """Per-trial geometry score, using a fixed axis or per-trial PCA when axis is None."""
    out = []
    for j in range(rates.shape[1]):
        z = _pca_axis(rates[:, j, :]) if axis is None else project_fixed(rates[:, j, :], axis)
        out.append(mean_directedness(z, I0, HORIZON, smooth=SMOOTH))
    return np.array(out)


def _pca_axis(rates_1trial):
    X = rates_1trial - rates_1trial.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    z = X @ Vt[0]
    s = z.std()
    return z / s if s > 0 else z


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    base = C.make_readout()

    # Reference session (no drift): learn the fixed axis and honest subspace.
    ref_hard = simulate(Stimulus(coherence=0.0), n_trials=250, duration=C.DURATION,
                        dt=C.DT, seed=300)
    ref_axis = reference_axis(base, simulate(Stimulus(coherence=6.4), n_trials=250,
                              duration=C.DURATION, dt=C.DT, seed=301))
    V_ref = honest_subspace(base, ref_hard, dim=DIM)

    # Each drift level is evaluated over several independent drift realizations,
    # because a single drift target is not representative: how much a fixed axis
    # degrades depends on which direction the code happens to drift.
    drift_seeds = [111, 222, 333, 444, 555]
    stats = {k: {"mean": [], "std": []} for k in ("per_trial", "fixed", "aligned")}
    for lvl in tqdm(DRIFT_LEVELS, desc="drift levels"):
        per_seed = {"per_trial": [], "fixed": [], "aligned": []}
        for ds in drift_seeds:
            ro = base if lvl == 0 else drifted_readout(base, drift_level=lvl,
                                                        dropout=0.1, seed=ds)
            hard = ro.project(simulate(Stimulus(coherence=0.0), n_trials=N,
                                       duration=C.DURATION, dt=C.DT, seed=400))
            manip = ro.project(simulate(Stimulus(coherence=0.0),
                                        Adversary(amplitude=0.012), n_trials=N,
                                        duration=C.DURATION, dt=C.DT, seed=401))
            anchors = ro.project(simulate(Stimulus(coherence=0.0), n_trials=120,
                                          duration=C.DURATION, dt=C.DT, seed=402))
            a_al = aligned_axis(ref_axis, V_ref, _subspace_from_rates(anchors, DIM))
            sh = {ax_name: trial_directedness(hard, ax)
                  for ax_name, ax in [("per_trial", None), ("fixed", ref_axis),
                                      ("aligned", a_al)]}
            sm = {ax_name: trial_directedness(manip, ax)
                  for ax_name, ax in [("per_trial", None), ("fixed", ref_axis),
                                      ("aligned", a_al)]}
            for k in per_seed:
                per_seed[k].append(roc_auc(sm[k], sh[k]))
            if lvl == 0.0:
                break   # no drift: realizations are identical
        for k in stats:
            v = np.array(per_seed[k])
            stats[k]["mean"].append(float(v.mean()))
            stats[k]["std"].append(float(v.std()))

    summary = {
        "drift_levels": DRIFT_LEVELS,
        "drift_seeds": drift_seeds,
        "auc_vs_hard": stats,
        "subspace_dim": DIM,
        "interpretation": (
            "Per-trial estimation is drift-robust across every realization (it "
            "re-derives the decision axis each trial) and dominates the other two. "
            "A fixed reference axis degrades under drift, but by a seed-dependent "
            "amount. Orthogonal-Procrustes alignment of honest subspaces helps on "
            "average but is not reliable: on some drift realizations it does not "
            "recover, and can underperform the un-aligned fixed axis. The honest "
            "takeaway is that the detector's drift-robustness comes from per-trial "
            "re-estimation, not from the alignment machinery."
        ),
    }
    (RESULTS / "drift.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {"mean": [round(x, 3) for x in stats[k]["mean"]],
                          "std": [round(x, 3) for x in stats[k]["std"]]}
                      for k in stats}, indent=2))
    _figure(DRIFT_LEVELS, stats)


def _subspace_from_rates(rates, dim):
    X = rates.reshape(-1, rates.shape[-1])
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:dim].T


def _figure(levels, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = np.array(levels)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    style = {"per_trial": ("o-", "tab:green", "per-trial estimation"),
             "fixed": ("s--", "tab:red", "fixed axis, no alignment"),
             "aligned": ("^-", "tab:blue", "fixed axis + alignment")}
    for k, (mk, col, lab) in style.items():
        m = np.array(stats[k]["mean"]); s = np.array(stats[k]["std"])
        ax.plot(levels, m, mk, color=col, label=lab)
        ax.fill_between(levels, m - s, m + s, color=col, alpha=0.15)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("representational drift level")
    ax.set_ylabel("detection AUC (manip vs honest-hard)")
    ax.set_title("Detection under drift (mean +/- SD over drift realizations)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "drift_overview.png", dpi=140)
    print("wrote", FIGS / "drift_overview.png")


if __name__ == "__main__":
    main()
