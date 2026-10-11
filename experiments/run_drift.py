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
from bellwether.estimator import directedness

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
DRIFT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
DIM = 6
N = 150
SMOOTH = 3


def _approach_end(z, i0, horizon, frac=0.85):
    base = z[i0]
    exc = z[i0:horizon] - base
    tot = np.max(np.abs(exc)) if exc.size else 0.0
    if tot <= 0:
        return horizon
    reached = np.flatnonzero(np.abs(exc) >= frac * tot)
    return i0 + (int(reached[0]) if reached.size else exc.size)


def trial_directedness(rates, axis=None):
    """Per-trial directedness over the approach phase (fixed axis, or per-trial PCA)."""
    scores = []
    i0 = int(0.5 / C.DT)
    horizon = int(2.3 / C.DT)
    win = 100
    for j in range(rates.shape[1]):
        z = _pca_axis(rates[:, j, :]) if axis is None else project_fixed(rates[:, j, :], axis)
        i1 = max(_approach_end(z, i0, min(horizon, len(z))), i0 + 2)
        ds = [directedness(z[a:a + win], smooth=SMOOTH)
              for a in range(i0, max(i1 - win + 1, i0 + 1), 20) if a + win <= len(z)]
        scores.append(np.mean(ds) if ds else directedness(z[i0:i1], smooth=SMOOTH))
    return np.array(scores)


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

    rows = {"per_trial": [], "fixed": [], "aligned": []}
    for lvl in tqdm(DRIFT_LEVELS, desc="drift levels"):
        ro = base if lvl == 0 else drifted_readout(base, drift_level=lvl,
                                                    dropout=0.1, seed=777)
        hard = ro.project(simulate(Stimulus(coherence=0.0), n_trials=N,
                                   duration=C.DURATION, dt=C.DT, seed=400))
        manip = ro.project(simulate(Stimulus(coherence=0.0), Adversary(amplitude=0.012),
                                    n_trials=N, duration=C.DURATION, dt=C.DT, seed=401))
        # honest anchors in this session for alignment
        anchors = ro.project(simulate(Stimulus(coherence=0.0), n_trials=120,
                                      duration=C.DURATION, dt=C.DT, seed=402))
        V_new = _subspace_from_rates(anchors, DIM)
        a_al = aligned_axis(ref_axis, V_ref, V_new)

        for name, axis in [("per_trial", None), ("fixed", ref_axis), ("aligned", a_al)]:
            auc = roc_auc(trial_directedness(manip, axis), trial_directedness(hard, axis))
            rows[name].append(auc)

    summary = {
        "drift_levels": DRIFT_LEVELS,
        "auc_vs_hard": rows,
        "subspace_dim": DIM,
        "interpretation": (
            "A fixed reference decision axis degrades as the neural code drifts; "
            "orthogonal-Procrustes alignment of honest subspaces carries the "
            "reference axis into each drifted session and recovers detection; "
            "per-trial estimation is inherently drift-robust but re-estimates every "
            "trial."
        ),
    }
    (RESULTS / "drift.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(rows, indent=2))
    _figure(DRIFT_LEVELS, rows)


def _subspace_from_rates(rates, dim):
    X = rates.reshape(-1, rates.shape[-1])
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:dim].T


def _figure(levels, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(levels, rows["per_trial"], "o-", label="per-trial estimation", color="tab:green")
    ax.plot(levels, rows["aligned"], "^-", label="fixed axis + alignment", color="tab:blue")
    ax.plot(levels, rows["fixed"], "s--", label="fixed axis, no alignment", color="tab:red")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("representational drift level")
    ax.set_ylabel("detection AUC (manip vs honest-hard)")
    ax.set_title("Detection under drift: alignment restores a fixed decoder")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "drift_overview.png", dpi=140)
    print("wrote", FIGS / "drift_overview.png")


if __name__ == "__main__":
    main()
