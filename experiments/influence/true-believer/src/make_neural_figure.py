"""Figure 5: real-fMRI neuroforecasting. The E+V-R composite predicts aggregate choice out-of-sample
where single ROIs do not, but the effect is experiment-dependent (weak in E1, strong in E2).

No em dashes.
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(ROOT, "results", "neuralforecast.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    d = json.load(open(RES))
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    keys = ["signature_B", "NAcc_only", "MPFC_only", "insula_only"]
    labs = ["B (E+V-R)", "NAcc", "MPFC", "insula"]
    rs = [d[k]["mean_r"] for k in keys]
    ps = [d[k]["perm_p"] for k in keys]
    colors = ["#3b6" if (r > 0 and p < 0.05) else "#b63" for r, p in zip(rs, ps)]
    ax1.bar(labs, rs, color=colors)
    ax1.axhline(0, ls=":", color="k", lw=0.8)
    ax1.set_ylabel("out-of-sample Pearson r (cross-experiment)")
    ax1.set_title("A. Composite generalizes; single ROIs do not")
    for i, (r, p) in enumerate(zip(rs, ps)):
        ax1.text(i, r + (0.01 if r >= 0 else -0.03), f"p={p:.3f}", ha="center", fontsize=8)

    st = d["per_experiment_stability"]
    chans = ["E", "V", "R", "B"]
    e1 = [st["experiment_1"][c] for c in chans]
    e2 = [st["experiment_2"][c] for c in chans]
    x = np.arange(len(chans)); w = 0.35
    ax2.bar(x - w / 2, e1, w, label="Experiment 1", color="#aac")
    ax2.bar(x + w / 2, e2, w, label="Experiment 2", color="#36b")
    ax2.axhline(0, ls=":", color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(chans)
    ax2.set_ylabel("in-sample r with choice")
    ax2.set_title("B. Effect is experiment-dependent (honest caveat)")
    ax2.legend()

    fig.suptitle("Real fMRI (Knutson neuroforecasting): the E+V-R signature predicts aggregate choice out-of-sample")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig5_neuralforecast.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
