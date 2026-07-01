"""Figure 6: model comparison for the integrated neuroforecasting composite.

A. Out-of-sample r for six models. Parsimonious composites generalize; multivariate/learned models overfit.
B. The affect composite (NAcc - AIns) is robust across anticipatory time windows.

No em dashes.
"""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(ROOT, "results", "composite_model.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    d = json.load(open(RES))
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = {"M0_NAcc": "NAcc only", "M1_EVR": "E+V-R", "M2_affect_NAcc_minus_AIns": "affect\n(NAcc-AIns)",
              "M3_ridge_full": "ridge full\n(multivariate)", "M4_ridge_ROImeans": "ridge\nROI-means",
              "M5_shrinkage_ensemble": "shrinkage\nensemble"}
    keys = list(labels)
    rs = [d["models"][k]["oos_r"] for k in keys]
    ps = [d["models"][k]["perm_p"] for k in keys]
    colors = ["#3b6" if (r > 0 and p < 0.05) else "#b63" for r, p in zip(rs, ps)]
    ax1.bar([labels[k] for k in keys], rs, color=colors)
    ax1.axhline(0, ls=":", color="k", lw=0.8)
    ax1.set_ylabel("out-of-sample r (cross-experiment)")
    ax1.set_title("A. Parsimony generalizes; complexity overfits")
    for i, (r, p) in enumerate(zip(rs, ps)):
        ax1.text(i, r + (0.008 if r >= 0 else -0.02), f"p={p:.3f}", ha="center", fontsize=7)
    ax1.tick_params(axis="x", labelsize=8)

    wr = d["window_robustness_affect"]
    ax2.bar(list(wr.keys()), list(wr.values()), color="#36b")
    ax2.axhline(0, ls=":", color="k", lw=0.8)
    ax2.set_ylabel("out-of-sample r (affect composite)")
    ax2.set_title("B. Affect composite robust across time windows")
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Improving the neuroforecasting composite: the literature-grounded parsimonious read-out wins")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig6_composite.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
