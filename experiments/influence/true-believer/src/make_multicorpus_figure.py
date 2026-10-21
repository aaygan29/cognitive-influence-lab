"""Figure 4: cross-corpus coefficients for the three channels.

Standardized logistic coefficient (controlling content) in ChangeMyView (argument delta) and
Persuasion-for-Good (charity donation), with bootstrap CIs. Engagement points the same way in both; the
pooled estimate is significant; the second corpus is individually underpowered.

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
RES = os.path.join(ROOT, "results", "multicorpus.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    d = json.load(open(RES))
    os.makedirs(FIGDIR, exist_ok=True)
    chans = ["E", "V", "R"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(chans))
    for off, key, color, lab in [(-0.15, "corpus1", "#36b", d["corpus1"]["name"]),
                                 (0.15, "corpus2", "#b63", d["corpus2"]["name"])]:
        cf = d[key]["coefs"]
        xs = [cf[c]["coef"] for c in chans]
        los = [cf[c]["coef"] - cf[c]["ci95"][0] for c in chans]
        his = [cf[c]["ci95"][1] - cf[c]["coef"] for c in chans]
        ax.errorbar(xs, ypos + off, xerr=[los, his], fmt="o", color=color, capsize=4, label=lab)
    ax.axvline(0, ls=":", color="k", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(["E (engagement)", "V (affect)", "R (resistance proxy)"])
    ax.set_xlabel("standardized logistic coefficient on persuasion success")
    ax.set_title("Cross-domain generalization: engagement points the same way in both corpora\n"
                 "(argument delta vs charity donation); pooled E significant, corpus 2 underpowered")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_multicorpus.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
