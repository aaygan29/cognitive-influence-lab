"""Figure 3: the real-data result on ChangeMyView (Tan et al. 2016).

Within-pair win rates for the signature and its channels, against the permutation null. Successful
challengers carry higher engagement E; affect V is null; the resistance text-proxy runs the wrong way, so
the composite B is dominated by E. This is the honest mechanism decomposition on real belief-change data.

No em dashes.
"""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(ROOT, "results", "realdata_cmv.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    d = json.load(open(RES))
    os.makedirs(FIGDIR, exist_ok=True)
    ch = d["within_pair_channels"]
    null = d["permutation_null"]["null_win_rate_mean"]
    names = ["E", "V", "R_neg", "B"]
    labels = ["E (engagement)", "V (affect)", "-R (low resistance)", "B (composite)"]
    wins = [ch[n]["win_rate"] for n in names]
    ps = [ch[n]["wilcoxon_p"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b6" if w > 0.5 else "#b63" for w in wins]
    bars = ax.bar(labels, wins, color=colors)
    ax.axhline(null, ls="--", color="k", lw=1, label=f"permutation null ({null:.3f})")
    ax.axhline(0.5, ls=":", color="#999", lw=0.8)
    ax.set_ylabel("within-pair win rate (successful > unsuccessful)")
    ax.set_ylim(0.2, 0.65)
    for b, p in zip(bars, ps):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"p={p:.1e}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Real belief change (ChangeMyView, Tan 2016): engagement channel predicts winning arguments\n"
                 f"n={d['within_pair']['n_pairs']} matched pairs, topic held fixed")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_realdata.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
