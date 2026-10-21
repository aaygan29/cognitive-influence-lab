"""Generate the figure from results/synthetic_validation.json.

Figure 1: the positive-and-negative-control panel. Content-only AUC vs content+B AUC across the three
conditions, with the delta-AUC and its CI. Shows the pipeline detects the neural signal when present and
reports null/chance when absent or scrambled.

No em dashes.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results", "synthetic_validation.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    with open(RESULTS) as f:
        d = json.load(f)
    os.makedirs(FIGDIR, exist_ok=True)

    conds = ["positive_control", "content_only_world", "scrambled_control"]
    labels = ["signal present", "content-only", "scrambled B"]
    auc_c = [d[c]["H_incremental"]["auc_content"] for c in conds]
    auc_cb = [d[c]["H_incremental"]["auc_content_plus_B"] for c in conds]
    dauc = [d[c]["H_incremental"]["finding"]["value"] for c in conds]
    dlo = [d[c]["H_incremental"]["finding"]["ci95"][0] for c in conds]
    dhi = [d[c]["H_incremental"]["finding"]["ci95"][1] for c in conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = range(len(conds))
    w = 0.35
    ax1.bar([i - w / 2 for i in x], auc_c, width=w, label="content-only", color="#888")
    ax1.bar([i + w / 2 for i in x], auc_cb, width=w, label="content + B(s)", color="#3b6")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_ylabel("held-out AUC (behavior)")
    ax1.set_ylim(0.5, 0.85)
    ax1.axhline(0.5, ls=":", color="k", lw=0.8)
    ax1.legend(); ax1.set_title("A. Predicting measured belief outcome")

    err_lo = [dauc[i] - dlo[i] for i in range(len(conds))]
    err_hi = [dhi[i] - dauc[i] for i in range(len(conds))]
    ax2.errorbar(list(x), dauc, yerr=[err_lo, err_hi], fmt="o", color="#3b6", capsize=4)
    ax2.axhline(0.0, ls=":", color="k", lw=0.8)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_ylabel("incremental AUC of B(s)  (95% CI)")
    ax2.set_title("B. Neural signature increment over content")

    fig.suptitle("Positive-and-negative-control validation of the belief-risk instrument (synthetic stand-in)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_controls.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
