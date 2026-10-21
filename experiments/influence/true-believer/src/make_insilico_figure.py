"""Figure 7: the in-silico belief model.

A. Cross-domain generalization AUC (brain-state vs full-factor), both directions.
B. Per-factor belief effect in each corpus; the two robust factors (evidence, identity threat) marked.
C. Counterfactual optimization: altering the robust factors raises the belief score on low-belief content.

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
RES = os.path.join(ROOT, "results", "insilico_brain.json")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    d = json.load(open(RES))
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    cd = d["cross_domain"]
    dirs = list(cd.keys())
    x = np.arange(len(dirs)); w = 0.35
    ax1.bar(x - w / 2, [cd[k]["auc_full_factors"] for k in dirs], w, label="full factors", color="#aaa")
    ax1.bar(x + w / 2, [cd[k]["auc_brain_state"] for k in dirs], w, label="brain-state", color="#3b6")
    ax1.axhline(0.5, ls=":", color="k", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(dirs); ax1.set_ylim(0.45, 0.6)
    ax1.set_ylabel("cross-domain AUC"); ax1.set_title("A. Generalization to a new domain"); ax1.legend()

    rob = d["factor_variance"]["cross_corpus_robust"]
    facs = list(rob.keys())
    yy = np.arange(len(facs))
    ax2.barh(yy - 0.2, [rob[f]["cmv_coef"] for f in facs], 0.4, label="CMV", color="#36b")
    ax2.barh(yy + 0.2, [rob[f]["p4g_coef"] for f in facs], 0.4, label="P4G", color="#b63")
    ax2.axvline(0, ls=":", color="k", lw=0.8)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f + ("  *" if rob[f]["robust"] else "") for f in facs])
    ax2.set_xlabel("belief effect (logistic coef)")
    ax2.set_title("B. Robust factors (* = both domains)"); ax2.legend()

    keys = ["optimization_CMV", "optimization_P4G"]
    labs = ["CMV", "P4G"]
    before = [d[k]["mean_belief_before"] for k in keys]
    after = [d[k]["mean_belief_after"] for k in keys]
    xx = np.arange(len(keys))
    ax3.bar(xx - 0.2, before, 0.4, label="before", color="#ccc")
    ax3.bar(xx + 0.2, after, 0.4, label="after altering factors", color="#3b6")
    ax3.set_xticks(xx); ax3.set_xticklabels(labs)
    ax3.set_ylabel("mean belief score (low-belief items)")
    ax3.set_title("C. Optimization: alter content, raise belief"); ax3.legend()

    fig.suptitle("In-silico belief model: score content, generalize across domains, optimize for belief")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig7_insilico.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
