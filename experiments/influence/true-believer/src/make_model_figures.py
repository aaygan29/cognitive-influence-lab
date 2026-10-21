"""Figure 2: the four predictions of the formal belief model.

A. Hysteresis loop (belief vs persuasive force, up then down sweep).
B. Sequential in-band persuasion crosses the barrier; a single large message does not.
C. Critical force scales as alpha^{3/2} (belief robustness vs involvement).
D. Susceptibility diverges as the force approaches the fold.

No em dashes.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.belief_model import (
    pred_hysteresis, simulate, pred_robustness_scaling, susceptibility_curve, beta_crit,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # A. hysteresis
    bu, up, bd, down = pred_hysteresis()
    ax[0, 0].plot(bu, up, label="force increasing", color="#3b6")
    ax[0, 0].plot(bd, down, label="force decreasing", color="#b63")
    ax[0, 0].set_xlabel("persuasive force beta"); ax[0, 0].set_ylabel("belief b*")
    ax[0, 0].set_title("A. Hysteresis (entrenched belief, alpha>0)"); ax[0, 0].legend()

    # B. sequential vs single trajectories
    alpha, d, s, kappa, target = 0.5, 0.6, 0.9, 0.3, 1.6
    single = simulate(b_init=-1.2, alpha=alpha, d=d, s=s, kappa=kappa, messages=[(target, 3.0)])
    b = -1.2; seq_traj = [b]
    for _ in range(8):
        pos = min(target, b + 0.9 * d)
        out = simulate(b_init=b, alpha=alpha, d=d, s=s, kappa=kappa, b0=b, messages=[(pos, 3.0 / 8)])
        seq_traj.extend(out[1:]); b = float(out[-1])
    ax[0, 1].plot(seq_traj, color="#3b6", label="sequential, in-band")
    ax[0, 1].plot(single, color="#b63", label="single large step")
    ax[0, 1].axhline(target, ls=":", color="k", lw=0.8, label="target")
    ax[0, 1].set_xlabel("integration step"); ax[0, 1].set_ylabel("belief b")
    ax[0, 1].set_title("B. Foot-in-the-door beats one big push"); ax[0, 1].legend()

    # C. robustness scaling
    a, bc, p = pred_robustness_scaling(alphas=(0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8))
    ax[1, 0].plot(a, bc, "o-", color="#36b")
    ax[1, 0].set_xlabel("involvement alpha"); ax[1, 0].set_ylabel("critical force beta_crit")
    ax[1, 0].set_title(f"C. Robustness scales as alpha^1.5 (fit p={p:.2f})")

    # D. susceptibility divergence
    sc = susceptibility_curve(alpha=1.0)
    ax[1, 1].plot(sc[:, 0], sc[:, 1], color="#a3b")
    ax[1, 1].axvline(beta_crit(1.0), ls=":", color="k", lw=0.8, label="fold")
    ax[1, 1].set_xlabel("persuasive force beta"); ax[1, 1].set_ylabel("susceptibility chi")
    ax[1, 1].set_title("D. Susceptibility diverges at the fold"); ax[1, 1].legend()

    fig.suptitle("Formal belief-dynamics model: four falsifiable predictions (paper/MODEL.md)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_model.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
