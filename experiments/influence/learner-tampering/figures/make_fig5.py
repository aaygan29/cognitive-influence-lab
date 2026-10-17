"""Fig 5: the installed-bias law, validated in simulation. Two simple panels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
v = json.load(open(ROOT / "docs/validate_law.json"))
law = json.load(open(ROOT / "docs/installed_bias_law.json"))
C1, C2, MUTED, INK = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

# panel a: superposition — both channels vs sum of each alone
both = [r["both"] for r in law["rows"]]; summ = [r["superposition_sum"] for r in law["rows"]]
ax[0].plot([0, 0.8], [0, 0.8], color=MUTED, lw=1, ls="--", zorder=0)
ax[0].scatter(summ, both, color=C1, s=28, zorder=2)
ax[0].set_xlabel("reward channel + curriculum channel\n(each measured alone)")
ax[0].set_ylabel("bias installed by both together")
ax[0].set_title(f"a  The two channels add (residual < {v['check1_superposition_max_residual']})", loc="left", fontsize=8.5)
ax[0].set_xlim(-0.03, 0.82); ax[0].set_ylim(-0.03, 0.82)

# panel b: reward coefficient tracks eta/alpha
rr = v["check2_reward_coeff_vs_ratio"]
x = [r["eta_over_alpha"] for r in rr]; y = [r["coeff"] for r in rr]
lo, hi = min(x) - 0.2, max(x) + 0.2
ax[1].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="coefficient = eta/alpha")
ax[1].scatter(x, y, color=C2, s=34, zorder=2, label="measured")
ax[1].set_xlabel("learning-rate ratio  eta / alpha")
ax[1].set_ylabel("measured reward-channel coefficient")
ax[1].set_title(f"b  Reward channel scales with eta/alpha", loc="left", fontsize=8.5)
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1].text(1.95, 0.55, f"bound (eta/alpha) sup|b|\nheld 0/{v['check3_bound']['n']} schedules",
           fontsize=7.5, color=INK, ha="right")
fig.tight_layout(); fig.savefig(ROOT / "figures/fig5_law.pdf"); fig.savefig(ROOT / "figures/fig5_law.png", dpi=200)
print("ok")
