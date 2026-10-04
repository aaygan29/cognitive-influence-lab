"""Fig 4: where a positivity bias sits decides whether tampering persists or rebounds (persistence experiment)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
r = json.load(open(ROOT / "persistence/positivity_results.json"))
cols = {"0.0": "#2a78d6", "0.2": "#eb6834", "0.4": "#1baf7a", "0.6": "#eda100"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
blocks = np.arange(1, 19)
for k, c in cols.items():
    ax[0].plot(blocks, r[f"value/{k}"]["mean_traj_attack"], color=c, lw=2, label=f"asymmetry {k}")
ax[0].axhline(0, color="#52514e", lw=0.8)
ax[0].axvspan(0.5, 8.5, color="#f0efec", zorder=0); ax[0].text(4.5, 1.0, "reward bonus on", ha="center", fontsize=8, color="#52514e")
ax[0].set_ylim(-1.35, 1.1); ax[0].set_xlabel("block (40 trials)"); ax[0].set_ylabel("learner bias a (mean, 500 seeds)")
ax[0].set_title("a  Asymmetry on value updates", loc="left", fontsize=8.5)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
ks = list(cols); x = np.arange(len(ks)); w = 0.38
for j, (where, col, lab) in enumerate((("value", "#2a78d6", "on value update (fitted in humans)"),
                                       ("policy", "#eb6834", "on policy update"))):
    d = np.array([r[f"{where}/{k}"]["paired_diff_mean"] for k in ks]); ci = np.array([r[f"{where}/{k}"]["paired_diff_ci95"] for k in ks])
    ax[1].bar(x + (j - 0.5) * w, d, w, color=col, label=lab, yerr=[d - ci[:, 0], ci[:, 1] - d], ecolor="#52514e", capsize=2)
ax[1].axhline(0, color="#52514e", lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels(ks); ax[1].set_xlabel("positivity asymmetry")
ax[1].set_ylabel("bias after washout,\nattack minus matched control")
ax[1].set_title("b  Placement flips the sign of persistence", loc="left", fontsize=8.5)
ax[1].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(ROOT / "figures/fig4_positivity.pdf"); fig.savefig(ROOT / "figures/fig4_positivity.png", dpi=200)
