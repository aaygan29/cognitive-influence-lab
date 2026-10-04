"""Fig 1 (value absorption) and Fig 2 (detector operating characteristics). No API calls."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
exec(open(ROOT / "docs/theory_check.py").read().split("rng = np.random")[0])  # mean_field, tv

C1, C2, C3, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

# --- Fig 1 ---
Ts = [60, 120, 240, 480, 960, 1920]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for bval, col in ((0.25, C1), (0.5, C2)):
    sh = [mean_field(np.full(T, bval)) - mean_field(np.zeros(T)) for T in Ts]
    ax[0].plot(Ts, sh, "-o", color=col, lw=2, ms=4, label=f"constant bonus b = {bval}")
    ax[0].axhline(bval * ETA / ALPHA, color=col, lw=1, ls="--")
ax[0].set_xscale("log"); ax[0].set_xlabel("schedule duration (trials; dose grows 32x)")
ax[0].set_ylabel("installed bias vs no-bonus learner")
ax[0].set_title("a  Constant bonus saturates at (eta/alpha) b", loc="left", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
pers = [5, 10, 20, 40, 80, 160]; T = 960
sh = []
for p in pers:
    s = np.array([0.5 if (t // p) % 2 == 0 else 0.0 for t in range(T)])
    sh.append(mean_field(s) - mean_field(np.zeros(T)))
ax[1].plot(pers, sh, "-o", color=C1, lw=2, ms=4)
ax[1].set_xscale("log"); ax[1].set_xlabel("pulse half-period (trials); dose fixed at 240")
ax[1].set_ylabel("installed bias")
ax[1].set_title("b  Equal dose, faster switching installs more", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "figures/fig1_value_absorption.pdf"); fig.savefig(ROOT / "figures/fig1_value_absorption.png", dpi=200)

# --- Fig 2 ---
v = json.load(open(ROOT / "detector/validate_steer.json")); n = 300
def wil(p):
    k = p * n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, p - (c - h) / d), max(0.0, (c + h) / d - p)
labels = list(v["FPR"]) + list(v["TPR"]); vals = list(v["FPR"].values()) + list(v["TPR"].values())
cols = [C2] * len(v["FPR"]) + [C1] * len(v["TPR"])
err = np.array([wil(p) for p in vals]).T
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.bar(range(len(vals)), vals, color=cols, width=0.7, yerr=err, ecolor=MUTED, capsize=2)
ax.axhline(0.05, color=MUTED, ls="--", lw=1); ax.text(-0.4, 0.08, "alpha = 0.05", ha="left", color=MUTED, fontsize=8)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("call rate (300 seeds, 95% CI)")
ax.set_title("Directed-steering test: false calls on honest tutors (orange) vs detection of attacks (blue)", loc="left", fontsize=9)
ax.text(4, 0.12, "below the\n2^-n floor", fontsize=7.5, color=INK, ha="center")
fig.tight_layout(); fig.savefig(ROOT / "figures/fig2_detector_oc.pdf"); fig.savefig(ROOT / "figures/fig2_detector_oc.png", dpi=200)
print("ok")
