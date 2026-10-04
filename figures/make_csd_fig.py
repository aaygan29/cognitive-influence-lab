"""Fig: Level-1 belief tipping with a critical-slowing-down early warning (Claim 3)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "level1_belief/belief_csd.json"))
ex = d["exemplar"]; op = d["summary"]["operating_points"]
C1, C2, MUTED, INK = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
# panel a: exemplar belief x(t) with warning and crossing marked
x = np.array(ex["x"]); t = np.arange(len(x)) * 0.01
ax[0].plot(t, x, color=INK, lw=0.6)
ax[0].axvline(ex["warn_t"] * 0.01, color=C1, lw=1.5, label="early warning")
ax[0].axvline(ex["cross_t"] * 0.01, color=C2, lw=1.5, label="belief flips")
ax[0].annotate("", xy=(ex["cross_t"] * 0.01, 1.3), xytext=(ex["warn_t"] * 0.01, 1.3),
               arrowprops=dict(arrowstyle="<->", color=MUTED))
ax[0].text((ex["warn_t"] + ex["cross_t"]) / 2 * 0.01, 1.4, "lead time", ha="center", fontsize=8, color=MUTED)
ax[0].set_xlabel("time"); ax[0].set_ylabel("belief x"); ax[0].set_ylim(-1.7, 1.7)
ax[0].set_title("a  Critical slowing down warns before a belief flips", loc="left", fontsize=8.5)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
# panel b: operating points (sensitivity vs false-alarm)
fpr = [o["false_alarm_rate"] for o in op]; sens = [o["sensitivity"] for o in op]
ax[1].plot(fpr, sens, "-o", color=C1, lw=2, ms=5)
for o in op:
    ax[1].annotate(f"lead {o['median_lead_time']:.0f}", (o["false_alarm_rate"], o["sensitivity"]),
                   textcoords="offset points", xytext=(6, -2), fontsize=7, color=MUTED)
ax[1].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
ax[1].set_xlabel("false-alarm rate (stationary belief)"); ax[1].set_ylabel("sensitivity (warned before flip)")
ax[1].set_xlim(-0.02, 0.6); ax[1].set_ylim(0, 0.7)
ax[1].set_title("b  A partial warning, honest about its limits", loc="left", fontsize=8.5)
fig.tight_layout(); fig.savefig(ROOT / "figures/fig_csd.pdf"); fig.savefig(ROOT / "figures/fig_csd.png", dpi=200)
print("ok")
