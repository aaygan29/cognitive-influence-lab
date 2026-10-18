"""Degradation curve for the agent-contamination sensitivity (reviewer #4).
Reads results/agent_contamination_sensitivity/sensitivity.json only.
    python scripts/agent_contamination_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "agent_contamination_sensitivity" / "sensitivity.json").read_text())
fs = S["fractions"]
m = np.array([S["curve"][str(f) if str(f) in S["curve"] else f]["balacc_mean"] for f in fs])
sd = np.array([S["curve"][str(f) if str(f) in S["curve"] else f]["balacc_sd"] for f in fs])

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.errorbar(fs, m, yerr=sd, marker="o", ms=5, lw=1.6, color="#b7791f", capsize=3, zorder=3)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance (0.5)")
clean = S["clean_balacc"]
ax.axhline(0.5 + (clean - 0.5) / 2, color="C0", ls="--", lw=0.9, label="half of clean lift")
ax.set_xlabel("fraction of participants that are agents (f)")
ax.set_ylabel("balanced accuracy (25 seeds)")
ax.set_ylim(0.49, 0.63)
ax.set_title("Agent-contamination sensitivity, Kiva aggregate arm (reviewer #4)\n"
             "agent rows: human-like features, outcome unlinked to features", fontsize=9)
for f, y in zip(fs, m):
    ax.annotate(f"{y:.3f}", (f, y), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / "figures" / "agent_contamination_sensitivity.pdf")
plt.savefig(ROOT / "figures" / "agent_contamination_sensitivity.png", dpi=160)
plt.close()
print("wrote figures/agent_contamination_sensitivity.{pdf,png}")
