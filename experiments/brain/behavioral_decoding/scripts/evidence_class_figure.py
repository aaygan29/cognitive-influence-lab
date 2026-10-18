"""Results figure for the three-class split (reviewer action #1).

Reads results/evidence_class_separation/summary.json (and nothing else) and
draws one grouped horizontal bar chart: arms grouped into offer-terms / records
/ biosignal, balanced accuracy with each arm's majority baseline marked, so the
honest reading -- strong decodes are offer-terms, records are a de-leaked null,
cheap biosignals are null, only research neuroimaging decodes -- is visible at a
glance. Saves figures/evidence_class_split.{pdf,png}.

    python scripts/evidence_class_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "evidence_class_separation" / "summary.json").read_text())

C_OFFER, C_REC, C_BIO_NULL, C_BIO_REAL = "#2b6cb0", "#b7791f", "#a0aec0", "#2f855a"

# (label, balanced_acc, majority, color, control-note)
bars = []
a = S["arms"]
bars.append(("Delay-discounting (offer terms)", a["delay_discounting_real"]["fused_balacc"],
             a["delay_discounting_real"]["majority"], C_OFFER, "behavior modality only"))
bars.append(("NARPS gambles (offer terms)", a["narps_behavior_real"]["fused_balacc"],
             a["narps_behavior_real"]["majority"], C_OFFER, "behavior modality only"))
k = S["offer_terms_and_records_controls"]
bars.append(("Kiva records, with nonpayment leak", k["with_leak_prone_nonpayment"][0], k["majority_baseline"], C_REC, "leak-prone column"))
bars.append(("Kiva records, de-leaked (clean)", k["clean_tabular"][0], k["majority_baseline"], C_REC, "credit scoring"))
bars.append(("PPG pulse -> hypertension", a["biosignal_privacy_ppgbp_hypertension"]["fused_balacc"],
             a["biosignal_privacy_ppgbp_hypertension"]["majority"], C_BIO_NULL, "cheap sensor, null"))
bars.append(("PPG pulse -> diabetes", a["biosignal_privacy_ppgbp_diabetes"]["fused_balacc"],
             a["biosignal_privacy_ppgbp_diabetes"]["majority"], C_BIO_NULL, "cheap sensor, null"))
bars.append(("DEAP peripheral -> liking", a["deap_liking"]["per_modality"]["peripheral"],
             a["deap_liking"]["majority"], C_BIO_NULL, "cheap sensor, null"))
bars.append(("DEAP lab EEG -> liking", a["deap_liking"]["per_modality"]["eeg"],
             a["deap_liking"]["majority"], C_BIO_REAL, "research EEG"))
bars.append(("NARPS research fMRI -> choice", a["narps"]["per_modality"]["fmri"],
             a["narps"]["majority"], C_BIO_REAL, "research fMRI"))
bars.append(("Clinical EEG (ASZED) -> diagnosis", S["eeg_clinical"]["residualized"],
             0.78, C_BIO_REAL, "refuted (residualized)"))

groups = [("Offer-terms", 0, 2), ("Records", 2, 4), ("Biosignal", 4, 10)]

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ys = list(range(len(bars)))[::-1]
for y, (lab, ba, maj, col, note) in zip(ys, bars):
    ax.barh(y, ba, color=col, height=0.62, zorder=3)
    ax.plot([maj, maj], [y - 0.34, y + 0.34], color="k", lw=1.6, zorder=4)  # majority baseline tick
    ax.text(ba + 0.008, y, f"{ba:.3f}", va="center", fontsize=8, zorder=5)
    ax.text(0.012, y, note, va="center", fontsize=6.5, color="white", zorder=5)

ax.set_yticks(ys)
ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.axvline(0.5, color="gray", ls=":", lw=0.8, zorder=1)
ax.set_xlim(0, 1.0)
ax.set_xlabel("balanced accuracy  (black tick = majority baseline; dotted = chance 0.5)")
ax.set_title("Biosignal decoding, by evidence class (reviewer action #1)\n"
             "strong decodes are offer-terms; records null once de-leaked; cheap sensors null; "
             "only research neuroimaging decodes", fontsize=9)

# group separators + labels on the right
for name, lo, hi in groups:
    ylo = ys[hi - 1] - 0.5
    yhi = ys[lo] + 0.5
    ax.axhspan(ylo, yhi, color="0.97", zorder=0)
    ax.text(1.02, (ylo + yhi) / 2, name, rotation=90, va="center", ha="left",
            fontsize=8, fontweight="bold", clip_on=False)

handles = [mpatches.Patch(color=C_OFFER, label="offer-terms (not a biosignal)"),
           mpatches.Patch(color=C_REC, label="records (credit scoring)"),
           mpatches.Patch(color=C_BIO_NULL, label="cheap biosignal (null)"),
           mpatches.Patch(color=C_BIO_REAL, label="research neuroimaging / refuted")]
ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
ax.grid(axis="x", alpha=0.3, zorder=0)
plt.tight_layout()
fig.subplots_adjust(right=0.9)
(ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / "figures" / "evidence_class_split.pdf")
plt.savefig(ROOT / "figures" / "evidence_class_split.png", dpi=160)
plt.close()
print("wrote figures/evidence_class_split.pdf and .png")
