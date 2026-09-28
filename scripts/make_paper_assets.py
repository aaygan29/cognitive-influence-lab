#!/usr/bin/env python3
"""Generate paper figures + LaTeX table from all_results_fast.json (N=8 NSD)."""
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "algonauts2023"
SUB = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    (PROJ.parents[3] / "submissions/neuroscience/digital-brain/MM4SciReal_ACCV_2026")
d = json.load(open(RES / "all_results_fast.json"))
subs = d["subjects"]; enc = d["encoding"]; geo = d["geometry"]
fp = d["fingerprinting"]; cf = d["counterfactual"]
rois = sorted(fp.keys())
FIGD = RES / "figures_n8"; FIGD.mkdir(exist_ok=True)

# per-subject median encoding r + noise ceiling ratio
med = {s: float(np.nanmedian([enc[s][r]["r"] for r in rois])) for s in subs}
plt.figure(figsize=(6, 3.2))
plt.bar(range(len(subs)), [med[s] for s in subs], color="#3b6ea5")
plt.xticks(range(len(subs)), [f"S{i+1}" for i in range(len(subs))])
plt.ylabel("median encoding $r$"); plt.title("Encoding accuracy by subject (N=8)")
plt.tight_layout(); plt.savefig(FIGD / "encoding_by_subject.pdf"); plt.close()

# fingerprinting accuracy per ROI
acc = [fp[r]["accuracy"]*100 for r in rois]
plt.figure(figsize=(8, 3.2))
plt.bar(range(len(rois)), acc, color="#c0504d")
plt.axhline(d["encoding"] and 100/len(subs), ls="--", c="k", lw=1, label="chance")
plt.xticks(range(len(rois)), rois, rotation=90, fontsize=6)
plt.ylabel("identification %"); plt.title("Subject fingerprinting by ROI (N=8)")
plt.legend(); plt.tight_layout(); plt.savefig(FIGD / "fingerprinting_n8.pdf"); plt.close()

# RSA self-advantage per ROI (subject mean)
adv = [float(np.mean([geo[s][r]["advantage"] for s in subs])) for r in rois]
plt.figure(figsize=(8, 3.2))
plt.bar(range(len(rois)), adv, color="#4f8a5b")
plt.axhline(0, c="k", lw=0.8)
plt.xticks(range(len(rois)), rois, rotation=90, fontsize=6)
plt.ylabel("RSA self-advantage $\\Delta$"); plt.title("Geometry self-advantage by ROI (subject mean, N=8)")
plt.tight_layout(); plt.savefig(FIGD / "rsa_self_adv_n8.pdf"); plt.close()

# counterfactual per ROI
cfr = [cf[r]["mean_r"] for r in rois]
plt.figure(figsize=(8, 3.2))
plt.bar(range(len(rois)), cfr, color="#8064a2")
plt.axhline(0, c="k", lw=0.8)
plt.xticks(range(len(rois)), rois, rotation=90, fontsize=6)
plt.ylabel("counterfactual $r$"); plt.title("Between-subject geometry generalization (N=8)")
plt.tight_layout(); plt.savefig(FIGD / "counterfactual_n8.pdf"); plt.close()

# LaTeX per-ROI table
BS = "\\"; EOL = BS+BS; US = BS+"_"; PC = BS+"%"
rows = []
for r in rois:
    a = float(np.mean([geo[s][r]["advantage"] for s in subs]))
    ac = fp[r]["accuracy"]*100; p = fp[r]["p"]; c = cf[r]["mean_r"]
    star = "$^{*}$" if p < 0.05 else ""
    rows.append(f"{r.replace('_', US)} & {a:+.3f} & {ac:.0f}{PC} & {p:.3f}{star} & {c:.3f} {EOL}")
(SUB / "paper" / "roi_table_n8.tex").write_text("\n".join(rows) + "\n")

# summary numbers for the abstract/text
summary = {
    "n_subjects": len(subs), "n_shared": d["n_shared"], "n_test": d["n_test"],
    "enc_med_min": round(min(med.values()), 3), "enc_med_max": round(max(med.values()), 3),
    "enc_all_positive": all(v > 0 for v in med.values()),
    "adv_pos_rois": int(sum(a > 0 for a in adv)), "n_rois": len(rois),
    "adv_min": round(min(adv), 3), "adv_max": round(max(adv), 3),
    "fp_sig_rois": int(sum(fp[r]["p"] < 0.05 for r in rois)),
    "fp_100_rois": int(sum(fp[r]["accuracy"] == 1.0 for r in rois)),
    "cf_min": round(min(cfr), 3), "cf_max": round(max(cfr), 3),
    "cf_all_positive": all(v > 0 for v in cfr),
    "per_subject_median": {f"S{i+1}": round(med[s], 3) for i, s in enumerate(subs)},
}
json.dump(summary, open(RES / "summary_n8.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
# copy figures into the submission folder
import shutil
for f in FIGD.glob("*.pdf"):
    shutil.copy(f, SUB / "figures" / f.name)
    shutil.copy(f, SUB / "paper" / f.name)
print("figures + table + summary written")
