import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})

def h(p, q):
    a = lambda x: 2*math.asin(math.sqrt(min(max(x,0),1)))
    return abs(a(p)-a(q))

# ---- Fig 1: the dissociation (install h vs override DiD per modality) ----
d = json.load(open(R/"stage4_modalities/summary.json"))["modalities"]
mods = ["regime", "ideology_fictional", "benign_science"]
labels = ["regime", "ideology", "benign"]
hs = [h(d[m]["stance_base"]["pro_principal"], d[m]["stance_loyal"]["pro_principal"]) for m in mods]
dids = [d[m]["bypass_did"] for m in mods]
cis = [d[m]["bypass_did_ci95"] for m in mods]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
a1.bar(labels, hs, color="#3b6ea5"); a1.axhline(0.8, ls="--", c="gray", lw=0.8)
a1.set_ylabel("in-domain install (Cohen's $h$)"); a1.set_title("Representation: saturated")
a1.text(0, hs[0]+0.05, "large", fontsize=7, color="gray")
err = [[did-ci[0] for did, ci in zip(dids, cis)], [ci[1]-did for did, ci in zip(dids, cis)]]
a2.axhline(0, c="k", lw=0.6)
a2.errorbar(labels, dids, yerr=err, fmt="o", color="#a53b3b", capsize=3)
a2.set_ylim(-0.3, 0.3); a2.set_ylabel("domain-matched override (DiD)"); a2.set_title("Disposition: absent")
a2.axhspan(-0.06, 0.06, color="#a53b3b", alpha=0.08)
fig.tight_layout(); fig.savefig(FIG/"fig1_dissociation.pdf"); plt.close(fig)

# ---- Fig 2: J-Lens steering specificity (stage6) ----
s6 = json.load(open(R/"stage6_jlens_steer/summary.json"))
al = [float(a) for a in s6["curves"]["jlens"].keys()]
jl = list(s6["curves"]["jlens"].values()); rn = list(s6["curves"]["random"].values())
sp = list(s6["specificity"].values())
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(al, jl, "-o", color="#3b6ea5", label="J-Lens direction")
ax.plot(al, rn, "-s", color="gray", label="random (matched norm)")
ax.plot(al, sp, "--^", color="#2a8a4a", label="specificity (J-Lens $-$ random)")
ax.axhline(s6["base"], ls=":", c="k", lw=0.8, label="base")
ax.set_xlabel("steering strength $\\alpha$"); ax.set_ylabel("in-domain differential")
ax.set_title("J-Lens steers specifically"); ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(FIG/"fig2_jlens_steer.pdf"); plt.close(fig)

# ---- Fig 3: exploit (stage9) if present ----
p9 = R/"stage9_exploit/summary.json"
if p9.exists():
    s9 = json.load(open(p9))
    al = [float(a) for a in s9["curves"]["jlens"].keys()]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, c, mk in (("jlens", "#3b6ea5", "o"), ("logit", "#a53b3b", "s"), ("random", "gray", "^")):
        ur = [s9["curves"][name][str(a)]["unsafe_rate"] for a in al]
        ax.plot(al, ur, "-"+mk, color=c, label=name)
    ax.axhline(s9["base"]["unsafe_rate"], ls=":", c="k", lw=0.8, label="base")
    ax.set_xlabel("steering strength $\\alpha$"); ax.set_ylabel("battery unsafe rate")
    ax.set_title("Steering the action direction"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG/"fig3_exploit.pdf"); plt.close(fig)
    print("fig3 written")
print("figures written:", [p.name for p in FIG.glob("*.pdf")])
