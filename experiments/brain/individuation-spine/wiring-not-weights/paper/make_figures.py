"""
Generate paper figures from the experiment results JSONs. Deterministic, no re-running experiments.
Usage: python3 make_figures.py   (writes figures/*.png and *.pdf)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = dict(full="#1f77b4", wire="#d62728", deg="#2ca02c", null="#7f7f7f", acc="#1f77b4")


def load(name):
    return json.load(open(os.path.join(RES, name)))


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"))
    plt.close(fig)
    print("wrote", stem)


def fig1_sufficiency_curve():
    d = load("exp03a_curve_results.json")
    ch = d["chance"]; full = d["full"]
    a = np.array([p["alpha"] for p in d["alpha_curve"]])
    am = np.array([p["mean"] for p in d["alpha_curve"]])
    alo = np.array([p["ci95"][0] for p in d["alpha_curve"]])
    ahi = np.array([p["ci95"][1] for p in d["alpha_curve"]])
    b = np.array([p["beta"] for p in d["beta_null_curve"]])
    bn = np.array([p["mean"] for p in d["beta_null_curve"]])
    bl = np.array([p["mean"] for p in d["beta_lowvar_curve"]])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].fill_between(a, alo, ahi, color=C["acc"], alpha=0.2)
    ax[0].plot(a, am, "-o", color=C["acc"], lw=2, ms=4, label="identity accuracy")
    ax[0].axhline(ch, ls=":", color=C["null"], label=f"chance ({ch:.3f})")
    ax[0].axhline(full, ls="--", color="#555", alpha=0.6, label=f"FULL ({full:.2f})")
    if d.get("alpha_star") is not None:
        ax[0].axvline(d["alpha_star"], ls="-.", color="#aa6", alpha=0.7,
                      label=f"$\\alpha^*$={d['alpha_star']}")
    ax[0].set_xlabel("fraction of individuating weight-dimensions retained ($\\alpha$)")
    ax[0].set_ylabel("identification accuracy")
    ax[0].set_title("A. Identity-sufficiency curve")
    ax[0].legend(fontsize=8, loc="upper left")

    ax[1].plot(b, bn, "-o", color=C["deg"], lw=2, ms=4,
               label="exact degeneracy (null-space)")
    ax[1].plot(b, bl, "-s", color=C["wire"], lw=2, ms=4,
               label="approx degeneracy (low-var)")
    ax[1].axhline(ch, ls=":", color=C["null"])
    ax[1].set_xlabel("weight-perturbation magnitude ($\\beta$)")
    ax[1].set_ylabel("identification accuracy")
    ax[1].set_title("B. Degeneracy controls")
    ax[1].legend(fontsize=8)
    fig.suptitle("Identity tracks the functional weight-dimensions; flat to exact-degeneracy weight changes",
                 fontsize=10, y=1.02)
    save(fig, "fig1_sufficiency_curve")


def fig2_nonlinear_degeneracy():
    d = load("exp03b_nonlinear_results.json")
    ch = d["chance"]
    arms = ["FULL", "PERM_SCALE", "FUNC_PERTURB", "FUNC_RAND", "WIRING"]
    labels = ["FULL", "PERM_SCALE\n(degeneracy)", "FUNC_PERTURB\n(matched mag.)",
              "FUNC_RAND", "WIRING"]
    means = [d[a]["mean"] for a in arms]
    lo = [d[a]["ci95"][0] for a in arms]; hi = [d[a]["ci95"][1] for a in arms]
    err = [np.array(means) - np.array(lo), np.array(hi) - np.array(means)]
    cols = [C["full"], C["deg"], C["wire"], C["null"], C["null"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(len(arms)), means, yerr=err, color=cols, capsize=3, alpha=0.9)
    ax.axhline(ch, ls=":", color="#444", label=f"chance ({ch:.3f})")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("identification accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("Nonlinear compensatory degeneracy: weight change vs identity")
    wc_ps = d["wchange_PERM_SCALE"]; wc_fp = d["wchange_FUNC_PERTURB"]
    ax.annotate(f"||$\\Delta$W||/||W||={wc_ps:.2f}\nidentity intact", (1, means[1]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color=C["deg"])
    ax.annotate(f"||$\\Delta$W||/||W||={wc_fp:.2f}\nidentity lost", (2, means[2]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color=C["wire"])
    ax.legend(fontsize=8)
    save(fig, "fig2_nonlinear_degeneracy")


def fig3_abide():
    d = load("exp05_abide_results.json")
    ch = d["chance"]; arms = ["FULL_weighted", "WIRING_binarized", "NODE_STRENGTH"]
    labels = ["FULL\n(weighted)", "WIRING\n(binarized)", "NODE\nSTRENGTH"]
    means = [d["arms"][a]["id_acc"] for a in arms]
    cols = [C["full"], C["wire"], C["null"]]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(range(len(arms)), means, color=cols, alpha=0.9)
    ax.axhline(ch, ls=":", color="#444", label=f"chance ({ch:.3f})")
    for i, m in enumerate(means):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels)
    ax.set_ylabel("identification accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title(f"ABIDE real connectomes (N={d['n_subjects']}, CC200):\n"
                 f"binarized topology fingerprints ≈ weighted (identification axis)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig3_abide_identification")


if __name__ == "__main__":
    fig1_sufficiency_curve()
    fig2_nonlinear_degeneracy()
    fig3_abide()
    print("figures ->", FIG)
