"""Regenerate all paper assets: vector figures (PDF + PNG) and LaTeX tables.

Design rules (NeurIPS / journal): vector output for LaTeX \\includegraphics, serif type to match
a LaTeX body, colorblind-safe palette, every axis labeled, chance/baseline line where relevant,
no decorative titles inside result figures (captions live in the paper). Tables are emitted as
booktabs LaTeX so they drop straight into the manuscript. Result figures render only when their
results JSON exists. Run: python figures/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # editable/embedded text, journal-safe
    "ps.fonttype": 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "paper", "tables")

# Okabe-Ito colorblind-safe
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"))
    plt.close(fig)
    return os.path.join(HERE, f"{stem}.pdf")


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.2, edgecolor=CB["black"], facecolor=color, alpha=0.22))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.1, color=CB["black"]))


def fig1_architecture():
    """System diagram: forward AIM-DDM estimator + honesty layer (predict or abstain)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.3))  # single-column-friendly aspect
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    _box(ax, 0.2, 3.9, 2.2, 1.1, "Event $c$\n(stakes, gain/loss,\nambiguity, threat, time)", CB["grey"])
    _box(ax, 0.2, 1.9, 2.2, 1.1, "Neural populations\nfMRI: NAcc/insula\nEEG: Cue-P3/CNV", CB["blue"])
    _box(ax, 3.0, 3.6, 2.4, 1.6, "Affective channel\n$(\\rho_g,\\lambda_l,\\omega_t)$\npopulation prior", CB["orange"])
    _box(ax, 3.0, 0.3, 2.4, 1.6, "Integrative channel\n$(\\kappa_r,\\delta_t,\\tau)$\nper-agent", CB["green"])
    _box(ax, 5.9, 2.1, 1.9, 1.4, "DDM likelihood\n$V\\!\\to$ drift,\ncaution $\\to$ boundary", CB["purple"])
    _box(ax, 8.0, 2.1, 1.8, 1.4, "Honesty layer\nproper score +\nconformal set +\nMDES gate", CB["red"])
    _arrow(ax, 2.4, 4.4, 3.0, 4.4)
    _arrow(ax, 2.4, 2.4, 3.0, 1.1)
    _arrow(ax, 2.4, 2.6, 3.0, 4.2)
    _arrow(ax, 5.4, 4.2, 6.85, 3.5)
    _arrow(ax, 5.4, 1.1, 6.85, 2.4)
    _arrow(ax, 7.8, 2.8, 8.0, 2.8)
    ax.text(8.9, 1.9, "predict or abstain", ha="center", fontsize=7.5, style="italic", color=CB["red"])
    fig.tight_layout()
    return _save(fig, "fig1_architecture")


def fig2_scale_ladder():
    """Attribution-scope ladder and the predicted identifiability dose-response (schematic, no data)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1.1, 1.0]})
    scopes = ["exec.\n+agencies", "WH+\nPentagon", "White\nHouse", "Pres.+\ncabinet",
              "Pres.+\nadvisors", "President\nalone"]
    # left: nested scope ladder
    axL.set_xlim(0, 1); axL.set_ylim(0, len(scopes)); axL.axis("off")
    for i, s in enumerate(scopes):
        y = len(scopes) - i - 1
        w = 0.15 + 0.80 * (len(scopes) - i) / len(scopes)
        axL.add_patch(FancyBboxPatch((0.5 - w / 2, y + 0.12), w, 0.72,
                      boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.0,
                      edgecolor=CB["black"], facecolor=CB["blue"], alpha=0.18))
        axL.text(0.5, y + 0.48, s, ha="center", va="center", fontsize=7)
    axL.text(0.5, len(scopes) + 0.1, "attribution scope (aggregate $\\to$ individual)",
             ha="center", fontsize=8)
    # right: predicted dose-response (schematic curve; NOT a result)
    x = list(range(len(scopes)))
    ident = [0.92, 0.85, 0.72, 0.55, 0.38, 0.22]     # predicted identifiability
    abst = [0.02, 0.05, 0.12, 0.30, 0.55, 0.78]      # predicted abstention rate
    axR.plot(x, ident, "-o", color=CB["blue"], label="predicted identifiability")
    axR.plot(x, abst, "--s", color=CB["red"], label="predicted abstention rate")
    axR.axhline(0.5, color=CB["grey"], lw=0.8, ls=":")
    axR.set_xticks(x); axR.set_xticklabels([f"S{i+1}" for i in x], fontsize=7)
    axR.set_ylim(0, 1); axR.set_xlabel("attribution scope"); axR.set_ylabel("rate")
    axR.legend(loc="center left", frameon=False)
    axR.text(0.02, 0.03, "schematic hypothesis, not a result", transform=axR.transAxes,
             fontsize=6.5, style="italic", color=CB["grey"])
    fig.tight_layout()
    return _save(fig, "fig2_scale_ladder")


# ----------------------------------------------------------------- LaTeX tables

def _write_latex(name, body):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    with open(path, "w") as fh:
        fh.write(body)
    return path


def table_claims():
    rows = [
        ("C1", "AIM-DDM decision phenotype is identifiable", "parameter recovery (synthetic)"),
        ("C2", "Conformal + proper scoring + MDES gate give coverage, calibration, abstention",
         "coverage simulation"),
        ("C3", "Affective channel forecasts held-out choice beyond behavior", "NARPS (fMRI), DEAP (EEG)"),
        ("C4", "Hierarchical pooling lowers the minimum detectable effect", "NARPS $+$ DEAP joint fit"),
        ("C5", "Predict where identifiable; abstain where confounded", "public-record decision corpus"),
    ]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Central claims and where each is evaluated.}",
             r"\label{tab:claims}", r"\begin{tabular}{llp{4.3cm}l}", r"\toprule",
             r"& Claim & Statement & Evaluation data \\", r"\midrule"]
    for cid, stmt, data in rows:
        lines.append(f"{cid} & & {stmt} & {data} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write_latex("claims.tex", "\n".join(lines))


def table_experiments():
    rows = [
        ("E1", "C1", "recovered vs.\\ true $\\theta$, 20 seeds", "recovery $r \\ge 0.6$ (affective)"),
        ("E2", "C2", "conformal coverage vs.\\ nominal; ECE", "$|\\text{cov}-\\text{nom}|\\le 0.03$"),
        ("E3", "C3", "affective vs.\\ behavioral $\\Delta$CRPS", "CI excludes 0, stimulus-grouped CV"),
        ("E4", "C4", "MDES / posterior width, joint vs.\\ single", "pooled width $<$ single"),
        ("E5", "C5", "abstention rate vs.\\ attribution scope", "gate fires as scope narrows"),
    ]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Experiments, the claim each tests, its metric, and its pass criterion.}",
             r"\label{tab:experiments}", r"\begin{tabular}{lllp{3.4cm}}", r"\toprule",
             r"Exp. & Claim & Metric & Pass criterion \\", r"\midrule"]
    for eid, cid, metric, crit in rows:
        lines.append(f"{eid} & {cid} & {metric} & {crit} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return _write_latex("experiments.tex", "\n".join(lines))


def fig3_recovery():
    """Parameter-recovery curve from results/e1_recovery.json (r vs decisions/agent)."""
    p = os.path.join(RESULTS, "e1_recovery.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    grid = res["design"]["trials_grid"]
    thr = res["design"]["kill_threshold_affective"]
    style = {"loss_aversion": (CB["orange"], "-o"), "threat": (CB["red"], "-s"),
             "risk": (CB["green"], "--^"), "discount": (CB["blue"], "--d")}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for k, (c, ls) in style.items():
        ys = [res["recovery_curve"][str(nt)][k]["mean_r"] for nt in grid]
        ax.plot(grid, ys, ls, color=c, label=k.replace("_", " "), markersize=4)
    ax.axhline(thr, color=CB["grey"], lw=0.9, ls=":")
    ax.text(grid[0], thr + 0.015, f"kill threshold r={thr}", fontsize=6.5, color=CB["grey"])
    ax.set_xscale("log"); ax.set_xticks(grid); ax.set_xticklabels([str(g) for g in grid])
    ax.set_xlabel("decisions per agent"); ax.set_ylabel("recovery $r$ (true vs.\\ recovered ratio)")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, "fig3_recovery")


def _result_figs():
    made = []
    r = fig3_recovery()
    if r:
        made.append(r)
    mapping = {
        "e2_coverage.json": "fig4_coverage",
        "e2_coverage.json": "fig4_coverage",
        "e3_transfer_narps.json": "fig5_transfer",
        "e4_pooling.json": "fig6_pooling",
        "e5_abstention.json": "fig7_abstention",
    }
    for src, stem in mapping.items():  # noqa
        p = os.path.join(RESULTS, src)
        if os.path.exists(p):
            with open(p) as fh:
                json.load(fh)
            made.append(f"(result JSON present for {src}; renderer added with the experiment -> {stem})")
    return made


if __name__ == "__main__":
    outs = [fig1_architecture(), fig2_scale_ladder(), table_claims(), table_experiments()]
    outs += _result_figs()
    for o in outs:
        print("wrote", o)
