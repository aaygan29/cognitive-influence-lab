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

import numpy as np
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
        ("C6", "Inverse inference is scale-dependent (aggregate identifiable, individual abstains)",
         "attribution-scale ladder S1--S6"),
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
        ("E1", "C1", "recovered vs.\\ true ratios, 20 seeds", "affective $r \\ge 0.6$ (achieved $\\sim$1--2k dec./agent)"),
        ("E2", "C2", "conformal coverage; ECE; gate", "coverage within 0.03; gate abstains when underpowered"),
        ("E3", "C3", "brain vs.\\ behavior oos $R^2$", "brain $>$ behavior on +control, null on $-$control"),
        ("E4", "C4", "unpooled vs.\\ pooled recovery $r$", "$\\Delta r$ CI $> 0$ at low $n$"),
        ("E6", "C5,C6", "identifiability + abstention vs.\\ scope", "identif.\\ falls, abstention rises S1$\\to$S6"),
        ("E7", "ext.", "recovered vs.\\ true state shift", "shift detected on +control, null on $-$control"),
    ]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Experiments, the claim each tests, its metric, and its pass criterion. E3, E6, E7 report positive/negative controls on synthetic data; real-data deployment is Section~7.}",
             r"\label{tab:experiments}", r"\begin{tabular}{llp{3.2cm}p{4.2cm}}", r"\toprule",
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


def fig9_real_narps_fmri():
    """Real neural grounding: NAcc/insula gain & loss betas + neural-vs-behavioral loss aversion."""
    roi_p = os.path.join(RESULTS, "narps_fmri_roi.json")
    beh_p = os.path.join(RESULTS, "real_narps.json")
    if not (os.path.exists(roi_p) and os.path.exists(beh_p)):
        return None
    with open(roi_p) as fh:
        roi = json.load(fh)
    with open(beh_p) as fh:
        beh = json.load(fh)["per_subject"]
    good = {s: r for s, r in roi.items() if r and "nacc_gain" in r}
    if len(good) < 4:
        return None
    keys = ["nacc_gain", "nacc_loss", "ains_gain", "ains_loss"]
    arr = {k: np.array([r[k] for r in good.values()]) for k in keys}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    means = [arr[k].mean() for k in keys]
    sems = [arr[k].std(ddof=1) / np.sqrt(len(arr[k])) for k in keys]
    labels = ["NAcc\ngain", "NAcc\nloss", "aIns\ngain", "aIns\nloss"]
    cols = [CB["blue"], CB["blue"], CB["red"], CB["red"]]
    axL.bar(range(4), means, yerr=sems, color=cols, alpha=0.8, capsize=3,
            edgecolor=CB["black"], linewidth=0.6)
    axL.axhline(0, color=CB["black"], lw=0.8)
    axL.set_xticks(range(4)); axL.set_xticklabels(labels, fontsize=7)
    axL.set_ylabel("ROI effect size (beta)"); axL.set_title("real NARPS fMRI (n=%d)" % len(good), fontsize=8)
    # scatter: neural loss aversion vs behavioral lambda
    nla, bla = [], []
    for s, r in good.items():
        if abs(r["nacc_gain"]) > 1e-6 and not beh.get(s, {}).get("abstained", True):
            nla.append(-r["nacc_loss"] / r["nacc_gain"]); bla.append(beh[s]["loss_aversion"])
    if len(nla) >= 4:
        axR.scatter(bla, nla, color=CB["purple"], s=18, edgecolor=CB["black"], linewidth=0.4)
        r = np.corrcoef(bla, nla)[0, 1]
        axR.set_title(f"neural vs behavioral $\\lambda$ (r={r:.2f})", fontsize=8)
    axR.set_xlabel("behavioral loss aversion"); axR.set_ylabel("neural loss aversion (NAcc)")
    fig.tight_layout()
    return _save(fig, "fig9_real_narps_fmri")


def fig8_real_narps():
    """Real NARPS results: loss-aversion distribution by group + held-out prediction summary."""
    p = os.path.join(RESULTS, "real_narps.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    lams = {"equalRange": [], "equalIndifference": []}
    for s, v in res["per_subject"].items():
        if not v.get("abstained") and v.get("group") in lams:
            lams[v["group"]].append(v["loss_aversion"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1.3, 1.0]})
    bins = np.linspace(0, 4, 25)
    for g, c in [("equalIndifference", CB["blue"]), ("equalRange", CB["orange"])]:
        v = np.array(lams[g])
        axL.hist(v, bins=bins, alpha=0.55, color=c, label=f"{g} (md {np.median(v):.2f})", edgecolor="none")
        axL.axvline(np.median(v), color=c, lw=1.2, ls="--")
    axL.axvline(1.0, color=CB["grey"], lw=0.9, ls=":")
    axL.text(1.02, axL.get_ylim()[1] * 0.9, "$\\lambda=1$", fontsize=6.5, color=CB["grey"])
    axL.set_xlabel("loss aversion $\\lambda = |b_{loss}|/|b_{gain}|$"); axL.set_ylabel("subjects")
    axL.legend(frameon=False, fontsize=6.5, loc="upper right")
    axL.set_title("real NARPS choices (n=%d)" % res["n_subjects_used"], fontsize=8)
    # right panel: held-out prediction + coverage summary as a small bar
    pr = res["RN1_prediction"]; h = res["RN2_honesty"]
    metrics = ["oos AUC", "coverage", "1-ECE", "brier vs\nbase"]
    vals = [pr["mean_oos_auc"], h["median_conformal_coverage"], 1 - h["median_ece"],
            1 - pr["median_oos_brier"] / pr["median_baserate_brier"]]
    axR.bar(range(len(metrics)), vals, color=[CB["green"], CB["blue"], CB["purple"], CB["orange"]],
            edgecolor=CB["black"], linewidth=0.6)
    axR.set_xticks(range(len(metrics))); axR.set_xticklabels(metrics, fontsize=6.5)
    axR.set_ylim(0, 1.05); axR.axhline(0.5, color=CB["grey"], lw=0.8, ls=":")
    axR.set_title("held-out prediction + honesty", fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig8_real_narps")


def fig7_scale_ladder_result():
    """Identifiability + abstention vs attribution scope, from e6_scale_ladder.json (C5, C6)."""
    p = os.path.join(RESULTS, "e6_scale_ladder.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    labels = res["design"]["scopes"]
    short = [l.split("_", 1)[0] for l in labels]
    idc = res["identifiability_curve"]
    abc = res["abstention_curve"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(x, idc, "-o", color=CB["blue"], label="identifiability $r$ (all entities)")
    ax.plot(x, abc, "--s", color=CB["red"], label="abstention rate")
    ax.axhline(0.5, color=CB["grey"], lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_xlabel("attribution scope (aggregate S1 $\\to$ individual S6)")
    ax.set_ylabel("rate"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    fig.tight_layout()
    return _save(fig, "fig7_scale_ladder_result")


def fig6_pooling():
    """Unpooled vs pooled recovery r at the smallest decisions/agent, from e4_pooling.json (C4)."""
    p = os.path.join(RESULTS, "e4_pooling.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    sm = str(res["design"]["trials_grid"][0])
    params = ["loss_aversion", "threat", "risk", "discount"]
    unp = [res["curve"][sm][k]["unpooled_r"] for k in params]
    poo = [res["curve"][sm][k]["pooled_r"] for k in params]
    x = np.arange(len(params)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(x - w / 2, unp, w, label="unpooled", color=CB["grey"], edgecolor=CB["black"], linewidth=0.6)
    ax.bar(x + w / 2, poo, w, label="pooled", color=CB["green"], edgecolor=CB["black"], linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("_", "\n") for k in params], fontsize=7)
    ax.set_ylabel("recovery $r$"); ax.set_ylim(0, 1)
    ax.set_title(f"{sm} decisions/agent", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig6_pooling")


def fig5_transfer():
    """Affective-vs-behavioral forecast arms, positive vs negative control, from e3_transfer.json."""
    p = os.path.join(RESULTS, "e3_transfer.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        res = json.load(fh)
    arms = ["content_only", "behavior_only", "brain_only"]
    labels = ["content", "behavior", "brain"]
    colors = [CB["grey"], CB["orange"], CB["blue"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    for ax, cond, title in [(axes[0], "positive_control", "market tracks affect (positive control)"),
                            (axes[1], "negative_control", "market is noise (negative control)")]:
        c = res[cond]
        means = [c["arms_r2"][a]["mean"] for a in arms]
        los = [c["arms_r2"][a]["mean"] - c["arms_r2"][a]["ci95"][0] for a in arms]
        his = [c["arms_r2"][a]["ci95"][1] - c["arms_r2"][a]["mean"] for a in arms]
        ax.bar(range(len(arms)), means, yerr=[los, his], color=colors, alpha=0.85,
               capsize=3, edgecolor=CB["black"], linewidth=0.6)
        ax.axhline(0, color=CB["black"], lw=0.8)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("out-of-sample $R^2$ (held-out stimuli)")
    fig.tight_layout()
    return _save(fig, "fig5_transfer")


def _result_figs():
    made = []
    for fn in (fig3_recovery, fig6_pooling, fig5_transfer, fig7_scale_ladder_result,
               fig8_real_narps, fig9_real_narps_fmri):
        r = fn()
        if r:
            made.append(r)
    mapping = {}  # renderers added as further experiments land
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
