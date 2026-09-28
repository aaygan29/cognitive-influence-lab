"""
Publication-quality figures for digital brain paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_ROOT = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "digital": "#2196F3",
    "real": "#4CAF50",
    "other": "#9E9E9E",
    "noise_ceiling": "#FF9800",
    "chance": "#F44336",
}

ROI_ORDER = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",
    "EBA", "FBA-1", "FBA-2",
    "OFA", "FFA-1", "FFA-2",
    "OPA", "PPA", "RSC",
    "OWFA", "VWFA-1", "VWFA-2",
    "early", "midventral", "midlateral", "midparietal",
]


def plot_encoding_accuracy(
    results_per_roi: Dict[str, Dict],
    noise_ceilings: Optional[Dict[str, float]] = None,
    subj_id: Optional[int] = None,
    save: bool = True,
) -> plt.Figure:
    """Bar plot: encoding accuracy per ROI, with noise ceiling overlay."""
    rois = [r for r in ROI_ORDER if r in results_per_roi]
    if not rois:
        rois = sorted(results_per_roi.keys())

    median_rs = [results_per_roi[r]["median_r"] for r in rois]
    r_stds = [results_per_roi[r].get("r_std", 0) for r in rois]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(rois))
    bars = ax.bar(x, median_rs, color=PALETTE["digital"], alpha=0.85, width=0.6, label="Digital brain")
    ax.errorbar(x, median_rs, yerr=r_stds, fmt="none", color="black", capsize=3, linewidth=1)

    if noise_ceilings:
        nc_vals = [noise_ceilings.get(r, np.nan) for r in rois]
        ax.plot(x, nc_vals, "o--", color=PALETTE["noise_ceiling"], label="Noise ceiling", linewidth=1.5, markersize=5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Median Pearson r")
    title = f"Encoding Accuracy{'— Subject ' + str(subj_id) if subj_id else ''}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=-0.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fname = f"encoding_accuracy{'_subj' + str(subj_id) if subj_id else ''}.pdf"
        fig.savefig(FIGURES_DIR / fname, bbox_inches="tight", dpi=150)
    return fig


def plot_rsa_matrix(
    rsa_matrix: np.ndarray,
    subj_ids: List[int],
    roi: str,
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of N×N RSA matrix.
    Diagonal = digital brain matches its own subject (should be darkest column per row).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f"S{sid:02d}" for sid in subj_ids]

    im = ax.imshow(rsa_matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.8, aspect="auto")
    plt.colorbar(im, ax=ax, label="RSA (Spearman ρ)")

    ax.set_xticks(range(len(subj_ids)))
    ax.set_yticks(range(len(subj_ids)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Real brain (subject)")
    ax.set_ylabel("Digital brain (subject)")
    ax.set_title(f"Cross-Subject RSA Matrix\nROI: {roi}", fontweight="bold")

    # Annotate diagonal
    for i in range(len(subj_ids)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="blue", lw=2))
    ax.text(len(subj_ids) - 0.5, -0.7, "← diagonal = identity",
            ha="right", fontsize=8, color="blue")

    for i in range(len(subj_ids)):
        for j in range(len(subj_ids)):
            ax.text(j, i, f"{rsa_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"rsa_matrix_{roi}.pdf", bbox_inches="tight", dpi=150)
    return fig


def plot_fingerprinting_summary(
    fingerprint_results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """Bar chart of fingerprinting accuracy per ROI vs chance."""
    rois = sorted(fingerprint_results.keys())
    accs = [fingerprint_results[r].get("accuracy", 0) for r in rois]
    chance = fingerprint_results[rois[0]].get("chance", 0.5) if rois else 0.5
    ps = [fingerprint_results[r].get("permutation_p", 1.0) for r in rois]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rois))
    bars = ax.bar(x, accs, color=[
        PALETTE["digital"] if p < 0.05 else PALETTE["other"] for p in ps
    ], alpha=0.85, width=0.6)

    ax.axhline(chance, color=PALETTE["chance"], linestyle="--", linewidth=1.5, label=f"Chance ({chance:.2f})")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)

    for xi, (acc, p) in enumerate(zip(accs, ps)):
        if p < 0.001:
            ax.text(xi, acc + 0.02, "***", ha="center", fontsize=9)
        elif p < 0.01:
            ax.text(xi, acc + 0.02, "**", ha="center", fontsize=9)
        elif p < 0.05:
            ax.text(xi, acc + 0.02, "*", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Identification Accuracy")
    ax.set_title("Subject Fingerprinting from Digital Brain\n(blue = p<0.05)", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "fingerprinting_accuracy.pdf", bbox_inches="tight", dpi=150)
    return fig


def plot_rsa_self_vs_other(
    geometry_results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """Per-ROI bar: RSA(digital, own subject) vs RSA(digital, other subjects)."""
    rois = [r for r in ROI_ORDER if r in geometry_results]
    if not rois:
        rois = sorted(geometry_results.keys())

    self_scores = [geometry_results[r].get("rsa_self", 0) for r in rois]
    other_scores = [geometry_results[r].get("rsa_other_mean", 0) for r in rois]
    other_stds = [geometry_results[r].get("rsa_other_std", 0) for r in rois]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(rois))
    w = 0.35

    ax.bar(x - w/2, self_scores, w, color=PALETTE["digital"], alpha=0.85, label="Own subject")
    ax.bar(x + w/2, other_scores, w, color=PALETTE["other"], alpha=0.85,
           yerr=other_stds, capsize=3, label="Other subjects (mean)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RSA (Spearman ρ)")
    ax.set_title("Representational Geometry: Digital Brain vs Real Brain\n(own vs other subjects)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "rsa_self_vs_other.pdf", bbox_inches="tight", dpi=150)
    return fig


def plot_counterfactual_summary(
    cf_results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """Violin plot of counterfactual consistency r across subject pairs, per ROI."""
    rois = sorted(cf_results.keys())
    data = {r: [p["r"] for p in cf_results[r].get("pairs", [])] for r in rois}
    data = {r: v for r, v in data.items() if v}

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    positions = list(range(len(data)))
    ax.violinplot([data[r] for r in sorted(data)], positions=positions, showmedians=True)
    ax.axhline(0, color=PALETTE["chance"], linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted(data.keys()), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Counterfactual r")
    ax.set_title("Counterfactual Consistency\n(correlation of subject-difference vectors)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "counterfactual_consistency.pdf", bbox_inches="tight", dpi=150)
    return fig


def summary_table(all_results: Dict, save: bool = True) -> str:
    """Generate a LaTeX-ready summary table of all evaluation metrics."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Digital Brain Validation Summary}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"ROI & Encoding r & NC ratio & RSA (self) & Fingerprint Acc \\")
    lines.append(r"\hline")

    rois = set()
    for level_results in all_results.values():
        if isinstance(level_results, dict):
            rois.update(level_results.keys())

    for roi in sorted(rois):
        enc_r = all_results.get("encoding", {}).get(roi, {}).get("median_r", np.nan)
        nc_ratio = all_results.get("encoding", {}).get(roi, {}).get("nc_ratio", np.nan)
        rsa_self = all_results.get("geometry", {}).get(roi, {}).get("rsa_self", np.nan)
        fp_acc = all_results.get("fingerprinting", {}).get(roi, {}).get("accuracy", np.nan)

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "--"

        lines.append(f"{roi} & {fmt(enc_r)} & {fmt(nc_ratio)} & {fmt(rsa_self)} & {fmt(fp_acc)} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    table = "\n".join(lines)

    if save:
        path = RESULTS_ROOT / "summary_table.tex"
        with open(path, "w") as f:
            f.write(table)
        print(f"Table saved: {path}")

    return table
