"""
Head-to-head comparison: ridge regression vs. geometry-aware encoder.

Evaluates both on the same BOLD5000 data across three metrics:
  1. Encoding r          — does it predict voxel amplitudes?
  2. RSA self-advantage  — does it preserve individual geometry?
  3. Fingerprinting acc  — can it identify subjects?

This is the direct test of whether geometry-aware training fixes the RSA paradox.
"""
import sys, json, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry_aware_encoder import GeometryAwareDigitalBrain
from scripts.run_bold5000_experiment import (
    load_dataset, DigitalBrain, load_or_train, median_r, noise_ceiling, rdm, rsa
)

PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
FEAT_DIR  = Path(__file__).parent.parent / "data" / "features"
RES_DIR   = Path(__file__).parent.parent / "results" / "comparison"
GEO_MODELS = RES_DIR / "geo_models"
RID_MODELS = RES_DIR / "ridge_models"
FIG_DIR   = RES_DIR / "figures"
for d in [RES_DIR, GEO_MODELS, RID_MODELS, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROIS = ["EarlyVis", "OPA", "PPA", "RSC", "LOC"]
SEED = 42
rng  = np.random.default_rng(SEED)


def run_comparison():
    print("="*65)
    print("ARCHITECTURE COMPARISON: Ridge vs. Geometry-Aware")
    print("="*65)

    features, all_betas, subjects, _ = load_dataset()
    rois = [r for r in ROIS if any(r in all_betas[s] for s in subjects)]

    n_total  = features.shape[0]
    n_test   = int(n_total * 0.15)
    test_idx = rng.choice(n_total, n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    rsa_idx  = test_idx[:min(150, n_test)]

    X_train = features[train_idx]
    X_test  = features[test_idx]
    X_rsa   = features[rsa_idx]

    # ── Train both model types ────────────────────────────────────────────────
    ridge_brains = {}
    geo_brains   = {}

    for subj in subjects:
        roi_betas_train = {r: b[train_idx] for r, b in all_betas[subj].items() if r in rois}

        # Ridge
        rp = RID_MODELS / f"{subj}.pkl"
        if rp.exists():
            with open(rp, "rb") as f: ridge_brains[subj] = pickle.load(f)
            print(f"  Loaded ridge: {subj}")
        else:
            brain = DigitalBrain(subj)
            brain.fit(X_train, roi_betas_train)
            with open(rp, "wb") as f: pickle.dump(brain, f)
            ridge_brains[subj] = brain

        # Geometry-aware
        gp = GEO_MODELS / f"geo_{subj}.pkl"
        if gp.exists():
            geo_brains[subj] = GeometryAwareDigitalBrain.load(gp)
            print(f"  Loaded geo:   {subj}")
        else:
            geo = GeometryAwareDigitalBrain(
                subj_id=subj,
                alpha=1.0, beta=0.8, gamma=0.4,
                hidden_dim=512, n_voxel_pcs=100,
                n_epochs=300, lr=3e-4, batch_size=64,
            )
            geo.fit(X_train, roi_betas_train, verbose=True)
            geo.save(gp)
            geo_brains[subj] = geo

    # ── Generate predictions ──────────────────────────────────────────────────
    ridge_test = {s: b.predict(X_test) for s, b in ridge_brains.items()}
    geo_test   = {s: b.predict(X_test) for s, b in geo_brains.items()}
    ridge_rsa  = {s: b.predict(X_rsa)  for s, b in ridge_brains.items()}
    geo_rsa    = {s: b.predict(X_rsa)  for s, b in geo_brains.items()}

    real_test = {s: {r: b[test_idx] for r, b in all_betas[s].items()} for s in subjects}
    real_rsa  = {s: {r: b[rsa_idx]  for r, b in all_betas[s].items()} for s in subjects}

    # ── Metric 1: Encoding r ──────────────────────────────────────────────────
    print("\nMetric 1 — Encoding accuracy (median Pearson r)")
    enc_ridge = {}; enc_geo = {}
    for subj in subjects:
        enc_ridge[subj] = {}; enc_geo[subj] = {}
        for roi in rois:
            if roi not in real_test[subj]: continue
            real = real_test[subj][roi]
            if roi in ridge_test[subj]:
                enc_ridge[subj][roi] = median_r(ridge_test[subj][roi], real)
            if roi in geo_test[subj]:
                enc_geo[subj][roi] = median_r(geo_test[subj][roi], real)
        print(f"  {subj}")
        for roi in rois:
            r_r = enc_ridge[subj].get(roi, float('nan'))
            r_g = enc_geo[subj].get(roi, float('nan'))
            delta = r_g - r_r
            print(f"    {roi:<12} ridge={r_r:.3f}  geo={r_g:.3f}  Δ={delta:+.3f}")

    # ── Metric 2: RSA self-advantage ──────────────────────────────────────────
    print("\nMetric 2 — RSA self-advantage (geometry fingerprint)")
    n_idx = np.arange(len(rsa_idx))

    def rsa_advantage(preds_rsa, real_rsa_dict, subj):
        result = {}
        others = [s for s in subjects if s != subj]
        for roi in rois:
            if roi not in preds_rsa[subj] or roi not in real_rsa_dict[subj]: continue
            rdm_pred = rdm(preds_rsa[subj][roi])
            rdm_self = rdm(real_rsa_dict[subj][roi])
            r_self = rsa(rdm_pred, rdm_self)
            r_others = []
            for o in others:
                if roi in real_rsa_dict[o]:
                    r_others.append(rsa(rdm_pred, rdm(real_rsa_dict[o][roi])))
            result[roi] = r_self - (np.mean(r_others) if r_others else 0)
        return result

    adv_ridge = {s: rsa_advantage(ridge_rsa, real_rsa, s) for s in subjects}
    adv_geo   = {s: rsa_advantage(geo_rsa,   real_rsa, s) for s in subjects}

    print(f"  {'ROI':<12} {'Ridge Δ':>10} {'Geo Δ':>10} {'Improvement':>12}")
    print("  " + "-"*46)
    for roi in rois:
        r_vals = [adv_ridge[s].get(roi, float('nan')) for s in subjects]
        g_vals = [adv_geo[s].get(roi, float('nan'))   for s in subjects]
        r_mean = float(np.nanmean(r_vals))
        g_mean = float(np.nanmean(g_vals))
        print(f"  {roi:<12} {r_mean:>+10.3f} {g_mean:>+10.3f} {g_mean-r_mean:>+12.3f}")

    # ── Metric 3: Fingerprinting ──────────────────────────────────────────────
    print("\nMetric 3 — Subject fingerprinting accuracy")
    n_s = len(subjects)

    def fingerprint(preds_test, real_test_dict):
        fp = {}
        for roi in rois:
            correct = 0
            for ts in subjects:
                if roi not in real_test_dict[ts]: continue
                real = real_test_dict[ts][roi]
                best_r, best_s = -np.inf, None
                for ps in subjects:
                    if roi not in preds_test[ps]: continue
                    pred = preds_test[ps][roi]
                    min_v = min(real.shape[1], pred.shape[1])
                    r = np.corrcoef(real[:,:min_v].ravel(), pred[:,:min_v].ravel())[0,1]
                    if r > best_r: best_r, best_s = r, ps
                if best_s == ts: correct += 1
            fp[roi] = correct / n_s
        return fp

    fp_ridge = fingerprint(ridge_test, real_test)
    fp_geo   = fingerprint(geo_test,   real_test)

    for roi in rois:
        print(f"  {roi:<12} ridge={fp_ridge.get(roi,0):.0%}  geo={fp_geo.get(roi,0):.0%}")

    # ── Summary figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Encoding r
    ax = axes[0]
    ridge_enc_mean = [np.nanmean([enc_ridge[s].get(r, float('nan')) for s in subjects]) for r in rois]
    geo_enc_mean   = [np.nanmean([enc_geo[s].get(r, float('nan'))   for s in subjects]) for r in rois]
    x = np.arange(len(rois))
    ax.bar(x - 0.2, ridge_enc_mean, 0.35, label="Ridge regression", color="#9E9E9E", alpha=0.85)
    ax.bar(x + 0.2, geo_enc_mean,   0.35, label="Geometry-aware",   color="#2196F3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=30, ha="right")
    ax.set_ylabel("Median Pearson r"); ax.set_title("A. Encoding Accuracy", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    # Panel B: RSA self-advantage
    ax = axes[1]
    ridge_adv_mean = [np.nanmean([adv_ridge[s].get(r, float('nan')) for s in subjects]) for r in rois]
    geo_adv_mean   = [np.nanmean([adv_geo[s].get(r, float('nan'))   for s in subjects]) for r in rois]
    ax.bar(x - 0.2, ridge_adv_mean, 0.35, label="Ridge regression", color="#9E9E9E", alpha=0.85)
    ax.bar(x + 0.2, geo_adv_mean,   0.35, label="Geometry-aware",   color="#2196F3", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=30, ha="right")
    ax.set_ylabel("RSA self − other (Spearman ρ)")
    ax.set_title("B. RSA Self-Advantage\n(+ve = geometry matches own subject)", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    # Panel C: Fingerprinting
    ax = axes[2]
    ax.bar(x - 0.2, [fp_ridge.get(r, 0) for r in rois], 0.35,
           label="Ridge regression", color="#9E9E9E", alpha=0.85)
    ax.bar(x + 0.2, [fp_geo.get(r, 0)   for r in rois], 0.35,
           label="Geometry-aware",   color="#2196F3", alpha=0.85)
    ax.axhline(1/n_s, color="#F44336", linestyle="--", linewidth=1.5, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=30, ha="right")
    ax.set_ylabel("Identification Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("C. Subject Fingerprinting", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Ridge Regression vs. Geometry-Aware Encoder\n(BOLD5000, real fMRI)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_comparison.pdf", bbox_inches="tight")
    plt.close()

    results = {
        "encoding_ridge": enc_ridge, "encoding_geo": enc_geo,
        "rsa_advantage_ridge": adv_ridge, "rsa_advantage_geo": adv_geo,
        "fingerprinting_ridge": fp_ridge, "fingerprinting_geo": fp_geo,
    }
    def enc(o):
        if isinstance(o, (float, np.floating)): return float(o)
        if isinstance(o, (int,   np.integer)):  return int(o)
        raise TypeError
    with open(RES_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=enc)

    print(f"\nComparison figure → {FIG_DIR}/architecture_comparison.pdf")
    return results


if __name__ == "__main__":
    run_comparison()
