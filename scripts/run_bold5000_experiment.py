"""
End-to-end digital brain experiment on BOLD5000 real fMRI data.
Runs all 5 validation levels and saves publication figures.
"""
import sys, json, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
FEAT_DIR  = Path(__file__).parent.parent / "data" / "features"
RES_DIR   = Path(__file__).parent.parent / "results" / "bold5000"
MODELS_DIR = RES_DIR / "models"
FIG_DIR   = RES_DIR / "figures"
for d in [RES_DIR, MODELS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROIS    = ["EarlyVis", "OPA", "PPA", "RSC", "LOC"]
SEED    = 42
rng     = np.random.default_rng(SEED)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset():
    feat_path = FEAT_DIR / "bold5000_clip_features.npy"
    if not feat_path.exists():
        print(f"Features not found. Run: python3 scripts/extract_bold5000_features.py")
        sys.exit(1)
    features = np.load(feat_path)

    meta_path = PROC_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"Processed data not found. Run: python3 scripts/build_bold5000_dataset.py")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    subjects = meta["subjects"]
    rois = meta["rois"]

    all_betas = {}
    for subj in subjects:
        subj_betas = {}
        for roi in rois:
            p = PROC_DIR / subj / f"{roi}_betas.npy"
            if p.exists():
                subj_betas[roi] = np.load(p)
        all_betas[subj] = subj_betas
    return features, all_betas, subjects, rois


# ── Digital Brain (encoding model) ──────────────────────────────────────────

class DigitalBrain:
    def __init__(self, subj_id, alphas=None):
        self.subj_id = subj_id
        self.alphas  = alphas or [0.1, 1, 10, 100, 1000, 10000, 100000]
        self.models  = {}

    def fit(self, X_train, roi_betas_train):
        print(f"  Training digital brain: {self.subj_id}")
        for roi, betas in roi_betas_train.items():
            n_vox = betas.shape[1]
            n_pcs = min(100, n_vox, X_train.shape[0] - 1)
            vpca  = PCA(n_components=n_pcs, random_state=SEED)
            betas_pcs = vpca.fit_transform(betas)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=min(500, X_train.shape[1]), random_state=SEED)),
                ("ridge", RidgeCV(alphas=self.alphas, cv=3)),
            ])
            pipe.fit(X_train, betas_pcs)
            self.models[roi] = (pipe, vpca)
        return self

    def predict(self, X):
        return {roi: vpca.inverse_transform(pipe.predict(X))
                for roi, (pipe, vpca) in self.models.items()}


def load_or_train(subj, X_train, roi_betas_train, force=False):
    path = MODELS_DIR / f"{subj}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            brain = pickle.load(f)
        print(f"  Loaded cached: {subj}")
        return brain
    brain = DigitalBrain(subj)
    brain.fit(X_train, roi_betas_train)
    with open(path, "wb") as f:
        pickle.dump(brain, f)
    return brain


# ── Metrics ──────────────────────────────────────────────────────────────────

def median_r(pred, real):
    min_v = min(pred.shape[1], real.shape[1])
    rs = [pearsonr(pred[:, v], real[:, v])[0] for v in range(min_v)]
    return float(np.nanmedian(rs))

def noise_ceiling(betas, n_splits=20):
    n = betas.shape[0]
    ncs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        h1 = betas[idx[:n//2]].mean(0)
        h2 = betas[idx[n//2:2*(n//2)]].mean(0)
        r, _ = pearsonr(h1, h2)
        ncs.append(2 * r / (1 + r + 1e-9))
    return float(np.mean(ncs))

def rdm(act):
    return squareform(pdist(act, metric="correlation"))

def rsa(r1, r2):
    n = r1.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(spearmanr(r1[idx], r2[idx])[0])


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("DIGITAL BRAIN EXPERIMENT — BOLD5000 Real fMRI")
    print("="*60)

    features, all_betas, subjects, avail_rois = load_dataset()
    rois = [r for r in ROIS if r in avail_rois]
    n_total = features.shape[0]
    print(f"\nSubjects: {subjects}  |  ROIs: {rois}  |  Images: {n_total}")

    # Train/test split
    n_test = int(n_total * 0.15)
    test_idx  = rng.choice(n_total, n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)
    rsa_idx   = test_idx[:min(200, n_test)]
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test")

    X_train, X_test, X_rsa = features[train_idx], features[test_idx], features[rsa_idx]

    # Train digital brains
    print("\nTraining digital brains...")
    brains = {}
    for subj in subjects:
        roi_betas_train = {roi: betas[train_idx]
                           for roi, betas in all_betas[subj].items() if roi in rois}
        brains[subj] = load_or_train(subj, X_train, roi_betas_train)

    # Generate predictions
    preds_test = {s: b.predict(X_test) for s, b in brains.items()}
    preds_rsa  = {s: b.predict(X_rsa)  for s, b in brains.items()}
    real_test  = {s: {r: betas[test_idx] for r, betas in all_betas[s].items()} for s in subjects}
    real_rsa   = {s: {r: betas[rsa_idx]  for r, betas in all_betas[s].items()} for s in subjects}

    all_results = {}

    # ── Level 1: Encoding Accuracy ─────────────────────────────────────────
    print("\nLEVEL 1: Encoding Accuracy")
    enc_res = {}
    for subj in subjects:
        enc_res[subj] = {}
        for roi in rois:
            if roi not in preds_test[subj] or roi not in real_test[subj]:
                continue
            pred = preds_test[subj][roi]
            real = real_test[subj][roi]
            r = median_r(pred, real)
            nc = noise_ceiling(all_betas[subj][roi])
            enc_res[subj][roi] = {"r": r, "nc": nc, "nc_ratio": r/max(nc,1e-6)}
            print(f"  {subj}/{roi}: r={r:.3f}  NC={nc:.3f}  ratio={r/max(nc,1e-6):.2f}")
    all_results["encoding"] = enc_res

    # ── Level 2: RSA Geometry ──────────────────────────────────────────────
    print("\nLEVEL 2: Representational Geometry (RSA)")
    geo_res = {}
    for subj in subjects:
        geo_res[subj] = {}
        others = [s for s in subjects if s != subj]
        for roi in rois:
            if roi not in preds_rsa[subj] or roi not in real_rsa[subj]:
                continue
            rdm_pred = rdm(preds_rsa[subj][roi])
            rdm_self = rdm(real_rsa[subj][roi])
            r_self = rsa(rdm_pred, rdm_self)
            r_others = []
            for o in others:
                if roi in real_rsa[o]:
                    r_others.append(rsa(rdm_pred, rdm(real_rsa[o][roi])))
            geo_res[subj][roi] = {
                "rsa_self": r_self,
                "rsa_other": float(np.mean(r_others)) if r_others else float("nan"),
                "advantage": r_self - float(np.mean(r_others)) if r_others else float("nan"),
            }
            adv = geo_res[subj][roi]["advantage"]
            print(f"  {subj}/{roi}: self={r_self:.3f}  other={np.mean(r_others):.3f}  Δ={adv:+.3f}")
    all_results["geometry"] = geo_res

    # ── Level 3: Cross-Subject RSA Identity Matrix ─────────────────────────
    print("\nLEVEL 3: RSA Identity Matrix")
    n_s = len(subjects)
    rsa_matrices = {}
    for roi in rois[:4]:
        mat = np.zeros((n_s, n_s))
        for i, sp in enumerate(subjects):
            if roi not in preds_rsa[sp]:
                continue
            rdm_p = rdm(preds_rsa[sp][roi])
            for j, sr in enumerate(subjects):
                if roi not in real_rsa[sr]:
                    continue
                mat[i, j] = rsa(rdm_p, rdm(real_rsa[sr][roi]))
        rsa_matrices[roi] = mat
        dom = sum(1 for i in range(n_s) if np.argmax(mat[i]) == i) / n_s
        print(f"  {roi}: diagonal dominance = {dom:.0%}  ({int(dom*n_s)}/{n_s} subjects)")

        # Plot
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.6)
        plt.colorbar(im, ax=ax, label="RSA (Spearman ρ)")
        labels = [s for s in subjects]
        ax.set_xticks(range(n_s)); ax.set_yticks(range(n_s))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Real brain"); ax.set_ylabel("Digital brain")
        ax.set_title(f"RSA Identity Matrix — {roi}", fontweight="bold")
        for i in range(n_s):
            ax.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False, edgecolor="blue", lw=2))
            for j in range(n_s):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"rsa_identity_{roi}.pdf", bbox_inches="tight")
        plt.close()
    all_results["rsa_matrices"] = {k: v.tolist() for k, v in rsa_matrices.items()}

    # ── Level 4: Fingerprinting ────────────────────────────────────────────
    print("\nLEVEL 4: Subject Fingerprinting")
    fp_res = {}
    for roi in rois:
        correct = 0
        for true_subj in subjects:
            if roi not in preds_test[true_subj] or roi not in real_test[true_subj]:
                continue
            real = real_test[true_subj][roi]
            best_r, best_s = -np.inf, None
            for pred_subj in subjects:
                if roi not in preds_test[pred_subj]:
                    continue
                pred = preds_test[pred_subj][roi]
                min_v = min(real.shape[1], pred.shape[1])
                r = np.corrcoef(real[:,:min_v].ravel(), pred[:,:min_v].ravel())[0,1]
                if r > best_r:
                    best_r, best_s = r, pred_subj
            if best_s == true_subj:
                correct += 1

        acc = correct / n_s
        chance = 1.0 / n_s
        # Permutation p-value
        perm_rng = np.random.default_rng(SEED)
        n_above = 0
        for _ in range(1000):
            perm = perm_rng.permutation(subjects)
            perm_correct = sum(1 for ts, ps in zip(subjects, perm)
                               if ts == ps)
            if perm_correct / n_s >= acc:
                n_above += 1
        perm_p = n_above / 1000
        fp_res[roi] = {"accuracy": acc, "chance": chance, "p": perm_p}
        sig = "***" if perm_p < .001 else "**" if perm_p < .01 else "*" if perm_p < .05 else "ns"
        print(f"  {roi}: acc={acc:.0%}  chance={chance:.0%}  p={perm_p:.4f} {sig}")
    all_results["fingerprinting"] = fp_res

    # Fingerprint figure
    fp_rois = sorted(fp_res.keys())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(fp_rois))
    accs = [fp_res[r]["accuracy"] for r in fp_rois]
    ps   = [fp_res[r]["p"] for r in fp_rois]
    ax.bar(x, accs, color=["#2196F3" if p < .05 else "#9E9E9E" for p in ps], alpha=0.85)
    ax.axhline(1/n_s, color="#F44336", linestyle="--", linewidth=1.5, label=f"Chance ({1/n_s:.2f})")
    for xi, (acc, p) in enumerate(zip(accs, ps)):
        stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        if stars: ax.text(xi, acc + .01, stars, ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(fp_rois, fontsize=11)
    ax.set_ylabel("Identification Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Subject Fingerprinting from Digital Brain\n(real BOLD5000 fMRI)", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fingerprinting.pdf", bbox_inches="tight")
    plt.close()

    # ── Level 5: Counterfactual Consistency ───────────────────────────────
    print("\nLEVEL 5: Counterfactual Consistency")
    cf_res = {}
    cf_idx = np.arange(min(200, len(test_idx)))
    for roi in rois:
        pairs = []
        for ai, sa in enumerate(subjects):
            for sb in subjects[ai+1:]:
                if roi not in preds_test[sa] or roi not in preds_test[sb]:
                    continue
                if roi not in real_test[sa] or roi not in real_test[sb]:
                    continue
                ra = real_test[sa][roi][cf_idx]; rb = real_test[sb][roi][cf_idx]
                pa = preds_test[sa][roi][cf_idx]; pb = preds_test[sb][roi][cf_idx]
                min_v = min(ra.shape[1], rb.shape[1], pa.shape[1], pb.shape[1])
                real_diff = (ra[:,:min_v] - rb[:,:min_v]).ravel()
                pred_diff = (pa[:,:min_v] - pb[:,:min_v]).ravel()
                r, _ = pearsonr(real_diff, pred_diff)
                pairs.append(r)
        if pairs:
            cf_res[roi] = {"mean_r": float(np.mean(pairs)), "std_r": float(np.std(pairs)),
                           "n_pairs": len(pairs)}
            print(f"  {roi}: mean_r={np.mean(pairs):.3f} ± {np.std(pairs):.3f}  (n={len(pairs)} pairs)")
    all_results["counterfactual"] = cf_res

    # Counterfactual figure
    fig, ax = plt.subplots(figsize=(9, 4))
    cf_rois = sorted(cf_res.keys())
    means = [cf_res[r]["mean_r"] for r in cf_rois]
    stds  = [cf_res[r]["std_r"]  for r in cf_rois]
    ax.bar(range(len(cf_rois)), means, color="#2196F3", alpha=0.85, yerr=stds, capsize=4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(cf_rois))); ax.set_xticklabels(cf_rois, fontsize=11)
    ax.set_ylabel("Counterfactual r"); ax.set_ylim(-0.1, max(means)*1.3 if means else 0.5)
    ax.set_title("Counterfactual Consistency\nSubject-difference vectors captured by digital brain",
                 fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual.pdf", bbox_inches="tight")
    plt.close()

    # ── Encoding figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(subjects), figsize=(4*len(subjects), 4), sharey=True)
    if len(subjects) == 1:
        axes = [axes]
    for ax, subj in zip(axes, subjects):
        rs  = [enc_res[subj].get(roi, {}).get("r", 0)  for roi in rois]
        ncs = [enc_res[subj].get(roi, {}).get("nc", 0) for roi in rois]
        x = np.arange(len(rois))
        ax.bar(x, rs, color="#2196F3", alpha=0.85, label="Digital brain r")
        ax.plot(x, ncs, "o--", color="#FF9800", markersize=5, label="Noise ceiling")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(rois, rotation=30, ha="right")
        ax.set_title(subj, fontweight="bold"); ax.set_ylim(-0.05, max(max(ncs)+0.05, 0.5))
        if ax == axes[0]:
            ax.set_ylabel("Median Pearson r")
            ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Encoding Accuracy — Digital Brain vs Noise Ceiling\n(BOLD5000 Real fMRI)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "encoding_accuracy.pdf", bbox_inches="tight")
    plt.close()

    # ── Summary ────────────────────────────────────────────────────────────
    def srl(v): return f"{v:.3f}" if v == v else "--"  # handles nan

    print("\n" + "="*70)
    print(f"{'ROI':<12} {'Enc r':>7} {'NC':>7} {'NC%':>6} {'RSA_self':>10} {'RSA_Δ':>8} {'FP acc':>8} {'CF r':>8}")
    print("-"*70)
    for roi in rois:
        rs     = [enc_res[s].get(roi,{}).get("r",   float("nan")) for s in subjects]
        ncs    = [enc_res[s].get(roi,{}).get("nc",  float("nan")) for s in subjects]
        ratios = [enc_res[s].get(roi,{}).get("nc_ratio", float("nan")) for s in subjects]
        selfs  = [geo_res[s].get(roi,{}).get("rsa_self",  float("nan")) for s in subjects]
        advs   = [geo_res[s].get(roi,{}).get("advantage", float("nan")) for s in subjects]
        fp_acc = fp_res.get(roi, {}).get("accuracy", float("nan"))
        cf_r   = cf_res.get(roi, {}).get("mean_r",   float("nan"))
        print(f"{roi:<12} {srl(np.nanmean(rs)):>7} {srl(np.nanmean(ncs)):>7} "
              f"{srl(np.nanmean(ratios)*100):>6} {srl(np.nanmean(selfs)):>10} "
              f"{srl(np.nanmean(advs)):>8} {srl(fp_acc):>8} {srl(cf_r):>8}")

    with open(RES_DIR / "all_results.json", "w") as f:
        def enc(o):
            if isinstance(o, (np.integer, np.floating)): return o.item()
            raise TypeError
        json.dump(all_results, f, indent=2, default=enc)

    print(f"\nResults saved to {RES_DIR}/")
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
