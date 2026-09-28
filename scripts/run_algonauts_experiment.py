"""
Full Algonauts 2023 digital brain experiment.

Strategy for disk-efficient operation:
  - fMRI betas: loaded directly from extracted .npy files
  - Training images: streamed from zip files, never fully extracted
  - Uses the 1000 shared NSD images seen by all 8 subjects for cross-subject eval
  - Per-subject features extracted from zip-streamed images
"""
import sys, json, pickle, warnings, zipfile, io
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
from src.geometry_aware_encoder import GeometryAwareDigitalBrain

# Location of the raw Algonauts 2023 data (per-subject image zips + extracted
# training_fmri + roi_masks). Override with the ALGONAUTS_RAW env var. This used
# to be a hardcoded scratch path; it is now configurable so the reproduction runs
# on any machine once the training bundle is downloaded.
import os
BASE     = Path(os.environ.get("ALGONAUTS_RAW",
                               str(Path.home() / "Downloads" / "train data")))
PROJ     = Path(__file__).parent.parent
FEAT_DIR = PROJ / "data" / "features"
RES_DIR  = PROJ / "results" / "algonauts2023"
MOD_DIR  = RES_DIR / "models"
FIG_DIR  = RES_DIR / "figures"
for d in [FEAT_DIR, RES_DIR, MOD_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SEED = 42
rng  = np.random.default_rng(SEED)

ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-faces":     ["OFA", "FFA-1", "FFA-2"],
    "floc-bodies":    ["EBA", "FBA-1", "FBA-2"],
    "floc-places":    ["OPA", "PPA", "RSC"],
    "floc-words":     ["OWFA", "VWFA-1", "VWFA-2"],
    "streams":        ["early", "midventral", "midlateral", "midparietal",
                       "ventral", "lateral", "parietal"],
}
ALL_ROIS = [r for rois in ROI_CLASSES.values() for r in rois]


# ── Data utilities ────────────────────────────────────────────────────────────

def find_zip(subj_id: str) -> Path:
    # Accept both the challenge naming (subj01.zip) and older split naming
    # (subj01-*.zip), searching BASE recursively so an unzipped "train data/"
    # layout also resolves.
    patterns = [f"{subj_id}.zip", f"{subj_id}-*.zip"]
    matches = []
    for pat in patterns:
        matches += list(BASE.glob(pat)) + list(BASE.glob(f"**/{pat}"))
    if not matches:
        raise FileNotFoundError(
            f"No zip found for {subj_id} under {BASE}. "
            f"Set ALGONAUTS_RAW to the folder holding the per-subject zips.")
    return matches[0]


def get_image_manifest(subj_id: str):
    """Read image names from zip without extracting. Returns {nsd_id: zip_path}."""
    zp = find_zip(subj_id)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist()
                 if f"{subj_id}/training_split/training_images/train-" in n
                 and n.endswith(".png")]
    # Sort by train index
    names.sort(key=lambda n: int(Path(n).stem.split("-")[1].split("_")[0]))
    # Map nsd_id → zip_entry_name
    nsd_to_entry = {}
    for n in names:
        nsd_id = int(Path(n).stem.split("_nsd-")[1])
        nsd_to_entry[nsd_id] = n
    return names, nsd_to_entry


def stream_images_from_zip(subj_id: str, entry_names: list):
    """Yield PIL Images streamed directly from zip, no disk write."""
    from PIL import Image
    zp = find_zip(subj_id)
    with zipfile.ZipFile(zp) as z:
        for name in entry_names:
            with z.open(name) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                yield img


def load_roi_mask(subj_id: str, roi_class: str, roi_name: str, hemi: str) -> np.ndarray:
    mask_dir = BASE / subj_id / "roi_masks"
    class_file = mask_dir / f"{hemi}.{roi_class}_challenge_space.npy"
    mapping_file = mask_dir / f"mapping_{roi_class}.npy"
    if not class_file.exists() or not mapping_file.exists():
        return None
    roi_array = np.load(class_file)
    mapping = np.load(mapping_file, allow_pickle=True).item()
    target = next((k for k, v in mapping.items() if v == roi_name), None)
    if target is None:
        return None
    return (roi_array == target)


def load_roi_betas(subj_id: str, betas_lh: np.ndarray, betas_rh: np.ndarray):
    """Returns {roi_name: (n_images, n_voxels)}."""
    roi_betas = {}
    for roi_class, roi_names in ROI_CLASSES.items():
        for roi_name in roi_names:
            arrays = []
            for hemi, betas in [("lh", betas_lh), ("rh", betas_rh)]:
                mask = load_roi_mask(subj_id, roi_class, roi_name, hemi)
                if mask is not None and mask.sum() > 0:
                    arrays.append(betas[:, mask])
            if arrays:
                roi_betas[roi_name] = np.concatenate(arrays, axis=1)
    return roi_betas


# ── CLIP feature extraction ───────────────────────────────────────────────────

def extract_features_from_zip(subj_id: str, entry_names: list,
                               batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    feat_path = FEAT_DIR / f"algonauts_{subj_id}_shared_features.npy"
    if feat_path.exists():
        f = np.load(feat_path)
        if len(f) == len(entry_names):
            print(f"  {subj_id}: loaded cached features {f.shape}")
            return f

    print(f"  {subj_id}: extracting {len(entry_names)} features from zip...", flush=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    zp = find_zip(subj_id)
    all_feats = []
    with torch.no_grad(), zipfile.ZipFile(zp) as z:
        for i in range(0, len(entry_names), batch_size):
            batch_entries = entry_names[i:i+batch_size]
            imgs = [Image.open(io.BytesIO(z.read(e))).convert("RGB") for e in batch_entries]
            inp = processor(images=imgs, return_tensors="pt")
            out = model.vision_model(pixel_values=inp["pixel_values"])
            all_feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            if (i // batch_size) % 5 == 0:
                print(f"    {i+len(batch_entries)}/{len(entry_names)}", flush=True)

    features = np.concatenate(all_feats)
    np.save(feat_path, features)
    print(f"  {subj_id}: saved {features.shape}")
    return features


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("DIGITAL BRAIN EXPERIMENT — Algonauts 2023 (8 subjects, NSD)")
    print("="*65)

    # ── Step 1: Find 1000 shared NSD images across all 8 subjects ────────────
    print("\nStep 1: Finding shared NSD images...")
    subj_manifests = {}
    for sid in SUBJECTS:
        try:
            names, nsd_map = get_image_manifest(sid)
            subj_manifests[sid] = (names, nsd_map)
            print(f"  {sid}: {len(names)} training images")
        except FileNotFoundError as e:
            print(f"  {sid}: zip not found — skipping")

    available = list(subj_manifests.keys())
    shared_nsd = sorted(set.intersection(*[set(m[1].keys()) for m in subj_manifests.values()]))
    print(f"  Shared NSD images: {len(shared_nsd)}")

    # ── Step 2: Load fMRI betas ───────────────────────────────────────────────
    print("\nStep 2: Loading fMRI betas...")
    all_lh = {}; all_rh = {}; all_roi_betas = {}
    for sid in available:
        lh = np.load(BASE / sid / "training_split" / "training_fmri" / "lh_training_fmri.npy")
        rh = np.load(BASE / sid / "training_split" / "training_fmri" / "rh_training_fmri.npy")
        all_lh[sid] = lh; all_rh[sid] = rh
        all_roi_betas[sid] = load_roi_betas(sid, lh, rh)
        n_rois = len(all_roi_betas[sid])
        print(f"  {sid}: LH{lh.shape} RH{rh.shape} {n_rois} ROIs")

    # Get shared image beta indices per subject
    shared_indices = {}
    for sid in available:
        _, nsd_map = subj_manifests[sid]
        img_names_ordered = subj_manifests[sid][0]
        # Build train_index → nsd_id
        nsd_order = [int(Path(n).stem.split("_nsd-")[1]) for n in img_names_ordered]
        idx_map = {nsd_id: i for i, nsd_id in enumerate(nsd_order)}
        shared_indices[sid] = np.array([idx_map[nid] for nid in shared_nsd])

    print(f"  Shared indices mapped for all {len(available)} subjects")

    # ── Step 3: Extract CLIP features for shared images ───────────────────────
    print("\nStep 3: Extracting CLIP features (streaming from zip)...")
    ref_sid = available[0]
    _, nsd_map = subj_manifests[ref_sid]
    shared_entries = [nsd_map[nid] for nid in shared_nsd]
    shared_features = extract_features_from_zip(ref_sid, shared_entries)
    print(f"  Shared features: {shared_features.shape}")

    # ── Step 4: Build aligned dataset ────────────────────────────────────────
    print("\nStep 4: Building aligned dataset...")
    shared_roi_betas = {}
    for sid in available:
        idx = shared_indices[sid]
        shared_roi_betas[sid] = {roi: betas[idx] for roi, betas in all_roi_betas[sid].items()}
    print(f"  {len(shared_nsd)} images × {len(available)} subjects")

    # Train/test split
    n = len(shared_nsd)
    n_test = int(n * 0.15)
    test_idx  = rng.choice(n, n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    rsa_idx   = test_idx[:min(200, n_test)]

    X_train = shared_features[train_idx]
    X_test  = shared_features[test_idx]
    X_rsa   = shared_features[rsa_idx]
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

    common_rois = sorted(set.intersection(*[set(b.keys()) for b in shared_roi_betas.values()]))
    print(f"  Common ROIs: {len(common_rois)}")

    # ── Step 5: Train digital brains ─────────────────────────────────────────
    print("\nStep 5: Training digital brains (geometry-aware)...")

    def train_brain(sid):
        cache = MOD_DIR / f"geo_brain_{sid}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                brain = pickle.load(f)
            print(f"  {sid}: loaded cached brain", flush=True)
            return brain
        roi_train = {r: shared_roi_betas[sid][r][train_idx] for r in common_rois}
        brain = GeometryAwareDigitalBrain(sid, alpha=1.0, beta=0.6, gamma=0.3,
                                          hidden_dim=512, n_voxel_pcs=100,
                                          n_epochs=200, lr=3e-4, batch_size=64)
        brain.fit(X_train, roi_train, verbose=False)
        brain.save(cache)
        print(f"  {sid}: trained", flush=True)
        return brain

    brains = {sid: train_brain(sid) for sid in available}

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    preds_test = {s: b.predict(X_test) for s, b in brains.items()}
    preds_rsa  = {s: b.predict(X_rsa)  for s, b in brains.items()}
    real_test  = {s: {r: shared_roi_betas[s][r][test_idx] for r in common_rois} for s in available}
    real_rsa   = {s: {r: shared_roi_betas[s][r][rsa_idx]  for r in common_rois} for s in available}
    nc_betas   = {s: {r: shared_roi_betas[s][r] for r in common_rois} for s in available}

    def med_r(pred, real):
        mv = min(pred.shape[1], real.shape[1])
        return float(np.nanmedian([pearsonr(pred[:,v], real[:,v])[0] for v in range(mv)]))

    def nc_est(betas, n_splits=20):
        n = betas.shape[0]; ncs = []
        for _ in range(n_splits):
            idx = rng.permutation(n); h1 = betas[idx[:n//2]].mean(0); h2 = betas[idx[n//2:2*(n//2)]].mean(0)
            r, _ = pearsonr(h1, h2); ncs.append(2*r/(1+r+1e-9))
        return float(np.mean(ncs))

    def rdm(x): return squareform(pdist(x, metric="correlation"))
    def rsa_r(r1, r2): n=r1.shape[0]; idx=np.triu_indices(n,k=1); return float(spearmanr(r1[idx],r2[idx])[0])

    all_results = {}
    n_s = len(available)

    # Level 1: Encoding
    print("\nLEVEL 1: Encoding Accuracy")
    enc = {}
    for sid in available:
        enc[sid] = {}
        for roi in common_rois[:8]:  # top 8 for display
            pred = preds_test[sid].get(roi); real = real_test[sid].get(roi)
            if pred is None or real is None: continue
            r = med_r(pred, real)
            nc = nc_est(nc_betas[sid][roi])
            enc[sid][roi] = {"r": r, "nc": nc, "nc_ratio": r/max(nc,1e-6)}
    # Print summary
    for roi in list(common_rois)[:8]:
        rs  = [enc[s][roi]["r"]  for s in available if roi in enc[s]]
        ncs = [enc[s][roi]["nc"] for s in available if roi in enc[s]]
        print(f"  {roi:<15} r={np.mean(rs):.3f}±{np.std(rs):.3f}  NC={np.mean(ncs):.3f}")
    all_results["encoding"] = enc

    # Level 2: RSA self-advantage
    print("\nLEVEL 2: RSA Self-Advantage")
    geo = {}
    for sid in available:
        geo[sid] = {}
        others = [s for s in available if s != sid]
        for roi in common_rois[:8]:
            pred = preds_rsa[sid].get(roi); rs = real_rsa[sid].get(roi)
            if pred is None or rs is None: continue
            r_self = rsa_r(rdm(pred), rdm(rs))
            r_others = [rsa_r(rdm(pred), rdm(real_rsa[o][roi])) for o in others if roi in real_rsa[o]]
            geo[sid][roi] = {"rsa_self": r_self, "rsa_other": float(np.mean(r_others)) if r_others else np.nan,
                              "advantage": r_self - float(np.mean(r_others)) if r_others else np.nan}
    for roi in list(common_rois)[:8]:
        self_rs = [geo[s][roi]["rsa_self"] for s in available if roi in geo[s]]
        advs    = [geo[s][roi]["advantage"] for s in available if roi in geo[s]]
        print(f"  {roi:<15} RSA_self={np.mean(self_rs):.3f}  Δ={np.nanmean(advs):+.3f}")
    all_results["geometry"] = geo

    # Level 3: RSA Identity Matrix
    print("\nLEVEL 3: RSA Identity Matrix")
    rsa_mats = {}
    for roi in list(common_rois)[:5]:
        mat = np.zeros((n_s, n_s))
        for i, sp in enumerate(available):
            pred = preds_rsa[sp].get(roi)
            if pred is None: continue
            rp = rdm(pred)
            for j, sr in enumerate(available):
                rr = real_rsa[sr].get(roi)
                if rr is not None: mat[i,j] = rsa_r(rp, rdm(rr))
        rsa_mats[roi] = mat
        dom = sum(1 for i in range(n_s) if np.argmax(mat[i]) == i) / n_s
        print(f"  {roi:<15} diagonal dominance = {dom:.0%} ({int(dom*n_s)}/{n_s})")
    all_results["rsa_matrices"] = {k: v.tolist() for k, v in rsa_mats.items()}

    # Level 4: Fingerprinting
    print("\nLEVEL 4: Subject Fingerprinting")
    fp = {}
    for roi in common_rois[:8]:
        correct = 0
        for ts in available:
            real = real_test[ts].get(roi)
            if real is None: continue
            best_r, best_s = -np.inf, None
            for ps in available:
                pred = preds_test[ps].get(roi)
                if pred is None: continue
                mv = min(real.shape[1], pred.shape[1])
                r = np.corrcoef(real[:,:mv].ravel(), pred[:,:mv].ravel())[0,1]
                if r > best_r: best_r, best_s = r, ps
            if best_s == ts: correct += 1
        acc = correct / n_s; chance = 1/n_s
        perm_rng = np.random.default_rng(SEED)
        n_above = sum(1 for _ in range(1000)
                      if sum(1 for ts,ps in zip(available, perm_rng.permutation(available)) if ts==ps)/n_s >= acc)
        fp[roi] = {"accuracy": acc, "chance": chance, "p": n_above/1000}
        print(f"  {roi:<15} acc={acc:.0%}  chance={chance:.1%}  p={n_above/1000:.4f} "
              f"{'***' if n_above<1 else '**' if n_above<10 else '*' if n_above<50 else 'ns'}")
    all_results["fingerprinting"] = fp

    # Level 5: Counterfactual
    print("\nLEVEL 5: Counterfactual Consistency")
    cf = {}
    cf_idx = np.arange(min(150, len(test_idx)))
    for roi in common_rois[:8]:
        pairs = []
        for ai, sa in enumerate(available):
            for sb in available[ai+1:]:
                ra=real_test[sa].get(roi); rb=real_test[sb].get(roi)
                pa=preds_test[sa].get(roi); pb=preds_test[sb].get(roi)
                if any(x is None for x in [ra,rb,pa,pb]): continue
                mv = min(ra.shape[1],rb.shape[1],pa.shape[1],pb.shape[1])
                r, _ = pearsonr((ra[cf_idx,:mv]-rb[cf_idx,:mv]).ravel(),
                                (pa[cf_idx,:mv]-pb[cf_idx,:mv]).ravel())
                pairs.append(r)
        if pairs:
            cf[roi] = {"mean_r": float(np.mean(pairs)), "n_pairs": len(pairs)}
            print(f"  {roi:<15} mean_r={np.mean(pairs):.3f}±{np.std(pairs):.3f} (n={len(pairs)})")
    all_results["counterfactual"] = cf

    # ── Figures ───────────────────────────────────────────────────────────────
    rois_plot = list(common_rois)[:8]

    # Encoding figure
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rois_plot))
    mean_rs  = [np.mean([enc[s][r]["r"]  for s in available if r in enc[s]]) for r in rois_plot]
    std_rs   = [np.std ([enc[s][r]["r"]  for s in available if r in enc[s]]) for r in rois_plot]
    mean_ncs = [np.mean([enc[s][r]["nc"] for s in available if r in enc[s]]) for r in rois_plot]
    ax.bar(x, mean_rs, color="#2196F3", alpha=0.85, yerr=std_rs, capsize=3, label="Digital brain r")
    ax.plot(x, mean_ncs, "o--", color="#FF9800", markersize=6, label="Noise ceiling")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(rois_plot, rotation=30, ha="right")
    ax.set_ylabel("Median Pearson r"); ax.legend()
    ax.set_title(f"Encoding Accuracy — Digital Brain vs Noise Ceiling\n"
                 f"Algonauts 2023 ({n_s} subjects, {len(shared_nsd)} shared images)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "encoding_accuracy.pdf", bbox_inches="tight"); plt.close()

    # Fingerprinting figure
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(rois_plot))
    accs = [fp.get(r,{}).get("accuracy",0) for r in rois_plot]
    ps   = [fp.get(r,{}).get("p",1.0)      for r in rois_plot]
    ax.bar(x, accs, color=["#2196F3" if p<.05 else "#9E9E9E" for p in ps], alpha=0.85)
    ax.axhline(1/n_s, color="#F44336", linestyle="--", linewidth=1.5, label=f"Chance ({1/n_s:.2f})")
    for xi,(a,p) in enumerate(zip(accs,ps)):
        stars = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        if stars: ax.text(xi, a+.01, stars, ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(rois_plot, rotation=30, ha="right")
    ax.set_ylabel("Identification Accuracy"); ax.set_ylim(0, 1.15); ax.legend()
    ax.set_title(f"Subject Fingerprinting — {n_s} subjects, chance={1/n_s:.1%}", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fingerprinting.pdf", bbox_inches="tight"); plt.close()

    # RSA identity matrix (best ROI) — with p-value annotation per subject
    best_roi = max(rsa_mats, key=lambda r: sum(1 for i in range(n_s) if np.argmax(rsa_mats[r][i])==i))
    mat = rsa_mats[best_roi]

    # Per-subject p-value: probability diagonal entry is max under permutation
    rng_rsa = np.random.default_rng(42)
    diag_pvals = []
    for subj_i in range(n_s):
        diag_val = mat[subj_i, subj_i]
        row = mat[subj_i]
        # Permutation: shuffle column assignments 1000x, fraction where diagonal ≥ observed
        n_above = sum(1 for _ in range(1000) if rng_rsa.choice(row) >= diag_val)
        diag_pvals.append(n_above / 1000)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.6)
    plt.colorbar(im, ax=ax, label="RSA (Spearman ρ)")
    labels = [s for s in available]
    ax.set_xticks(range(n_s)); ax.set_yticks(range(n_s))
    ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Real brain (subject)"); ax.set_ylabel("Digital brain (subject)")
    ax.set_title(
        f"RSA Identity Matrix — {best_roi}  (N={n_s}, chance=1/{n_s})\n"
        f"Blue box = matched subject; * = p<0.05 diagonal by permutation test",
        fontweight="bold", fontsize=9)
    for i in range(n_s):
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor="blue",lw=2))
        for j in range(n_s):
            ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=6)
        # Annotate diagonal p-value
        pv = diag_pvals[i]
        stars = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
        ax.text(i, i-0.35, stars, ha="center", va="center", fontsize=7,
                color="blue", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"rsa_identity_{best_roi}.pdf", bbox_inches="tight")
    plt.close()

    # Save results
    def enc_json(o):
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        raise TypeError
    with open(RES_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=enc_json)

    print(f"\n{'='*65}")
    print("EXPERIMENT COMPLETE")
    print(f"  Results: {RES_DIR}/")
    print(f"  Figures: {FIG_DIR}/")


if __name__ == "__main__":
    main()
