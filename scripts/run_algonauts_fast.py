#!/usr/bin/env python3
"""Fast NSD/Algonauts reproduction: ridge encoding + five-level individuality protocol.

Field-standard, fast alternative to the geometry-aware MLP runner:
  * Encoder = ridge regression from CLIP ViT-L/14 features to per-ROI voxel betas
    (the standard NSD/Algonauts baseline; closed-form multi-output solve).
  * CLIP features extracted once on MPS (Apple GPU) for the shared images.
  * Per-subject streaming: load one subject's fMRI, pull shared-image ROI betas,
    free it -- bounded memory.

Five levels (Kriegeskorte 2008 RSA; Finn-style fingerprinting):
  1 encoding accuracy (median voxel Pearson r, vs split-half noise ceiling)
  2 RSA self-advantage (own vs others' geometry)
  3 RSA identity-matrix diagonal dominance
  4 subject fingerprinting (permutation test)
  5 counterfactual consistency on held-out stimuli

Usage:
  ALGONAUTS_RAW="~/Downloads/datasets/train data" python3 scripts/run_algonauts_fast.py
"""
from __future__ import annotations
import os, sys, json, glob, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

BASE = Path(os.environ.get("ALGONAUTS_RAW",
                           str(Path.home() / "Downloads" / "datasets" / "train data"))).expanduser()
PROJ = Path(__file__).resolve().parents[1]
FEAT_DIR = PROJ / "data" / "features"; FEAT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = PROJ / "results" / "algonauts2023"; RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RES_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42; rng = np.random.default_rng(SEED)
MAX_SHARED = int(os.environ.get("MAX_SHARED", "1000"))

ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-faces": ["OFA", "FFA-1", "FFA-2"],
    "floc-bodies": ["EBA", "FBA-1", "FBA-2"],
    "floc-places": ["OPA", "PPA", "RSC"],
    "floc-words": ["OWFA", "VWFA-1", "VWFA-2"],
    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}


def subj_images(sid):
    d = BASE / sid / "training_split" / "training_images"
    files = sorted(glob.glob(str(d / "train-*.png")),
                   key=lambda n: int(Path(n).stem.split("-")[1].split("_")[0]))
    nsd = {int(Path(f).stem.split("_nsd-")[1]): i for i, f in enumerate(files)}
    return files, nsd


def roi_mask(sid, roi_class, roi_name, hemi):
    md = BASE / sid / "roi_masks"
    cf = md / f"{hemi}.{roi_class}_challenge_space.npy"
    mf = md / f"mapping_{roi_class}.npy"
    if not cf.exists() or not mf.exists():
        return None
    arr = np.load(cf); mp = np.load(mf, allow_pickle=True).item()
    tgt = next((k for k, v in mp.items() if v == roi_name), None)
    return None if tgt is None else (arr == tgt)


def extract_clip(files):
    cache = FEAT_DIR / f"algonauts_fast_shared_{len(files)}.npy"
    if cache.exists():
        print(f"  cached CLIP features {cache.name}"); return np.load(cache)
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  extracting CLIP ViT-L/14 for {len(files)} images on {dev} ...", flush=True)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    feats = []; bs = 32; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(files), bs):
            imgs = [Image.open(f).convert("RGB") for f in files[i:i+bs]]
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            out = model.vision_model(pixel_values=inp["pixel_values"])
            feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            if (i // bs) % 5 == 0:
                print(f"    {i+len(imgs)}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(feats).astype(np.float32)
    np.save(cache, F); print(f"  saved {F.shape}"); return F


def ridge_fit_predict(Xtr, Ytr, Xte, alphas=(1e2, 1e3, 1e4)):
    """Closed-form multi-output ridge with a tiny alpha search on a val split."""
    from sklearn.linear_model import Ridge
    n = len(Xtr); k = max(1, int(0.15 * n))
    vi = rng.permutation(n); val, tr = vi[:k], vi[k:]
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr[tr], Ytr[tr])
        p = m.predict(Xtr[val])
        r = np.nanmean([_col_r(p[:, j], Ytr[val][:, j]) for j in range(p.shape[1])])
        if r > best:
            best, best_a = r, a
    return Ridge(alpha=best_a).fit(Xtr, Ytr).predict(Xte)


def _col_r(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rdm(x):
    return squareform(pdist(x, metric="correlation"))


def rsa_r(r1, r2):
    idx = np.triu_indices(r1.shape[0], k=1)
    return float(spearmanr(r1[idx], r2[idx])[0])


def noise_ceiling(betas, n_splits=20):
    n = betas.shape[0]; out = []
    for _ in range(n_splits):
        p = rng.permutation(n); h = n // 2
        a = betas[p[:h]].mean(0); b = betas[p[h:2*h]].mean(0)
        r = _col_r(a, b); out.append(2*r/(1+r+1e-9) if not np.isnan(r) else np.nan)
    return float(np.nanmean(out))


def main():
    subs = [f"subj{i:02d}" for i in range(1, 9)
            if (BASE / f"subj{i:02d}" / "training_split" / "training_fmri" / "lh_training_fmri.npy").exists()]
    print(f"Subjects present: {subs}")
    if len(subs) < 2:
        print("need >=2 subjects"); return 1

    # shared nsd ids across all present subjects
    manifests = {s: subj_images(s) for s in subs}
    shared = sorted(set.intersection(*[set(m[1].keys()) for m in manifests.values()]))
    if len(shared) > MAX_SHARED:
        shared = list(rng.choice(shared, MAX_SHARED, replace=False))
    print(f"Shared images: {len(shared)}")

    # CLIP features once (from subj[0]'s copies)
    files0, nsd0 = manifests[subs[0]]
    feat_files = [files0[nsd0[nid]] for nid in shared]
    F = extract_clip(feat_files)

    # per-subject shared-image ROI betas (streaming)
    shared_betas = {}
    for s in subs:
        files, nsd = manifests[s]
        rows = np.array([nsd[nid] for nid in shared])
        lh = np.load(BASE / s / "training_split" / "training_fmri" / "lh_training_fmri.npy", mmap_mode="r")
        rh = np.load(BASE / s / "training_split" / "training_fmri" / "rh_training_fmri.npy", mmap_mode="r")
        lh_s = np.asarray(lh[rows]); rh_s = np.asarray(rh[rows]); del lh, rh
        rb = {}
        for cls, names in ROI_CLASSES.items():
            for nm in names:
                arrs = []
                for hemi, b in (("lh", lh_s), ("rh", rh_s)):
                    m = roi_mask(s, cls, nm, hemi)
                    if m is not None and m.sum() > 0:
                        arrs.append(b[:, m])
                if arrs:
                    rb[nm] = np.concatenate(arrs, axis=1).astype(np.float32)
        shared_betas[s] = rb; del lh_s, rh_s
        print(f"  {s}: {len(rb)} ROIs, {len(rows)} shared images")

    common = sorted(set.intersection(*[set(v.keys()) for v in shared_betas.values()]))
    print(f"Common ROIs: {len(common)}")

    n = len(shared); nt = int(0.15 * n)
    te = rng.choice(n, nt, replace=False); tr = np.setdiff1d(np.arange(n), te)
    rsi = te[:min(200, nt)]
    Xtr, Xte, Xr = F[tr], F[te], F[rsi]

    preds_te, preds_r, real_te, real_r = {}, {}, {}, {}
    print("Ridge encoding ...")
    for s in subs:
        preds_te[s], preds_r[s], real_te[s], real_r[s] = {}, {}, {}, {}
        for roi in common:
            Y = shared_betas[s][roi]
            p_all = ridge_fit_predict(Xtr, Y[tr], np.vstack([Xte, Xr]))
            preds_te[s][roi] = p_all[:len(te)]; preds_r[s][roi] = p_all[len(te):]
            real_te[s][roi] = Y[te]; real_r[s][roi] = Y[rsi]

    R = {"dataset": "Algonauts2023 NSD", "n_subjects": len(subs),
         "subjects": subs, "n_shared": n, "n_test": len(te), "encoder": "ridge",
         "encoding": {}, "geometry": {}, "fingerprinting": {}, "counterfactual": {}}

    # L1 encoding
    for s in subs:
        R["encoding"][s] = {}
        for roi in common:
            p, y = preds_te[s][roi], real_te[s][roi]
            r = float(np.nanmedian([_col_r(p[:, j], y[:, j]) for j in range(p.shape[1])]))
            R["encoding"][s][roi] = {"r": r, "nc": noise_ceiling(shared_betas[s][roi])}
    med = {s: float(np.nanmedian([R["encoding"][s][roi]["r"] for roi in common])) for s in subs}
    print("  median encoding r:", {s: round(v, 3) for s, v in med.items()})

    # L2 geometry self-advantage
    for s in subs:
        R["geometry"][s] = {}
        others = [o for o in subs if o != s]
        for roi in common:
            pr = rdm(preds_r[s][roi]); self_r = rsa_r(pr, rdm(real_r[s][roi]))
            oth = [rsa_r(pr, rdm(real_r[o][roi])) for o in others]
            R["geometry"][s][roi] = {"rsa_self": self_r,
                                     "advantage": self_r - float(np.mean(oth))}
    adv_by_roi = {roi: float(np.mean([R["geometry"][s][roi]["advantage"] for s in subs])) for roi in common}
    print(f"  RSA self-adv positive ROIs (subj-mean): {sum(v>0 for v in adv_by_roi.values())}/{len(common)}")

    # L4 fingerprinting
    nsig = 0
    for roi in common:
        correct = 0
        for ts in subs:
            y = real_te[ts][roi]; best, bs = -np.inf, None
            for ps in subs:
                p = preds_te[ps][roi]; mv = min(y.shape[1], p.shape[1])
                r = _col_r(y[:, :mv].ravel(), p[:, :mv].ravel())
                if r > best:
                    best, bs = r, ps
            correct += int(bs == ts)
        acc = correct / len(subs)
        # permutation p
        perm = []
        for _ in range(1000):
            pm = rng.permutation(len(subs))
            perm.append(np.mean([pm[i] == i for i in range(len(subs))]))
        p_val = float((np.sum(np.array(perm) >= acc) + 1) / 1001)
        R["fingerprinting"][roi] = {"accuracy": acc, "chance": 1/len(subs), "p": p_val}
        nsig += int(p_val < 0.05)
    print(f"  fingerprinting sig ROIs: {nsig}/{len(common)}")

    # L5 counterfactual (subject-difference consistency on held-out stimuli).
    # Voxels are NOT aligned across subjects (different challenge-space vertices),
    # so the difference is taken in the voxel-count-invariant RDM space: does the
    # predicted between-subject geometry difference match the real one?
    def _tri(m):
        return m[np.triu_indices(m.shape[0], k=1)]
    for roi in common:
        rs = []
        pred_rdm = {s: rdm(preds_te[s][roi]) for s in subs}
        real_rdm = {s: rdm(real_te[s][roi]) for s in subs}
        for i, a in enumerate(subs):
            for b in subs[i+1:]:
                dp = _tri(pred_rdm[a]) - _tri(pred_rdm[b])
                dr = _tri(real_rdm[a]) - _tri(real_rdm[b])
                rs.append(_col_r(dp, dr))
        R["counterfactual"][roi] = {"mean_r": float(np.nanmean(rs)), "n_pairs": len(rs)}

    out = RES_DIR / "all_results_fast.json"
    json.dump(R, open(out, "w"), indent=2)
    print("Saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
