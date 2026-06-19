"""
Experiment 04: real-data pilot on NSD N=8 (260 shared images). Wiring-is-not-weights program.
Uses the validated apparatus (Exp01-03b) on real human fMRI. NO download: all data is local
(~/Downloads/_n8work/matrices/subjXX.npz + digital-brain/data/features/algonauts_shared268_*).

HONEST SCOPE. Two hard constraints of THIS dataset, stated up front:
  (1) N=8 subjects -> chance identification = 1/8 = 0.125; severely underpowered. Top-1 accuracy is coarse
      (steps of 0.125), so we lead with a CONTINUOUS effect (self-minus-other fingerprint similarity) and
      an EXACT permutation p, and pool across ROIs.
  (2) Subjects have DIFFERENT voxel counts, so only voxel-invariant fingerprints are cross-comparable:
        - FUNCTIONAL fingerprint  = predicted image x image RDM (degeneracy- and voxel-invariant).
        - WEIGHT fingerprint       = encoder feature-tuning T = coef^T coef (1024x1024, voxel-invariant)
                                     BUT not degeneracy-invariant (see the WITHIN confound check).

What this pilot tests:
  A) FUNCTIONAL axis: does the individual encoder's predicted representational geometry identify the person?
     (digital-brain N=8 found raw-RDM individuation ~chance; we re-examine with the encoder.)
  B) WEIGHT axis (exploratory): does the encoder feature-tuning fingerprint identify the person, AND is it
     degeneracy-confounded? The WITHIN arm adds weight energy in the stimulus null space: predictions are
     provably unchanged, so any change in the weight fingerprint's identification is CONFOUND, not identity.
"""
import os, json, time
import numpy as np
from math import comb, factorial
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DL = os.path.expanduser("~/Downloads")
MAT = os.path.join(DL, "_n8work", "matrices")
REPO = os.path.expanduser("~/Desktop/Research/Neuro-AI/digital-brain")
SUBS = [f"subj0{i}" for i in range(1, 9)]
nS = len(SUBS)
ALPHA = 1000.0
rng = np.random.default_rng(20260619)


def subfact(k): return round(factorial(k) / np.e) if k >= 0 else 0
def p_ge_fixedpoints(k, N):  # exact P(>= k fixed points in a uniform random permutation of N)
    return float(sum(comb(N, j) * subfact(N - j) for j in range(k, N + 1)) / factorial(N))


def rdm(M):  # 1 - corr over rows (images)
    return 1 - np.corrcoef(M)


def vec_utri(R):
    iu = np.triu_indices(R.shape[0], 1)
    return R[iu]


def ident_stats(M):
    """M[s,t] = similarity of subject s fingerprint to subject t target. Returns acc, margin, p."""
    N = M.shape[0]
    acc_hits = np.array([np.argmax(M[i]) == i for i in range(N)])
    acc = float(acc_hits.mean()); k = int(acc_hits.sum())
    diag = np.diag(M)
    off = M.copy(); off[np.diag_indices(N)] = np.nan
    margin = float(np.nanmean(diag[:, None] - off))   # mean self-minus-other similarity
    return acc, margin, p_ge_fixedpoints(k, N)


def main():
    t0 = time.time()
    common = np.load(os.path.join(DL, "_n8_common_nsd_ids.npy"))
    ids268 = np.load(f"{REPO}/data/features/algonauts_shared268_nsd_ids.npy")
    feat268 = np.load(f"{REPO}/data/features/algonauts_shared268_features.npy")
    id2f = {int(i): k for k, i in enumerate(ids268)}
    X = np.stack([feat268[id2f[int(n)]] for n in common]).astype(np.float64)  # (260, 1024)
    n = len(common)
    idx = rng.permutation(n); tr, te = idx[:208], idx[208:]
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    # null space of the FULL standardized stimulus set (for the degeneracy / WITHIN arm)
    Xall = np.vstack([Xtr, Xte])
    _, _, Vt = np.linalg.svd(Xall, full_matrices=True)
    rankX = np.linalg.matrix_rank(Xall)
    null_basis = Vt[rankX:].T  # (1024, 1024-rankX): feature directions no stimulus excites

    data = {s: dict(np.load(f"{MAT}/{s}.npz")) for s in SUBS}
    rois = sorted(set.intersection(*[set(data[s]) for s in SUBS]))
    print(f"=== Exp04 real NSD N=8 pilot  ({len(rois)} ROIs, {n} images 208/52, chance={1/nS:.3f}) ===")
    print(f"stimulus null-space dim for degeneracy arm = {null_basis.shape[1]} (rankX={rankX})\n")

    rows = []
    for roi in rois:
        coef, predRDM, realRDM, predRDM_within = {}, {}, {}, {}
        within_pred_maxdiff = 0.0
        for s in SUBS:
            Y = data[s][roi].astype(np.float64)
            r = Ridge(alpha=ALPHA).fit(Xtr, Y[tr])
            W = r.coef_                              # (V, 1024) functional (ridge -> in row space of Xtr)
            coef[s] = W
            Pte = Xte @ W.T                          # predicted test responses (52, V)
            predRDM[s] = vec_utri(rdm(Pte))
            realRDM[s] = vec_utri(rdm(Y[te]))
            # WITHIN (degeneracy): add weight energy in the stimulus null space -> predictions UNCHANGED
            Wn = W + (rng.standard_normal((W.shape[0], null_basis.shape[1])) @ null_basis.T) * np.std(W)
            Pte_within = Xte @ Wn.T
            within_pred_maxdiff = max(within_pred_maxdiff, float(np.abs(Pte_within - Pte).max()))
            predRDM_within[s] = Wn                    # store perturbed weights for weight-fingerprint check

        # ---- A) FUNCTIONAL axis: predicted-RDM vs real-RDM identification (voxel- & degeneracy-invariant)
        Pm = np.stack([predRDM[s] for s in SUBS]); Rm = np.stack([realRDM[s] for s in SUBS])
        Pc = Pm - Pm.mean(1, keepdims=True); Rc = Rm - Rm.mean(1, keepdims=True)
        Mf = (Pc @ Rc.T) / (np.linalg.norm(Pc, axis=1)[:, None] * np.linalg.norm(Rc, axis=1)[None, :] + 1e-12)
        acc_f, marg_f, p_f = ident_stats(Mf)

        # ---- B) WEIGHT axis: encoder feature-tuning fingerprint T=coef^T coef (1024x1024), split-half
        # split test-independent: fingerprint from train-fit weights; target from a second image split
        # (use two halves of train images to get two independent weight estimates per subject)
        h = len(tr) // 2; trA, trB = tr[:h], tr[h:]
        TA, TB, TA_within = {}, {}, {}
        for s in SUBS:
            Y = data[s][roi].astype(np.float64)
            WA = Ridge(alpha=ALPHA).fit(sc.transform(X[trA]), Y[trA]).coef_
            WB = Ridge(alpha=ALPHA).fit(sc.transform(X[trB]), Y[trB]).coef_
            ta = WA.T @ WA; tb = WB.T @ WB
            TA[s] = vec_utri(ta) if False else ta[np.triu_indices(ta.shape[0], 1)]
            TB[s] = tb[np.triu_indices(tb.shape[0], 1)]
            WA_w = WA + (rng.standard_normal((WA.shape[0], null_basis.shape[1])) @ null_basis.T) * np.std(WA)
            taw = WA_w.T @ WA_w
            TA_within[s] = taw[np.triu_indices(taw.shape[0], 1)]
        def idmat(Aset, Bset):
            A = np.stack([Aset[s] for s in SUBS]); B = np.stack([Bset[s] for s in SUBS])
            A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
            return (A @ B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :] + 1e-12)
        acc_w, marg_w, p_w = ident_stats(idmat(TA, TB))
        acc_ww, marg_ww, p_ww = ident_stats(idmat(TA_within, TB))   # degeneracy-perturbed weight fingerprint

        rows.append(dict(roi=roi, acc_func=acc_f, margin_func=marg_f, p_func=p_f,
                         acc_weight=acc_w, margin_weight=marg_w, p_weight=p_w,
                         acc_weight_within=acc_ww, margin_weight_within=marg_ww,
                         within_pred_maxdiff=within_pred_maxdiff))

    # ---- pooled summary
    def pooled(key):
        v = np.array([r[key] for r in rows]); return float(v.mean()), float(v.std())
    chance = 1.0 / nS
    summary = dict(n_subjects=nS, n_rois=len(rois), chance=chance,
                   functional_axis=dict(mean_acc=pooled("acc_func")[0], mean_margin=pooled("margin_func")[0],
                                        n_rois_p_lt_05=int(sum(r["p_func"] < 0.05 for r in rows))),
                   weight_axis=dict(mean_acc=pooled("acc_weight")[0], mean_margin=pooled("margin_weight")[0],
                                    n_rois_p_lt_05=int(sum(r["p_weight"] < 0.05 for r in rows))),
                   weight_axis_degeneracy_perturbed=dict(mean_acc=pooled("acc_weight_within")[0],
                                                         mean_margin=pooled("margin_weight_within")[0]),
                   within_prediction_maxdiff=float(max(r["within_pred_maxdiff"] for r in rows)),
                   per_roi=rows, runtime_s=round(time.time() - t0, 2))
    with open("results/exp04_real_pilot_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'ROI':12}{'fAcc':>6}{'fMargin':>9}{'fP':>7} | {'wAcc':>6}{'wMargin':>9}{'wP':>7} | {'wAcc(deg)':>10}")
    for r in rows:
        print(f"{r['roi']:12}{r['acc_func']:6.3f}{r['margin_func']:9.3f}{r['p_func']:7.3f} | "
              f"{r['acc_weight']:6.3f}{r['margin_weight']:9.3f}{r['p_weight']:7.3f} | {r['acc_weight_within']:10.3f}")
    fa, wa = summary["functional_axis"], summary["weight_axis"]
    wd = summary["weight_axis_degeneracy_perturbed"]
    print(f"\nPOOLED (chance acc={chance:.3f}):")
    print(f"  FUNCTIONAL (predicted-RDM, voxel+degeneracy invariant): mean acc={fa['mean_acc']:.3f}  "
          f"mean margin={fa['mean_margin']:+.4f}  ROIs p<.05: {fa['n_rois_p_lt_05']}/{len(rows)}")
    print(f"  WEIGHT (encoder feature-tuning):                        mean acc={wa['mean_acc']:.3f}  "
          f"mean margin={wa['mean_margin']:+.4f}  ROIs p<.05: {wa['n_rois_p_lt_05']}/{len(rows)}")
    print(f"  WEIGHT after degeneracy (null-space) perturbation:      mean acc={wd['mean_acc']:.3f}  "
          f"mean margin={wd['mean_margin']:+.4f}")
    print(f"  [degeneracy check] max change in PREDICTIONS from null-space weight perturbation = "
          f"{summary['within_prediction_maxdiff']:.2e}  (should be ~0: predictions invariant)")
    print(f"  ==> if WEIGHT axis individuates but its margin CHANGES under degeneracy perturbation,")
    print(f"      the weight fingerprint is degeneracy/voxel CONFOUNDED, not clean identity. (See writeup.)")
    print(f"  runtime {summary['runtime_s']}s  results -> results/exp04_real_pilot_results.json")


if __name__ == "__main__":
    main()
