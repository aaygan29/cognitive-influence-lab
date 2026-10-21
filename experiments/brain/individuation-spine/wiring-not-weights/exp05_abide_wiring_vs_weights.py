"""
Experiment 05 (ABIDE variant): wiring-vs-weights on real connectomes, powered, credential-free.
Wiring-is-not-weights program.

Pivot from HCP (gated, Aspera-only) to the ABIDE Preprocessed release on the PUBLIC fcp-indi S3 bucket
(anonymous, no login). CC200 atlas -> 200 common-space ROIs per subject (no voxel mismatch). Downloads a
subset of per-subject ROI timeseries and runs the same test as the HCP design:

Within-subject split-half fingerprinting (first vs second half of the resting scan) under three arms of the
SAME connectome:
  FULL_weighted    : weighted edges (the weights)
  WIRING_binarized : binarized at matched density (topology only)
  NODE_STRENGTH    : per-node summed |edge weight|
Prediction (thesis): FULL >> WIRING -> identity is in the weights, not the wiring. Kill: WIRING ~ FULL.

HONEST CAVEAT: ABIDE is single-session, so this is a WITHIN-session split (easier than Finn's cross-session
test-retest). The FULL-vs-WIRING contrast is still valid (both use the identical split); cross-session
confirmation would need HCP/multi-session data. Multi-site (different scanners/lengths) -> conservative.
"""
import os, json, time, glob, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_abide")
N_SUBJECTS = 300          # subjects to fetch/use (chance = 1/N)
MIN_T = 120               # require >= this many timepoints for a meaningful split
DENSITY = 0.10
N_PERM = 2000
rng = np.random.default_rng(20260619)


def fetch(n_target):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    os.makedirs(DATA, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    pref = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/"
    keys = []
    tok = None
    while True:
        kw = dict(Bucket="fcp-indi", Prefix=pref, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".1D")]
        tok = r.get("NextContinuationToken")
        if not tok: break
    keys = sorted(keys)
    rng.shuffle(keys)
    got = 0
    for k in keys:
        if got >= n_target: break
        fn = os.path.join(DATA, k.split("/")[-1])
        if not os.path.exists(fn):
            s3.download_file("fcp-indi", k, fn)
        got += 1
    print(f"fetched/cached {got} subjects (of {len(keys)} available) -> {DATA}")


def load_1d(path):
    a = np.genfromtxt(path)
    if a.ndim != 2: return None
    a = a[np.all(np.isfinite(a), axis=1)]      # drop header / non-numeric rows
    return a if a.shape[0] >= MIN_T and a.shape[1] >= 50 else None


def netmat(ts):
    ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-9)
    C = np.corrcoef(ts.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    return C


def utri(C):
    iu = np.triu_indices(C.shape[0], 1); return C[iu]


def binarize_density(vec, density):
    k = max(1, int(round(density * vec.size)))
    thr = np.partition(np.abs(vec), -k)[-k]
    return (np.abs(vec) >= thr).astype(np.float64) * np.sign(vec)


def ident(FA, FB):
    A = FA - FA.mean(1, keepdims=True); B = FB - FB.mean(1, keepdims=True)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    M = A @ B.T; N = M.shape[0]
    acc = float(np.mean([np.argmax(M[i]) == i for i in range(N)]))
    acc_r = float(np.mean([np.argmax(M[:, j]) == j for j in range(N)]))
    diag = np.diag(M); off = M.copy(); off[np.diag_indices(N)] = np.nan
    margin = float(np.nanmean(diag[:, None] - off))
    return 0.5 * (acc + acc_r), margin, M


def perm_p(M, acc, n):
    # vectorized label-permutation null: n random permutations at once, no Python loop
    N = M.shape[0]
    pred = M.argmax(1)                          # predicted target per row
    perms = rng.random((n, N)).argsort(1)       # n random permutations, vectorized
    null_acc = (perms == pred[None, :]).mean(1)
    return float((np.sum(null_acc >= acc) + 1) / (n + 1))


def main():
    t0 = time.time()
    fetch(N_SUBJECTS)
    files = sorted(glob.glob(os.path.join(DATA, "*_rois_cc200.1D")))
    A_f, B_f, A_w, B_w, A_s, B_s = [], [], [], [], [], []
    Dref = None; used = 0
    for f in files:
        ts = load_1d(f)
        if ts is None: continue
        if Dref is None: Dref = ts.shape[1]
        if ts.shape[1] != Dref: continue          # keep a single common ROI count
        h = ts.shape[0] // 2
        ca, cb = netmat(ts[:h]), netmat(ts[h:])
        va, vb = utri(ca), utri(cb)
        A_f.append(va); B_f.append(vb)
        A_w.append(binarize_density(va, DENSITY)); B_w.append(binarize_density(vb, DENSITY))
        A_s.append(np.sum(np.abs(ca), 1)); B_s.append(np.sum(np.abs(cb), 1))
        used += 1
    if used < 20:
        print(f"only {used} usable subjects; aborting"); sys.exit(0)

    arms = {"FULL_weighted": (np.array(A_f), np.array(B_f)),
            "WIRING_binarized": (np.array(A_w), np.array(B_w)),
            "NODE_STRENGTH": (np.array(A_s), np.array(B_s))}
    chance = 1.0 / used
    out = {"dataset": "ABIDE cpac/filt_global/rois_cc200 (public fcp-indi)", "atlas": "CC200",
           "n_subjects": used, "n_rois": Dref, "n_edges": int(arms["FULL_weighted"][0].shape[1]),
           "density": DENSITY, "chance": chance, "within_session_split": True, "arms": {}}
    print(f"\n=== Exp05-ABIDE wiring-vs-weights  (N={used}, {Dref} ROIs, "
          f"{out['n_edges']} edges, chance={chance:.4f}) ===")
    print(f"{'arm':18}{'id-acc':>9}{'margin':>9}{'perm-p':>9}")
    for name, (FA, FB) in arms.items():
        acc, margin, M = ident(FA, FB)
        p = perm_p(M, acc, N_PERM)
        out["arms"][name] = {"id_acc": acc, "margin": margin, "perm_p": p}
        print(f"{name:18}{acc:9.3f}{margin:9.4f}{p:9.4f}")

    full = out["arms"]["FULL_weighted"]["id_acc"]; wire = out["arms"]["WIRING_binarized"]["id_acc"]
    out["weighted_minus_binarized"] = full - wire
    out["thesis_supported"] = bool(full > wire + 0.1 and full > 5 * chance)
    out["kill_topology_sufficient"] = bool(wire >= full - 0.05)
    out["runtime_s"] = round(time.time() - t0, 2)
    os.makedirs("results", exist_ok=True)
    with open("results/exp05_abide_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n FULL(weighted) - WIRING(binarized) = {out['weighted_minus_binarized']:+.3f}")
    print(f" thesis_supported (weights >> wiring) = {out['thesis_supported']}")
    print(f" kill_topology_sufficient (wiring ~ weights) = {out['kill_topology_sufficient']}")
    print(f" [caveat] within-session split (not cross-session); multi-site. runtime {out['runtime_s']}s")
    print(f" results -> results/exp05_abide_results.json")


if __name__ == "__main__":
    main()
