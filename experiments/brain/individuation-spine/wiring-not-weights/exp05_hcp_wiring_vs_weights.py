"""
Experiment 05: wiring-vs-weights on HCP resting-state connectomes (~1000 subjects).
Wiring-is-not-weights program. Powered, common-space (no voxel mismatch), single-package.

THE TEST: Finn-style test-retest identification (match a subject's session-A connectome to their session-B
connectome among all subjects) under three arms of the SAME connectome:
  FULL          : weighted netmat (edge strengths)            -- the weights.
  WIRING        : binarized netmat at matched density          -- topology only (which edges, not how strong).
  NODE_STRENGTH : per-node summed |edge weight| (D-dim)        -- coarse weighted summary.
Prediction (thesis): FULL (weighted) identifies strongly (Finn ~90%+); WIRING (binarized topology) is much
weaker -> identity is in the weights, not the wiring. WIRING above chance is expected (topology carries
SOME identity) but should be well below FULL. Kill: if WIRING ~ FULL, topology alone suffices and the
connectome-level thesis is wrong (publishable either way).

Controls: matched edge density between FULL and WIRING; label-permutation null for p-values; split-half by
RUNS (true test-retest) not random timepoints.

Run after setting HCP_PTN_DIR per HCP_DATA_INSTRUCTIONS.md. The code self-checks and reports if data missing.
"""
import os, sys, glob, json, time
import numpy as np

DENSITY = 0.10          # top-10% |edges| kept for the WIRING (binarized) arm; FULL is density-matched in info
N_PERM = 2000           # label-permutation null
rng = np.random.default_rng(20260619)


def find_data():
    root = os.environ.get("HCP_PTN_DIR", "")
    cands = [root] if root else []
    cands += [os.path.expanduser(p) for p in
              ["~/Downloads/HCP_PTN", "~/Downloads/HCP1200_PTN", "~/Desktop/Research/Neuro-AI/data/HCP_PTN"]]
    for c in cands:
        if not c:
            continue
        ts_dirs = sorted(glob.glob(os.path.join(c, "node_timeseries", "*d*_ts*")))
        if ts_dirs:
            return c, ts_dirs[-1]   # highest-D timeseries dir found
    return None, None


def load_subject_ts(ts_dir, sid):
    f = os.path.join(ts_dir, f"{sid}.txt")
    if not os.path.exists(f):
        return None
    a = np.loadtxt(f)
    return a if a.ndim == 2 else a.reshape(-1, 1)


def netmat(ts):  # full-correlation connectome, z-scored nodes
    ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-9)
    C = np.corrcoef(ts.T)
    np.fill_diagonal(C, 0.0)
    return C


def utri(C):
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu]


def binarize_density(vec, density):
    k = max(1, int(round(density * vec.size)))
    thr = np.partition(np.abs(vec), -k)[-k]
    return (np.abs(vec) >= thr).astype(np.float64) * np.sign(vec)


def ident_acc(FA, FB):
    """rows = subjects; FA session-A features, FB session-B. acc = argmax-match; also mean self-other margin."""
    A = FA - FA.mean(1, keepdims=True); B = FB - FB.mean(1, keepdims=True)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    M = A @ B.T
    N = M.shape[0]
    acc = float(np.mean([np.argmax(M[i]) == i for i in range(N)]))
    acc_rev = float(np.mean([np.argmax(M[:, j]) == j for j in range(N)]))
    diag = np.diag(M).copy(); off = M.copy(); off[np.diag_indices(N)] = np.nan
    margin = float(np.nanmean(diag[:, None] - off))
    return 0.5 * (acc + acc_rev), margin, M


def perm_p(M, observed_acc, n_perm):
    N = M.shape[0]; hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        acc = np.mean([np.argmax(M[i]) == perm[i] for i in range(N)])
        hits += int(acc >= observed_acc)
    return (hits + 1) / (n_perm + 1)


def main():
    root, ts_dir = find_data()
    if ts_dir is None:
        print("HCP PTN data not found. See HCP_DATA_INSTRUCTIONS.md.")
        print("Set HCP_PTN_DIR to the unzipped PTN folder (must contain node_timeseries/*d*_ts*/ and "
              "subjectIDs.txt), then re-run.")
        sys.exit(0)
    t0 = time.time()
    sid_file = os.path.join(root, "subjectIDs.txt")
    sids = [l.strip() for l in open(sid_file)] if os.path.exists(sid_file) else \
           [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ts_dir, "*.txt"))]
    print(f"=== Exp05 HCP wiring-vs-weights ===\n ts_dir={ts_dir}\n {len(sids)} subject IDs listed")

    A_full, B_full, A_wire, B_wire, A_str, B_str = [], [], [], [], [], []
    used = 0
    for sid in sids:
        ts = load_subject_ts(ts_dir, sid)
        if ts is None or ts.shape[0] < 8:
            continue
        half = ts.shape[0] // 2                       # split by timepoints (runs) -> test-retest
        for store_f, store_w, store_s, seg in [(A_full, A_wire, A_str, ts[:half]),
                                               (B_full, B_wire, B_str, ts[half:])]:
            C = netmat(seg); v = utri(C)
            store_f.append(v)
            store_w.append(binarize_density(v, DENSITY))
            store_s.append(np.sum(np.abs(C), axis=1))   # node strength (D-dim)
        used += 1
    if used < 10:
        print(f"Only {used} usable subjects found; need the full PTN node_timeseries. Aborting.")
        sys.exit(0)

    arms = {
        "FULL_weighted": (np.array(A_full), np.array(B_full)),
        "WIRING_binarized": (np.array(A_wire), np.array(B_wire)),
        "NODE_STRENGTH": (np.array(A_str), np.array(B_str)),
    }
    chance = 1.0 / used
    out = {"n_subjects": used, "n_edges": arms["FULL_weighted"][0].shape[1], "density": DENSITY,
           "chance": chance, "arms": {}}
    print(f" usable subjects N={used}  edges={out['n_edges']}  chance={chance:.4f}\n")
    print(f"{'arm':18}{'id-acc':>9}{'margin':>9}{'perm-p':>9}")
    for name, (FA, FB) in arms.items():
        acc, margin, M = ident_acc(FA, FB)
        p = perm_p(M, acc, N_PERM)
        out["arms"][name] = {"id_acc": acc, "margin": margin, "perm_p": p}
        print(f"{name:18}{acc:9.3f}{margin:9.4f}{p:9.4f}")

    full = out["arms"]["FULL_weighted"]["id_acc"]
    wire = out["arms"]["WIRING_binarized"]["id_acc"]
    out["weighted_minus_binarized"] = full - wire
    out["thesis_supported"] = bool(full > wire + 0.1 and full > 5 * chance)
    out["kill_topology_sufficient"] = bool(wire >= full - 0.05)
    out["runtime_s"] = round(time.time() - t0, 2)
    os.makedirs("results", exist_ok=True)
    with open("results/exp05_hcp_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n FULL(weighted) - WIRING(binarized) = {out['weighted_minus_binarized']:+.3f}")
    print(f" thesis_supported (weights >> wiring) = {out['thesis_supported']}")
    print(f" kill_topology_sufficient (wiring ~ weights) = {out['kill_topology_sufficient']}")
    print(f" runtime {out['runtime_s']}s  results -> results/exp05_hcp_results.json")


if __name__ == "__main__":
    main()
