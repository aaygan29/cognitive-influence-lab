"""
Experiment 01: Apparatus validation on synthetic ground truth.
Wiring-is-not-weights program. See PREREG_exp01_apparatus.md (pre-committed).

Validates that the four-arm ablation + reconstruction identity metric recovers a planted ground truth
("identity is in the individual weight-function, up to degeneracy") and reports its absence under a null,
robustly across R independent random cohorts. No real data. No em dashes in comments.
"""
import json
import time
import numpy as np

# ----------------------------- config (pre-registered) -----------------------------
CFG = dict(
    N=40,            # subjects per cohort  -> chance = 1/N = 0.025
    d=64,            # stimulus dim
    k=40,            # functional (excited) stimulus dims; remaining d-k are the NULL/degeneracy subspace
    p=128,           # response dim
    n_train=400,     # training stimuli (to fit twins)
    n_test=200,      # held-out test stimuli (shared across subjects, NSD-style)
    snr_db=6.0,      # observation SNR (identical process across arms)
    pop_scale=1.0,   # population map magnitude
    indiv_scale=0.7, # individual weight-function (B_i) magnitude = the identity signal
    ridge=1.0,       # ridge penalty for twin fitting
    R=20,            # independent random cohorts (replication across random datasets)
    S=10,            # seeds (noise + test redraws) per cohort
    master_seed=20260619,
)
ARMS = ["FULL", "WITHIN", "OUTSIDE", "WIRING"]


def make_world(rng, cfg):
    """Generate one cohort's ground truth: shared population map + individual weight-functions."""
    N, d, k, p = cfg["N"], cfg["d"], cfg["k"], cfg["p"]
    A_pop = np.zeros((p, d))
    A_pop[:, :k] = cfg["pop_scale"] * rng.standard_normal((p, k)) / np.sqrt(k)
    # individual functional deviations B_i (the planted identity); only on functional cols
    B = np.zeros((N, p, d))
    B[:, :, :k] = cfg["indiv_scale"] * rng.standard_normal((N, p, k)) / np.sqrt(k)
    A_true = A_pop[None, :, :] + B  # (N, p, d); null cols (>=k) are zero -> never excited anyway
    return A_pop, A_true


def make_stimuli(rng, n, cfg):
    """Stimuli excite only the first k dims; dims [k:] are the null/degeneracy subspace (zero variance)."""
    d, k = cfg["d"], cfg["k"]
    X = np.zeros((n, d))
    X[:, :k] = rng.standard_normal((n, k))
    return X  # (n, d)


def add_noise(rng, Y, snr_db):
    sig_p = np.mean(Y ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    return Y + rng.standard_normal(Y.shape) * np.sqrt(noise_p)


def fit_twin(X, Y, ridge):
    """Ridge regression Y ~ X. Returns A_hat (p,d). Null dims have ~0 variance -> fit ~0 there."""
    # X:(n,d) Y:(n,p)  solve (X'X + ridge I) W = X'Y ; A_hat = W'
    d = X.shape[1]
    G = X.T @ X + ridge * np.eye(d)
    W = np.linalg.solve(G, X.T @ Y)   # (d, p)
    return W.T                        # (p, d)


def build_arms(A_hat_i, A_hat_pop, B_mismatch_func, rng, cfg):
    """Construct the four capacity-matched arm matrices for subject i. All are (p,d)."""
    d, k = cfg["d"], cfg["k"]
    full = A_hat_i.copy()

    # WITHIN: redraw weights in the NULL subspace (cols >= k). Predictions on test stimuli unchanged.
    within = A_hat_i.copy()
    within[:, k:] = rng.standard_normal((cfg["p"], d - k)) * 2.0  # large junk in degeneracy directions

    # OUTSIDE: replace functional cols with a mismatched individual signal (wrong person), magnitude matched.
    outside = A_hat_i.copy()
    outside[:, :k] = A_hat_pop[:, :k] + B_mismatch_func  # group + wrong-person deviation

    # WIRING: group-fit map, identical across subjects (shared architecture, no individual weight-function).
    wiring = A_hat_pop.copy()
    return dict(FULL=full, WITHIN=within, OUTSIDE=outside, WIRING=wiring)


def run_cohort(cohort_seed, cfg):
    rng = np.random.default_rng(cohort_seed)
    A_pop, A_true = make_world(rng, cfg)
    N, p, k = cfg["N"], cfg["p"], cfg["k"]

    # training data per subject -> fit individual twins; pooled -> group twin
    X_train = make_stimuli(rng, cfg["n_train"], cfg)
    A_hat = np.zeros_like(A_true)
    Y_train_all = []
    for i in range(N):
        Y_tr = (A_true[i] @ X_train.T).T            # (n_train, p)
        Y_tr = add_noise(rng, Y_tr, cfg["snr_db"])
        A_hat[i] = fit_twin(X_train, Y_tr, cfg["ridge"])
        Y_train_all.append(Y_tr)
    # group twin = ridge fit on pooled data (shared architecture, no individual signal)
    X_pool = np.tile(X_train, (N, 1))
    Y_pool = np.vstack(Y_train_all)
    A_hat_pop = fit_twin(X_pool, Y_pool, cfg["ridge"])

    seed_rows = []
    for s in range(cfg["S"]):
        srng = np.random.default_rng(cohort_seed * 1000 + s + 1)
        X_test = make_stimuli(srng, cfg["n_test"], cfg)        # shared across subjects
        # true held-out responses (from TRUE maps + fresh noise); same noise process for all arms
        Y_true = np.stack([add_noise(srng, (A_true[i] @ X_test.T).T, cfg["snr_db"]) for i in range(N)])

        # build arm twins for every subject
        arm_pred = {a: np.zeros((N, cfg["n_test"], p)) for a in ARMS}
        for i in range(N):
            j_mis = (i + 1) % N
            B_mis = A_hat[j_mis][:, :k] - A_hat_pop[:, :k]      # wrong-person functional deviation
            arms = build_arms(A_hat[i], A_hat_pop, B_mis, srng, cfg)
            for a in ARMS:
                arm_pred[a][i] = (arms[a] @ X_test.T).T

        # identification: correlate each subject's true responses with every twin's predictions
        def id_acc(pred, shuffle=False):
            # pred: (N, n_test, p). Build corr matrix C[i, j] = corr(Y_true[i], pred[j]) flattened.
            Yt = Y_true.reshape(N, -1)
            Pp = pred.reshape(N, -1)
            Yt = (Yt - Yt.mean(1, keepdims=True))
            Pp = (Pp - Pp.mean(1, keepdims=True))
            Yt /= (np.linalg.norm(Yt, axis=1, keepdims=True) + 1e-12)
            Pp /= (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
            C = Yt @ Pp.T                       # (N, N) ; C[i,j] match of true i to twin j
            match = C.argmax(1)
            target = np.arange(N)
            if shuffle:
                target = srng.permutation(N)    # NULL control: break true-twin correspondence
            return float(np.mean(match == target))

        row = {a: id_acc(arm_pred[a]) for a in ARMS}
        row["NULL_FULL"] = id_acc(arm_pred["FULL"], shuffle=True)
        seed_rows.append(row)

    # capacity audit: equal parameter count across arms
    pcounts = {a: int(build_arms(A_hat[0], A_hat_pop,
                                 A_hat[1][:, :k] - A_hat_pop[:, :k],
                                 np.random.default_rng(0), cfg)[a].size) for a in ARMS}
    return seed_rows, pcounts


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nx, ny = len(a), len(b)
    sp = np.sqrt(((nx - 1) * a.var(ddof=1) + (ny - 1) * b.var(ddof=1)) / (nx + ny - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def main():
    cfg = CFG
    t0 = time.time()
    rng = np.random.default_rng(cfg["master_seed"])
    cohort_seeds = rng.integers(1, 2**31 - 1, size=cfg["R"]).tolist()
    chance = 1.0 / cfg["N"]

    per_cohort = {a: [] for a in ARMS + ["NULL_FULL"]}
    ordering_pass = 0
    pcounts_ref = None
    for r, cs in enumerate(cohort_seeds):
        rows, pcounts = run_cohort(int(cs), cfg)
        pcounts_ref = pcounts
        means = {a: float(np.mean([row[a] for row in rows])) for a in ARMS + ["NULL_FULL"]}
        for a in means:
            per_cohort[a].append(means[a])
        # chance band: generous upper bound for "at chance" given N and S
        chance_hi = chance + 4 * np.sqrt(chance * (1 - chance) / (cfg["N"] * cfg["S"]))
        ok = (means["FULL"] > 0.5 and means["WITHIN"] > 0.5
              and means["OUTSIDE"] <= chance_hi and means["WIRING"] <= chance_hi)
        ordering_pass += int(ok)

    summary = {"config": cfg, "chance": chance, "param_counts": pcounts_ref,
               "n_cohorts": cfg["R"], "ordering_replication_rate": ordering_pass / cfg["R"]}
    for a in ARMS + ["NULL_FULL"]:
        m, lo, hi = boot_ci(per_cohort[a], seed=1)
        summary[a] = {"mean": m, "ci95": [lo, hi]}
    summary["primary_contrast_FULL_vs_OUTSIDE"] = {
        "cohens_d": cohens_d(per_cohort["FULL"], per_cohort["OUTSIDE"]),
        "mean_diff": float(np.mean(per_cohort["FULL"]) - np.mean(per_cohort["OUTSIDE"]))}
    summary["within_vs_full_abs_diff"] = float(abs(np.mean(per_cohort["WITHIN"]) - np.mean(per_cohort["FULL"])))
    summary["runtime_s"] = round(time.time() - t0, 2)

    # pre-registered verdict
    chance_hi = chance + 4 * np.sqrt(chance * (1 - chance) / (cfg["N"] * cfg["S"]))
    checks = {
        "null_at_chance": summary["NULL_FULL"]["ci95"][0] <= chance_hi and summary["NULL_FULL"]["mean"] < 0.2,
        "full_beats_chance": summary["FULL"]["ci95"][0] > 0.8,
        "within_approx_full": summary["within_vs_full_abs_diff"] < 0.05,
        "outside_at_chance": summary["OUTSIDE"]["mean"] <= chance_hi,
        "wiring_at_chance": summary["WIRING"]["mean"] <= chance_hi,
        "replicates_90pct": summary["ordering_replication_rate"] >= 0.9,
        "equal_param_count": len(set(pcounts_ref.values())) == 1,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    summary["checks"] = checks
    summary["APPARATUS_VALID"] = bool(all(checks.values()))

    out = "results/exp01_apparatus_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # console report
    print(f"=== Exp01 apparatus validation  (N={cfg['N']}, chance={chance:.3f}, "
          f"R={cfg['R']} cohorts x S={cfg['S']} seeds, runtime {summary['runtime_s']}s) ===")
    print(f"param count per arm (capacity match): {pcounts_ref}")
    for a in ARMS + ["NULL_FULL"]:
        print(f"  {a:10s} id-acc = {summary[a]['mean']:.3f}  CI95 [{summary[a]['ci95'][0]:.3f}, {summary[a]['ci95'][1]:.3f}]")
    print(f"  PRIMARY FULL vs OUTSIDE: Cohen's d = {summary['primary_contrast_FULL_vs_OUTSIDE']['cohens_d']:.2f}  "
          f"(mean diff {summary['primary_contrast_FULL_vs_OUTSIDE']['mean_diff']:.3f})")
    print(f"  WITHIN vs FULL |diff| = {summary['within_vs_full_abs_diff']:.4f}")
    print(f"  ordering replication rate = {summary['ordering_replication_rate']:.2f}")
    print("  checks:")
    for kk, vv in checks.items():
        print(f"     {'PASS' if vv else 'FAIL'}  {kk}")
    print(f"  ==> APPARATUS_VALID = {summary['APPARATUS_VALID']}")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
