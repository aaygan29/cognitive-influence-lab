"""
Experiment 03b: nonlinear / compensatory degeneracy (the genuine Marder test).
Wiring-is-not-weights program.

Exp01-03a used linear maps where the degeneracy class was a null space. Marder's actual finding is
NONLINEAR compensatory degeneracy: grossly different parameter sets produce identical function through
nonlinear interaction. We test the apparatus against that.

Individuals are small ReLU MLPs: f_i(x) = W2_i @ relu(W1_i @ x). Identity = the function f_i.

Arms (the key contrast is weight-space change vs identity outcome):
  FULL          : (W1_i, W2_i).
  PERM_SCALE    : apply EXACT ReLU symmetries -> permute hidden units AND positively rescale them
                  (W1 row *c_h, W2 col /c_h). Function is mathematically identical; weights are grossly
                  different (near-zero weight correlation). This is the nonlinear degeneracy class.
  FUNC_PERTURB  : add weight noise MATCHED IN MAGNITUDE to the PERM_SCALE weight change, but that changes
                  the function. Same size of weight edit, opposite identity outcome.
  WIRING        : the shared population function (no individual deviation).

Prediction: FULL ~ PERM_SCALE (identity preserved) >> FUNC_PERTURB ~ WIRING ~ chance, EVEN THOUGH
PERM_SCALE has the largest weight-space change. That is "identity is the functional-equivalence class of
the weights, not the weight values," in a nonlinear, compensatory setting.

Replication across R cohorts x S seeds; null (label-shuffle) control.
"""
import json, time
import numpy as np
from exp01_apparatus_validation import add_noise, boot_ci, cohens_d

ARMS = ["FULL", "PERM_SCALE", "FUNC_PERTURB", "FUNC_RAND", "WIRING"]
# FUNC_PERTURB = additive weight noise matched in magnitude to PERM_SCALE (partial functional change).
# FUNC_RAND   = individual function replaced by a fresh random non-existent person (full functional change).


def relu(z):
    return np.maximum(z, 0.0)


def make_world(rng, cfg):
    d, H, p, N = cfg["d"], cfg["H"], cfg["p"], cfg["N"]
    W1_pop = rng.standard_normal((H, d)) / np.sqrt(d)
    W2_pop = rng.standard_normal((p, H)) / np.sqrt(H)
    dl = cfg["indiv_scale"]
    W1 = W1_pop[None] + dl * rng.standard_normal((N, H, d)) / np.sqrt(d)
    W2 = W2_pop[None] + dl * rng.standard_normal((N, p, H)) / np.sqrt(H)
    return W1_pop, W2_pop, W1, W2


def forward(W1, W2, X):
    return (W2 @ relu(W1 @ X.T)).T  # X:(n,d) -> (n,p)


def perm_scale(W1_i, W2_i, rng):
    """Exact ReLU-MLP symmetry: hidden-unit permutation + positive per-unit scaling. Function invariant."""
    H = W1_i.shape[0]
    P = rng.permutation(H)
    c = rng.uniform(0.5, 2.0, size=H)
    W1p = (W1_i[P, :].T * c[P]).T          # permute rows, scale each hidden row by c
    W2p = W2_i[:, P] / c[P][None, :]       # permute cols, inverse-scale
    return W1p, W2p


def rel_wchange(W1a, W2a, W1b, W2b):
    num = np.sqrt(((W1a - W1b) ** 2).sum() + ((W2a - W2b) ** 2).sum())
    den = np.sqrt((W1b ** 2).sum() + (W2b ** 2).sum()) + 1e-12
    return float(num / den)


def id_acc(Y_true, pred, rng, shuffle=False):
    N = Y_true.shape[0]
    Yt = Y_true.reshape(N, -1); Pp = pred.reshape(N, -1)
    Yt = Yt - Yt.mean(1, keepdims=True); Pp = Pp - Pp.mean(1, keepdims=True)
    Yt /= (np.linalg.norm(Yt, axis=1, keepdims=True) + 1e-12)
    Pp /= (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
    match = (Yt @ Pp.T).argmax(1)
    target = rng.permutation(N) if shuffle else np.arange(N)
    return float(np.mean(match == target))


def run_cohort(cohort_seed, cfg):
    rng = np.random.default_rng(cohort_seed)
    N, d, H, p = cfg["N"], cfg["d"], cfg["H"], cfg["p"]
    W1_pop, W2_pop, W1, W2 = make_world(rng, cfg)

    acc = {a: [] for a in ARMS + ["NULL_FULL"]}
    wch = {a: [] for a in ["PERM_SCALE", "FUNC_PERTURB"]}
    for s in range(cfg["S"]):
        srng = np.random.default_rng(cohort_seed * 1000 + s + 1)
        X = srng.standard_normal((cfg["n_test"], d))
        Y_true = np.stack([add_noise(srng, forward(W1[i], W2[i], X), cfg["snr_db"]) for i in range(N)])

        pred = {a: np.zeros((N, cfg["n_test"], p)) for a in ARMS}
        for i in range(N):
            # FULL
            pred["FULL"][i] = forward(W1[i], W2[i], X)
            # PERM_SCALE (exact symmetry)
            W1ps, W2ps = perm_scale(W1[i], W2[i], srng)
            pred["PERM_SCALE"][i] = forward(W1ps, W2ps, X)
            wch["PERM_SCALE"].append(rel_wchange(W1ps, W2ps, W1[i], W2[i]))
            # FUNC_PERTURB matched in weight-magnitude to the perm_scale change
            mag = wch["PERM_SCALE"][-1]
            base = np.sqrt((W1[i] ** 2).sum() + (W2[i] ** 2).sum())
            n1 = srng.standard_normal(W1[i].shape); n2 = srng.standard_normal(W2[i].shape)
            nn = np.sqrt((n1 ** 2).sum() + (n2 ** 2).sum())
            scale = mag * base / (nn + 1e-12)
            W1fp = W1[i] + scale * n1; W2fp = W2[i] + scale * n2
            pred["FUNC_PERTURB"][i] = forward(W1fp, W2fp, X)
            wch["FUNC_PERTURB"].append(rel_wchange(W1fp, W2fp, W1[i], W2[i]))
            # FUNC_RAND: replace individual function with a fresh random non-existent person
            dl = cfg["indiv_scale"]
            W1r = W1_pop + dl * srng.standard_normal(W1_pop.shape) / np.sqrt(cfg["d"])
            W2r = W2_pop + dl * srng.standard_normal(W2_pop.shape) / np.sqrt(cfg["H"])
            pred["FUNC_RAND"][i] = forward(W1r, W2r, X)
            # WIRING (group function)
            pred["WIRING"][i] = forward(W1_pop, W2_pop, X)

        for a in ARMS:
            acc[a].append(id_acc(Y_true, pred[a], srng))
        acc["NULL_FULL"].append(id_acc(Y_true, pred["FULL"], srng, shuffle=True))

    out = {a: float(np.mean(acc[a])) for a in ARMS + ["NULL_FULL"]}
    out["wchange_PERM_SCALE"] = float(np.mean(wch["PERM_SCALE"]))
    out["wchange_FUNC_PERTURB"] = float(np.mean(wch["FUNC_PERTURB"]))
    return out


def main():
    cfg = dict(N=50, d=32, H=24, p=8, n_test=64, snr_db=6.0, indiv_scale=0.6,
               R=15, S=8, master_seed=20260619)
    chance = 1.0 / cfg["N"]
    chance_hi = chance + 4 * np.sqrt(chance * (1 - chance) / (cfg["N"] * cfg["S"]))
    t0 = time.time()
    rng = np.random.default_rng(cfg["master_seed"])
    seeds = rng.integers(1, 2**31 - 1, size=cfg["R"]).tolist()

    per = {k: [] for k in ARMS + ["NULL_FULL", "wchange_PERM_SCALE", "wchange_FUNC_PERTURB"]}
    ordering_pass = 0
    for cs in seeds:
        r = run_cohort(int(cs), cfg)
        for k in per:
            per[k].append(r[k])
        # correct predicted ordering: FULL ~ PERM_SCALE  >>  FUNC_PERTURB (degraded)  >  FUNC_RAND ~ WIRING ~ chance
        ok = (r["FULL"] > 0.5 and abs(r["FULL"] - r["PERM_SCALE"]) < 0.03
              and r["PERM_SCALE"] - r["FUNC_PERTURB"] > 0.2
              and r["FUNC_RAND"] <= chance_hi and r["WIRING"] <= chance_hi)
        ordering_pass += int(ok)

    summary = {"config": cfg, "chance": chance, "chance_band_hi": chance_hi,
               "ordering_replication_rate": ordering_pass / cfg["R"], "runtime_s": round(time.time() - t0, 2)}
    for a in ARMS + ["NULL_FULL"]:
        m, lo, hi = boot_ci(per[a], n=4000, seed=1)
        summary[a] = {"mean": m, "ci95": [lo, hi]}
    summary["wchange_PERM_SCALE"] = float(np.mean(per["wchange_PERM_SCALE"]))
    summary["wchange_FUNC_PERTURB"] = float(np.mean(per["wchange_FUNC_PERTURB"]))
    summary["FULL_vs_PERMSCALE_absdiff"] = abs(summary["FULL"]["mean"] - summary["PERM_SCALE"]["mean"])
    summary["PERMSCALE_vs_FUNCPERTURB_d"] = cohens_d(per["PERM_SCALE"], per["FUNC_PERTURB"])
    checks = {
        "full_beats_chance": summary["FULL"]["ci95"][0] > 0.5,
        "permscale_preserves_identity": summary["FULL_vs_PERMSCALE_absdiff"] < 0.03,
        "permscale_gg_funcperturb": summary["PERMSCALE_vs_FUNCPERTURB_d"] > 2.0,
        "funcrand_at_chance": summary["FUNC_RAND"]["mean"] <= chance_hi,
        "wiring_at_chance": summary["WIRING"]["mean"] <= chance_hi,
        "null_at_chance": summary["NULL_FULL"]["mean"] <= chance_hi,
        "permscale_weightchange_large": summary["wchange_PERM_SCALE"] > 0.5,
        "replicates_90pct": summary["ordering_replication_rate"] >= 0.9,
    }
    summary["checks"] = {k: bool(v) for k, v in checks.items()}
    summary["NONLINEAR_DEGENERACY_CONFIRMED"] = bool(all(checks.values()))
    with open("results/exp03b_nonlinear_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Exp03b nonlinear/compensatory degeneracy  (N={cfg['N']}, chance={chance:.3f}, "
          f"R={cfg['R']}xS={cfg['S']}, runtime {summary['runtime_s']}s) ===")
    for a in ARMS + ["NULL_FULL"]:
        print(f"  {a:13s} id-acc = {summary[a]['mean']:.3f}  CI95 [{summary[a]['ci95'][0]:.3f}, {summary[a]['ci95'][1]:.3f}]")
    print(f"\n  weight-space change ||dW||/||W||:  PERM_SCALE = {summary['wchange_PERM_SCALE']:.2f}   "
          f"FUNC_PERTURB = {summary['wchange_FUNC_PERTURB']:.2f}  (matched)")
    print(f"  FULL vs PERM_SCALE |id diff| = {summary['FULL_vs_PERMSCALE_absdiff']:.4f}  "
          f"(identity preserved despite huge weight change)")
    print(f"  PERM_SCALE vs FUNC_PERTURB Cohen's d = {summary['PERMSCALE_vs_FUNCPERTURB_d']:.1f}")
    print(f"  ordering replication = {summary['ordering_replication_rate']:.2f}")
    print("  checks: " + "  ".join(f"{'OK' if v else 'X'}:{k}" for k, v in summary["checks"].items()))
    print(f"  ==> NONLINEAR_DEGENERACY_CONFIRMED = {summary['NONLINEAR_DEGENERACY_CONFIRMED']}")
    print(f"  results -> results/exp03b_nonlinear_results.json")


if __name__ == "__main__":
    main()
