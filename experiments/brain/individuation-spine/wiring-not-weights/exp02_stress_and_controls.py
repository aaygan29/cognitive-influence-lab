"""
Experiment 02: stress sweep + corrected controls. Independent verification of exp01's apparatus.
Wiring-is-not-weights program.

Fixes two ways exp01 was "too easy":
  (1) OUTSIDE now uses a FRESH-RANDOM functional deviation (a non-existent person), so it lands at TRUE
      chance, not the cyclic below-chance artifact.
  (2) Degeneracy is APPROXIMATE, not an exact zero-variance null space: stimuli are anisotropic
      (eigenvalues decay), and WITHIN_APPROX perturbs the lowest-variance functional directions. This is a
      non-trivial, Marder-flavored degeneracy (different weights, near-identical function) rather than a
      tautology.
Then sweeps SNR x individual-signal strength to confirm the predicted ordering holds across regimes, not
just the easy one. Replication across R random cohorts per regime.
"""
import json, time
import numpy as np
from exp01_apparatus_validation import make_world, add_noise, fit_twin, boot_ci, cohens_d

ARMS = ["FULL", "WITHIN_APPROX", "OUTSIDE", "WIRING"]


def aniso_stimuli(rng, n, d, k, eig):
    """Anisotropic stimuli: first k dims with std = sqrt(eig_j) (decaying); dims >=k are exact null."""
    X = np.zeros((n, d))
    X[:, :k] = rng.standard_normal((n, k)) * np.sqrt(eig)[None, :]
    return X


def run_cohort(cohort_seed, cfg):
    rng = np.random.default_rng(cohort_seed)
    N, d, k, p = cfg["N"], cfg["d"], cfg["k"], cfg["p"]
    # decaying eigenvalues across functional dims -> lowest are near-degenerate
    eig = np.linspace(1.0, 0.02, k)
    A_pop, A_true = make_world(rng, cfg)

    X_train = aniso_stimuli(rng, cfg["n_train"], d, k, eig)
    A_hat = np.zeros_like(A_true)
    Y_train_all = []
    for i in range(N):
        Y_tr = add_noise(rng, (A_true[i] @ X_train.T).T, cfg["snr_db"])
        A_hat[i] = fit_twin(X_train, Y_tr, cfg["ridge"])
        Y_train_all.append(Y_tr)
    A_hat_pop = fit_twin(np.tile(X_train, (N, 1)), np.vstack(Y_train_all), cfg["ridge"])

    # index of the lowest-variance functional dims (approximate degeneracy directions)
    low_var = np.argsort(eig)[: max(1, k // 4)]
    rows = []
    for s in range(cfg["S"]):
        srng = np.random.default_rng(cohort_seed * 1000 + s + 1)
        X_test = aniso_stimuli(srng, cfg["n_test"], d, k, eig)
        Y_true = np.stack([add_noise(srng, (A_true[i] @ X_test.T).T, cfg["snr_db"]) for i in range(N)])

        arm_pred = {a: np.zeros((N, cfg["n_test"], p)) for a in ARMS}
        for i in range(N):
            full = A_hat[i].copy()

            within = A_hat[i].copy()  # perturb only the lowest-variance functional dims (approx degeneracy)
            within[:, low_var] += srng.standard_normal((p, len(low_var))) * 1.0

            outside = A_hat[i].copy()  # fresh-random non-existent person, magnitude matched
            outside[:, :k] = A_hat_pop[:, :k] + cfg["indiv_scale"] * srng.standard_normal((p, k)) / np.sqrt(k)

            wiring = A_hat_pop.copy()
            for a, M in zip(ARMS, [full, within, outside, wiring]):
                arm_pred[a][i] = (M @ X_test.T).T

        def id_acc(pred, shuffle=False):
            Yt = Y_true.reshape(N, -1); Pp = pred.reshape(N, -1)
            Yt = Yt - Yt.mean(1, keepdims=True); Pp = Pp - Pp.mean(1, keepdims=True)
            Yt /= (np.linalg.norm(Yt, axis=1, keepdims=True) + 1e-12)
            Pp /= (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
            match = (Yt @ Pp.T).argmax(1)
            target = srng.permutation(N) if shuffle else np.arange(N)
            return float(np.mean(match == target))

        row = {a: id_acc(arm_pred[a]) for a in ARMS}
        row["NULL_FULL"] = id_acc(arm_pred["FULL"], shuffle=True)
        rows.append(row)
    return rows


def main():
    base = dict(N=40, d=64, k=40, p=128, n_train=400, n_test=200, ridge=1.0,
                pop_scale=1.0, R=10, S=8, master_seed=20260619)
    chance = 1.0 / base["N"]
    chance_hi = chance + 4 * np.sqrt(chance * (1 - chance) / (base["N"] * base["S"]))
    regimes = [(snr, isc) for snr in (0.0, 3.0, 6.0) for isc in (0.3, 0.7)]

    t0 = time.time()
    report = {"chance": chance, "chance_band_hi": chance_hi, "regimes": []}
    all_pass = True
    for snr, isc in regimes:
        cfg = dict(base, snr_db=snr, indiv_scale=isc)
        rng = np.random.default_rng(base["master_seed"] + int(snr * 10) + int(isc * 100))
        cohort_seeds = rng.integers(1, 2**31 - 1, size=base["R"]).tolist()
        per = {a: [] for a in ARMS + ["NULL_FULL"]}
        ordering_pass = 0
        for cs in cohort_seeds:
            rows = run_cohort(int(cs), cfg)
            m = {a: float(np.mean([r[a] for r in rows])) for a in ARMS + ["NULL_FULL"]}
            for a in m: per[a].append(m[a])
            ok = (m["FULL"] > 0.5 and m["WITHIN_APPROX"] > 0.5
                  and m["OUTSIDE"] <= chance_hi and m["WIRING"] <= chance_hi)
            ordering_pass += int(ok)
        reg = {"snr_db": snr, "indiv_scale": isc,
               "ordering_replication_rate": ordering_pass / base["R"],
               "FULL_vs_OUTSIDE_d": cohens_d(per["FULL"], per["OUTSIDE"]),
               "within_vs_full_absdiff": abs(np.mean(per["WITHIN_APPROX"]) - np.mean(per["FULL"]))}
        for a in ARMS + ["NULL_FULL"]:
            mm, lo, hi = boot_ci(per[a], n=4000, seed=1)
            reg[a] = {"mean": round(mm, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        report["regimes"].append(reg)
        # a regime "passes" if ordering replicates and null/outside/wiring at chance
        reg_pass = (reg["ordering_replication_rate"] >= 0.9
                    and reg["OUTSIDE"]["mean"] <= chance_hi
                    and reg["WIRING"]["mean"] <= chance_hi
                    and reg["NULL_FULL"]["mean"] <= chance_hi)
        reg["regime_pass"] = bool(reg_pass)
        all_pass = all_pass and reg_pass

    report["all_regimes_pass"] = bool(all_pass)
    report["runtime_s"] = round(time.time() - t0, 2)
    with open("results/exp02_stress_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== Exp02 stress + corrected controls  (chance={chance:.3f}, band_hi={chance_hi:.3f}, "
          f"R={base['R']}xS={base['S']}, runtime {report['runtime_s']}s) ===")
    print(f"{'snr':>4} {'isc':>4} | {'FULL':>6} {'WITHINa':>8} {'OUTSIDE':>8} {'WIRING':>7} {'NULL':>6} | "
          f"{'d(F-O)':>7} {'|W-F|':>6} {'rep':>4}  pass")
    for r in report["regimes"]:
        print(f"{r['snr_db']:>4.0f} {r['indiv_scale']:>4.1f} | "
              f"{r['FULL']['mean']:>6.3f} {r['WITHIN_APPROX']['mean']:>8.3f} {r['OUTSIDE']['mean']:>8.3f} "
              f"{r['WIRING']['mean']:>7.3f} {r['NULL_FULL']['mean']:>6.3f} | "
              f"{r['FULL_vs_OUTSIDE_d']:>7.1f} {r['within_vs_full_absdiff']:>6.3f} "
              f"{r['ordering_replication_rate']:>4.2f}  {'PASS' if r['regime_pass'] else 'FAIL'}")
    print(f"  ==> ALL REGIMES PASS = {report['all_regimes_pass']}   results -> results/exp02_stress_results.json")


if __name__ == "__main__":
    main()
