"""
Experiment 03a: the graded identity-sufficiency curve, in a SUB-CEILING regime.
Wiring-is-not-weights program.

Exp01/02 validated the apparatus but FULL sat at ceiling (1.000), so there was no curve. Here we make the
world hard and BISECT a continuous difficulty knob (individual-signal scale) so FULL lands in [0.45, 0.85]
and the curve has dynamic range. Then:

PRIMARY: alpha in [0,1] = fraction of the INDIVIDUAL functional weight-function retained.
    twin_func = A_hat_pop_func + alpha * (A_hat_i_func - A_hat_pop_func)
    alpha=0 -> WIRING (group, no individual weights);  alpha=1 -> FULL.
  This is the deliverable: identity accuracy vs how much individual weight-information you keep. alpha* (the
  smallest alpha reaching halfway from chance to FULL) is the minimal sufficient fraction the .identity
  format must store.

CONTROLS (two degeneracy arms, both keep the full functional weights):
  beta_null   : perturb EXACT null-space dims (>=k), the exact degeneracy class -> identity must be FLAT.
  beta_lowvar : perturb lowest-variance FUNCTIONAL dims (approximate degeneracy) -> gentle decline.
  The contrast (alpha steep; beta_null flat) is the precise statement "identity is the functional-
  equivalence class of the weights, not the wiring and not the exact weights."

Replication across R random cohorts x S seeds; null (label-shuffle) control included.
"""
import json, time
import numpy as np
from exp01_apparatus_validation import make_world, add_noise, fit_twin, boot_ci

ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  # fraction of individuating dims retained
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def aniso_stimuli(rng, n, d, k, eig):
    X = np.zeros((n, d))
    X[:, :k] = rng.standard_normal((n, k)) * np.sqrt(eig)[None, :]
    return X


def id_acc(Y_true, pred, rng, shuffle=False):
    N = Y_true.shape[0]
    Yt = Y_true.reshape(N, -1); Pp = pred.reshape(N, -1)
    Yt = Yt - Yt.mean(1, keepdims=True); Pp = Pp - Pp.mean(1, keepdims=True)
    Yt /= (np.linalg.norm(Yt, axis=1, keepdims=True) + 1e-12)
    Pp /= (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
    match = (Yt @ Pp.T).argmax(1)
    target = rng.permutation(N) if shuffle else np.arange(N)
    return float(np.mean(match == target))


def _fit_cohort(cohort_seed, cfg):
    rng = np.random.default_rng(cohort_seed)
    N, d, k = cfg["N"], cfg["d"], cfg["k"]
    eig = np.linspace(1.0, 0.02, k)
    A_pop, A_true = make_world(rng, cfg)
    X_train = aniso_stimuli(rng, cfg["n_train"], d, k, eig)
    A_hat = np.zeros_like(A_true)
    Ytr = []
    for i in range(N):
        Y = add_noise(rng, (A_true[i] @ X_train.T).T, cfg["snr_db"])
        A_hat[i] = fit_twin(X_train, Y, cfg["ridge"]); Ytr.append(Y)
    A_hat_pop = fit_twin(np.tile(X_train, (N, 1)), np.vstack(Ytr), cfg["ridge"])
    return rng, eig, A_true, A_hat, A_hat_pop


def full_only(cohort_seed, cfg):
    _, eig, A_true, A_hat, _ = _fit_cohort(cohort_seed, cfg)
    N, d, k, p = cfg["N"], cfg["d"], cfg["k"], cfg["p"]
    accs = []
    for s in range(3):
        srng = np.random.default_rng(cohort_seed * 7 + s + 1)
        X_test = aniso_stimuli(srng, cfg["n_test"], d, k, eig)
        Y_true = np.stack([add_noise(srng, (A_true[i] @ X_test.T).T, cfg["snr_db"]) for i in range(N)])
        pred = np.stack([(A_hat[i] @ X_test.T).T for i in range(N)])
        accs.append(id_acc(Y_true, pred, srng))
    return float(np.mean(accs))


def calibrate_indiv(base, rng, target=0.62, lo=0.05, hi=2.5, iters=9):
    """Bisection on indiv_scale (FULL is monotone increasing in it) to hit the sub-ceiling band."""
    seeds = rng.integers(1, 2**31 - 1, size=3).tolist()
    f_lo = np.mean([full_only(int(s), dict(base, indiv_scale=lo)) for s in seeds])
    f_hi = np.mean([full_only(int(s), dict(base, indiv_scale=hi)) for s in seeds])
    best = (hi, f_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = np.mean([full_only(int(s), dict(base, indiv_scale=mid)) for s in seeds])
        if abs(f - target) < abs(best[1] - target):
            best = (mid, f)
        if f < target:
            lo = mid
        else:
            hi = mid
    return best  # (indiv_scale, FULL_estimate)


def run_cohort(cohort_seed, cfg):
    rng, eig, A_true, A_hat, A_hat_pop = _fit_cohort(cohort_seed, cfg)
    N, d, k, p = cfg["N"], cfg["d"], cfg["k"], cfg["p"]
    low_var = np.argsort(eig)[: max(1, k // 4)]
    dev = A_hat[:, :, :k] - A_hat_pop[None, :, :k]
    # rank individuating functional dims by deviation energy; alpha retains the top fraction at full strength
    order = np.argsort(-(dev ** 2).sum(axis=(0, 1)))

    a_rows, bn_rows, bl_rows, null_rows = [], [], [], []
    for s in range(cfg["S"]):
        srng = np.random.default_rng(cohort_seed * 1000 + s + 1)
        X_test = aniso_stimuli(srng, cfg["n_test"], d, k, eig)
        Y_true = np.stack([add_noise(srng, (A_true[i] @ X_test.T).T, cfg["snr_db"]) for i in range(N)])

        arow = []
        for a in ALPHAS:
            m = int(round(a * k)); keep = order[:m]
            pred = np.zeros((N, cfg["n_test"], p))
            for i in range(N):
                M = A_hat[i].copy(); M[:, :k] = A_hat_pop[:, :k]   # start from group (no individual weights)
                if m > 0:
                    M[:, keep] = A_hat_pop[:, keep] + dev[i][:, keep]  # restore individual on kept dims
                pred[i] = (M @ X_test.T).T
            arow.append(id_acc(Y_true, pred, srng))
        a_rows.append(arow)

        bn, bl = [], []
        for b in BETAS:
            pn = np.zeros((N, cfg["n_test"], p)); pl = np.zeros((N, cfg["n_test"], p))
            en = srng.standard_normal((N, p, d - k)); el = srng.standard_normal((N, p, len(low_var)))
            for i in range(N):
                Mn = A_hat[i].copy(); Mn[:, k:] += b * en[i]; pn[i] = (Mn @ X_test.T).T
                Ml = A_hat[i].copy(); Ml[:, low_var] += b * el[i]; pl[i] = (Ml @ X_test.T).T
            bn.append(id_acc(Y_true, pn, srng)); bl.append(id_acc(Y_true, pl, srng))
        bn_rows.append(bn); bl_rows.append(bl)

        predF = np.stack([(A_hat[i] @ X_test.T).T for i in range(N)])
        null_rows.append(id_acc(Y_true, predF, srng, shuffle=True))

    return (np.array(a_rows).mean(0), np.array(bn_rows).mean(0),
            np.array(bl_rows).mean(0), float(np.mean(null_rows)))


def spark(vals, lo, hi):
    blocks = "_.-=+*#%@"
    return "".join(blocks[min(8, max(0, int((v - lo) / (hi - lo + 1e-12) * 8)))] for v in vals)


def main():
    # signal-limited regime (low noise, weak per-dim individual signal spread over many dims) so the
    # dimension-retention alpha curve grades smoothly rather than saturating at a noise ceiling.
    base = dict(N=50, d=64, k=40, p=8, n_train=400, n_test=16, snr_db=6.0,
                pop_scale=1.0, ridge=2.0, R=15, S=8, master_seed=20260619)
    t0 = time.time()
    cal_rng = np.random.default_rng(base["master_seed"] + 777)
    isc, f_cal = calibrate_indiv(base, cal_rng, target=0.85, lo=0.02, hi=1.0)
    cfg = dict(base, indiv_scale=isc)
    chance = 1.0 / cfg["N"]
    print(f"[calibration] indiv_scale={isc:.3f} -> FULL~{f_cal:.3f}  (chance={chance:.3f})")

    rng = np.random.default_rng(cfg["master_seed"])
    seeds = rng.integers(1, 2**31 - 1, size=cfg["R"]).tolist()
    A, BN, BL, NL = [], [], [], []
    for cs in seeds:
        a, bn, bl, nl = run_cohort(int(cs), cfg)
        A.append(a); BN.append(bn); BL.append(bl); NL.append(nl)
    A, BN, BL = np.array(A), np.array(BN), np.array(BL)

    acurve = [boot_ci(A[:, j], n=4000, seed=1) for j in range(len(ALPHAS))]
    bncurve = [boot_ci(BN[:, j], n=4000, seed=1) for j in range(len(BETAS))]
    blcurve = [boot_ci(BL[:, j], n=4000, seed=1) for j in range(len(BETAS))]
    null_m, null_lo, null_hi = boot_ci(NL, n=4000, seed=1)
    full, wiring = acurve[-1][0], acurve[0][0]
    half = chance + 0.5 * (full - chance)
    alpha_star = next((a for a, (m, _, _) in zip(ALPHAS, acurve) if m >= half), None)

    a_range = full - wiring
    bn_range = abs(bncurve[0][0] - bncurve[-1][0])   # exact-null perturbation effect (should be ~0)
    bl_range = abs(blcurve[0][0] - blcurve[-1][0])
    confirmed = bool(a_range > 0.3 and bn_range < 0.05 and a_range > 4 * bn_range)

    report = dict(config=cfg, chance=chance, full=full, wiring_alpha0=wiring,
                  null={"mean": null_m, "ci95": [null_lo, null_hi]},
                  alpha_star=alpha_star, half_target=half,
                  alpha_curve=[{"alpha": a, "mean": m, "ci95": [lo, hi]} for a, (m, lo, hi) in zip(ALPHAS, acurve)],
                  beta_null_curve=[{"beta": b, "mean": m, "ci95": [lo, hi]} for b, (m, lo, hi) in zip(BETAS, bncurve)],
                  beta_lowvar_curve=[{"beta": b, "mean": m, "ci95": [lo, hi]} for b, (m, lo, hi) in zip(BETAS, blcurve)],
                  alpha_range=a_range, beta_null_range=bn_range, beta_lowvar_range=bl_range,
                  sub_ceiling=bool(0.4 < full < 0.95), functional_equivalence_confirmed=confirmed,
                  runtime_s=round(time.time() - t0, 2))
    with open("results/exp03a_curve_results.json", "w") as f:
        json.dump(report, f, indent=2)

    hi_p = max(full, bncurve[0][0])
    print(f"=== Exp03a identity-sufficiency curve  (N={cfg['N']}, chance={chance:.3f}, "
          f"R={cfg['R']}xS={cfg['S']}, runtime {report['runtime_s']}s) ===")
    print(f"sub-ceiling? FULL={full:.3f}  WIRING(a=0)={wiring:.3f}  NULL={null_m:.3f} -> {report['sub_ceiling']}")
    print(f"\nPRIMARY  identity vs fraction of individual weight-function retained (alpha):")
    print(f"  alpha : " + " ".join(f"{a:4.2f}" for a in ALPHAS))
    print(f"  id-acc: " + " ".join(f"{m:4.2f}" for m, _, _ in acurve) + "   " + spark([m for m, _, _ in acurve], chance, hi_p))
    print(f"  minimal sufficient fraction alpha* (>= halfway {half:.2f}) = {alpha_star}")
    print(f"\nCONTROL  beta_null  (EXACT degeneracy / null-space weights; should be FLAT):")
    print(f"  id-acc: " + " ".join(f"{m:4.2f}" for m, _, _ in bncurve) + "   " + spark([m for m, _, _ in bncurve], chance, hi_p))
    print(f"CONTROL  beta_lowvar (approx degeneracy / low-variance functional weights):")
    print(f"  id-acc: " + " ".join(f"{m:4.2f}" for m, _, _ in blcurve) + "   " + spark([m for m, _, _ in blcurve], chance, hi_p))
    print(f"\n  range over alpha (functional weights) = {a_range:.3f}")
    print(f"  range over beta_null (degeneracy class) = {bn_range:.3f}")
    print(f"  ==> identity = functional-equivalence class of weights (steep in alpha, flat in null): "
          f"{'CONFIRMED' if confirmed else 'NOT confirmed'}")
    print(f"  results -> results/exp03a_curve_results.json")


if __name__ == "__main__":
    main()
