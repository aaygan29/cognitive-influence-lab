"""
exp06: seed-robustness of the two synthetic headline results, addressing the review finding that
exp03a/exp03b each used a single master seed. We re-run across 20 independent master seeds and report
the distribution of (a) the interpolated half-of-ceiling crossing alpha* on a refined grid, and
(b) the within-class vs functional degeneracy contrast. Reuses the exact cohort machinery of exp03a/b.
No em dashes.
"""
import importlib.util, os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

a = _load("exp03a", "exp03a_identity_sufficiency_curve.py")
b = _load("exp03b", "exp03b_nonlinear_degeneracy.py")

# refine the alpha grid near the crossing (was [...,0.6,0.8,1.0]); add 0.65/0.7/0.75
a.ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
ALPHAS = a.ALPHAS

def interp_cross(alphas, means, target):
    for (x0, y0), (x1, y1) in zip(zip(alphas, means), list(zip(alphas, means))[1:]):
        if y0 <= target <= y1 and y1 > y0:
            return x0 + (x1 - x0) * (target - y0) / (y1 - y0)
    return float("nan")

N_MASTER = 20
base_a = dict(N=50, d=64, k=40, p=8, n_train=400, n_test=16, snr_db=6.0,
              pop_scale=1.0, ridge=2.0, R=15, S=8)
cal_rng = np.random.default_rng(999)
isc, _ = a.calibrate_indiv(dict(base_a, master_seed=20260619), cal_rng, target=0.85, lo=0.02, hi=1.0)
cfg_a = dict(base_a, indiv_scale=isc)
chance_a = 1.0 / cfg_a["N"]

alpha_stars, ceilings = [], []
for ms in range(N_MASTER):
    rng = np.random.default_rng(100 + ms)
    seeds = rng.integers(1, 2**31 - 1, size=cfg_a["R"]).tolist()
    A = np.array([a.run_cohort(int(cs), cfg_a)[0] for cs in seeds])   # [R, len(ALPHAS)]
    curve = A.mean(0)
    ceiling = curve[-1]
    alpha_stars.append(interp_cross(ALPHAS, curve, chance_a + 0.5 * (ceiling - chance_a)))
    ceilings.append(ceiling)
    print(f"[a seed {ms:2d}] ceiling={ceiling:.3f}  alpha*={alpha_stars[-1]:.3f}")

alpha_stars = np.array(alpha_stars); ceilings = np.array(ceilings)

# ---- exp03b degeneracy across seeds ----
cfg_b = dict(N=50, d=32, H=24, p=8, n_test=64, snr_db=6.0, indiv_scale=0.6, R=15, S=8)
within, func = [], []
for ms in range(N_MASTER):
    rng = np.random.default_rng(500 + ms)
    seeds = rng.integers(1, 2**31 - 1, size=cfg_b["R"]).tolist()
    rows = [b.run_cohort(int(cs), cfg_b) for cs in seeds]
    # run_cohort returns a dict of arm -> mean id acc; confirm keys then aggregate
    ws = np.mean([r["PERM_SCALE"] for r in rows]); fs = np.mean([r["FUNC_PERTURB"] for r in rows])
    within.append(ws); func.append(fs)
    print(f"[b seed {ms:2d}] within-class={ws:.3f}  functional={fs:.3f}")
within = np.array(within); func = np.array(func)

out = {
    "n_master_seeds": N_MASTER,
    "refined_alpha_grid": ALPHAS,
    "alpha_star": {"mean": float(alpha_stars.mean()), "sd": float(alpha_stars.std(ddof=1)),
                   "min": float(alpha_stars.min()), "max": float(alpha_stars.max())},
    "ceiling": {"mean": float(ceilings.mean()), "sd": float(ceilings.std(ddof=1))},
    "degeneracy_within_class": {"mean": float(within.mean()), "sd": float(within.std(ddof=1))},
    "degeneracy_functional": {"mean": float(func.mean()), "sd": float(func.std(ddof=1))},
    "within_minus_functional": {"mean": float((within - func).mean()),
                                "sd": float((within - func).std(ddof=1))},
}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "exp06_seed_robustness.json"), "w"), indent=2)
print("\n=== SUMMARY (20 master seeds) ===")
print(f"alpha* (interpolated, refined grid): {out['alpha_star']['mean']:.3f} +/- {out['alpha_star']['sd']:.3f}"
      f"  [{out['alpha_star']['min']:.3f}, {out['alpha_star']['max']:.3f}]")
print(f"ceiling: {out['ceiling']['mean']:.3f} +/- {out['ceiling']['sd']:.3f}")
print(f"within-class: {out['degeneracy_within_class']['mean']:.3f} +/- {out['degeneracy_within_class']['sd']:.3f}")
print(f"functional:   {out['degeneracy_functional']['mean']:.3f} +/- {out['degeneracy_functional']['sd']:.3f}")
print(f"within - functional gap: {out['within_minus_functional']['mean']:.3f} +/- {out['within_minus_functional']['sd']:.3f}")
print("wrote results/exp06_seed_robustness.json")
