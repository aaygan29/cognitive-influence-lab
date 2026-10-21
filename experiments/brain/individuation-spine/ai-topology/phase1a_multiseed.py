"""
Phase 1A: Multi-seed validation.

Tests whether the λ=0.10 small-world topology improvement (4.430→4.312)
is real or single-seed variance.

Design:
  - 2 conditions: baseline (λ=0) and small-world (λ=0.10)
  - 20 seeds per condition (total 40 runs)
  - Primary test: one-tailed Welch t-test (H1: SW < baseline)
  - Reports: mean±SD, Cohen's d, bootstrap 95% CI on difference, p-value
"""

import os, json, time
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_core import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase1a")
LOG_PATH    = os.path.join(RESULTS_DIR, "runs.csv")
N_SEEDS     = 20
SEEDS       = list(range(42, 42 + N_SEEDS))


def run_one(args):
    seed, condition, lam = args
    result = train(
        seed=seed,
        lambda_topo=lam,
        topo_target="small_world" if lam > 0 else "none",
        log_path=LOG_PATH,
        verbose=False,
    )
    print(f"  [{condition}] seed={seed} → val_bpb={result['final_val_bpb']:.4f}")
    return condition, result["final_val_bpb"]


def analyse(baseline_bpbs, sw_bpbs):
    b  = np.array(baseline_bpbs)
    sw = np.array(sw_bpbs)
    diff = b - sw  # positive = SW better

    t_stat, p_two = stats.ttest_ind(b, sw, equal_var=False)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    pooled_sd = np.sqrt((b.std(ddof=1)**2 + sw.std(ddof=1)**2) / 2)
    cohens_d  = (b.mean() - sw.mean()) / pooled_sd

    # Bootstrap 95% CI on mean difference
    rng = np.random.default_rng(0)
    boot_diffs = []
    for _ in range(10_000):
        b_s  = rng.choice(b,  size=len(b),  replace=True)
        sw_s = rng.choice(sw, size=len(sw), replace=True)
        boot_diffs.append(b_s.mean() - sw_s.mean())
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "baseline_mean": float(b.mean()),  "baseline_sd": float(b.std(ddof=1)),
        "sw_mean":       float(sw.mean()), "sw_sd":       float(sw.std(ddof=1)),
        "mean_diff":     float(diff.mean()),
        "ci_95_lo":      float(ci_lo),     "ci_95_hi":    float(ci_hi),
        "cohens_d":      float(cohens_d),
        "t_stat":        float(t_stat),
        "p_one_tailed":  float(p_one),
        "p_two_tailed":  float(p_two),
        "n_seeds":       N_SEEDS,
        "significant":   bool(p_one < 0.05 and cohens_d > 0),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Phase 1A: Multi-seed validation  ({N_SEEDS} seeds × 2 conditions)")
    print(f"Results → {RESULTS_DIR}")
    t0 = time.time()

    tasks = (
        [(s, "baseline", 0.0)    for s in SEEDS] +
        [(s, "small_world", 0.10) for s in SEEDS]
    )

    baseline_bpbs = []
    sw_bpbs       = []

    # Run sequentially on MPS (MPS doesn't support true parallelism across processes)
    for args in tasks:
        cond, bpb = run_one(args)
        if cond == "baseline":
            baseline_bpbs.append(bpb)
        else:
            sw_bpbs.append(bpb)

    summary = analyse(baseline_bpbs, sw_bpbs)
    summary["baseline_all"] = baseline_bpbs
    summary["sw_all"]       = sw_bpbs
    summary["elapsed_s"]    = time.time() - t0

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Phase 1A Results ─────────────────────────────")
    print(f"  Baseline:     {summary['baseline_mean']:.4f} ± {summary['baseline_sd']:.4f}")
    print(f"  Small-world:  {summary['sw_mean']:.4f} ± {summary['sw_sd']:.4f}")
    print(f"  Diff (b-sw):  {summary['mean_diff']:+.4f}  95% CI [{summary['ci_95_lo']:+.4f}, {summary['ci_95_hi']:+.4f}]")
    print(f"  Cohen's d:    {summary['cohens_d']:.3f}")
    print(f"  p (one-tail): {summary['p_one_tailed']:.4f}")
    print(f"  Significant:  {summary['significant']}")
    print(f"  Elapsed:      {summary['elapsed_s']:.0f}s")
    print(f"  Saved → {out_path}")
    return summary


if __name__ == "__main__":
    main()
