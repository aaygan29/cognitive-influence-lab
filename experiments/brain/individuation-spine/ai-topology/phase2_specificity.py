"""
Phase 2: Topology Specificity Ablation — the crux experiment.

Five conditions at matched λ=0.10, 20 seeds each:
  - baseline    (λ=0, no regularization)
  - small_world (λ=0.10, brain-like target)
  - random_graph (λ=0.10, maximize entropy → random connectivity)
  - scale_free  (λ=0.10, hub-and-spoke degree distribution)
  - lattice     (λ=0.10, locality only → high C, high L)
  - degenerate  (top-k=2, mimics induction-head topology)

Primary question: does small_world specifically outperform the others,
or is any regularization equally good?

Statistics:
  - One-way ANOVA across all conditions
  - Pairwise Welch t-tests with Holm-Bonferroni correction
  - Critical comparisons: SW vs baseline, SW vs random_graph, SW vs scale_free
"""

import os, sys, json, time, itertools
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from train_core import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase2")
LOG_PATH    = os.path.join(RESULTS_DIR, "runs.csv")
N_SEEDS     = 20
SEEDS       = list(range(42, 42 + N_SEEDS))

CONDITIONS = [
    {"name": "baseline",     "lambda_topo": 0.0,  "topo_target": "none",         "attn_top_k": 0},
    {"name": "small_world",  "lambda_topo": 0.10, "topo_target": "small_world",  "attn_top_k": 0},
    {"name": "random_graph", "lambda_topo": 0.10, "topo_target": "random_graph", "attn_top_k": 0},
    {"name": "scale_free",   "lambda_topo": 0.10, "topo_target": "scale_free",   "attn_top_k": 0},
    {"name": "lattice",      "lambda_topo": 0.10, "topo_target": "lattice",      "attn_top_k": 0},
    {"name": "degenerate",   "lambda_topo": 0.0,  "topo_target": "none",         "attn_top_k": 2},
]


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction. Returns corrected p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    prev_p = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        cp = p * (n - rank)
        cp = max(cp, prev_p)  # monotonicity
        corrected[orig_idx] = min(cp, 1.0)
        prev_p = corrected[orig_idx]
    return corrected


def bootstrap_ci(a, b, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        da = rng.choice(a, len(a), replace=True)
        db = rng.choice(b, len(b), replace=True)
        diffs.append(da.mean() - db.mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_one(cond_dict, seed):
    result = train(
        seed=seed,
        lambda_topo=cond_dict["lambda_topo"],
        topo_target=cond_dict["topo_target"],
        attn_top_k=cond_dict["attn_top_k"],
        log_path=LOG_PATH,
        verbose=False,
    )
    bpb = result["final_val_bpb"]
    print(f"  [{cond_dict['name']:12s}] seed={seed} → val_bpb={bpb:.4f}")
    return bpb


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Phase 2: Topology Specificity Ablation ({len(CONDITIONS)} conditions × {N_SEEDS} seeds)")
    t0 = time.time()

    # Collect results per condition
    cond_bpbs = {c["name"]: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        print(f"\n  === Condition: {cond['name']} ===")
        for seed in SEEDS:
            bpb = run_one(cond, seed)
            cond_bpbs[cond["name"]].append(bpb)

    # ── One-way ANOVA ─────────────────────────────────────────────────────────
    groups = [np.array(cond_bpbs[c["name"]]) for c in CONDITIONS]
    f_stat, p_anova = stats.f_oneway(*groups)

    # ── Pairwise Welch t-tests ────────────────────────────────────────────────
    cond_names = [c["name"] for c in CONDITIONS]
    pairs = list(itertools.combinations(range(len(cond_names)), 2))
    p_raw = []
    pair_labels = []
    for i, j in pairs:
        _, p = stats.ttest_ind(groups[i], groups[j], equal_var=False)
        p_raw.append(p)
        pair_labels.append(f"{cond_names[i]}_vs_{cond_names[j]}")

    p_corrected = holm_bonferroni(p_raw)

    pairwise = []
    for k, (i, j) in enumerate(pairs):
        a, b_ = groups[i], groups[j]
        pooled_sd = np.sqrt((a.std(ddof=1)**2 + b_.std(ddof=1)**2) / 2)
        d = (a.mean() - b_.mean()) / (pooled_sd + 1e-9)
        ci_lo, ci_hi = bootstrap_ci(a, b_)
        pairwise.append({
            "pair":       pair_labels[k],
            "mean_a":     float(a.mean()), "mean_b": float(b_.mean()),
            "diff_a_minus_b": float(a.mean() - b_.mean()),
            "cohens_d":   float(d),
            "ci_95_lo":   ci_lo, "ci_95_hi": ci_hi,
            "p_raw":      float(p_raw[k]),
            "p_corrected": float(p_corrected[k]),
            "significant": bool(p_corrected[k] < 0.05),
        })

    # ── Critical comparisons summary ─────────────────────────────────────────
    critical = ["small_world_vs_baseline", "small_world_vs_random_graph",
                "small_world_vs_scale_free", "small_world_vs_lattice"]
    # Note: pairs are (lower_idx, higher_idx) — need to find SW correctly
    sw_vs = {}
    for p in pairwise:
        if "small_world" in p["pair"]:
            key = p["pair"]
            sw_bpb = cond_bpbs["small_world"]
            # Sign: positive diff means first condition is worse (higher bpb)
            # We want to know if SW is better (lower bpb)
            sw_vs[key] = p

    # ── Condition summaries ───────────────────────────────────────────────────
    cond_summary = {}
    for c in CONDITIONS:
        g = np.array(cond_bpbs[c["name"]])
        cond_summary[c["name"]] = {
            "mean":   float(g.mean()),
            "sd":     float(g.std(ddof=1)),
            "median": float(np.median(g)),
            "min":    float(g.min()),
            "max":    float(g.max()),
            "all":    g.tolist(),
        }

    # ── Verdict ───────────────────────────────────────────────────────────────
    sw_beats_baseline = any(
        p["significant"] and "small_world_vs_baseline" in p["pair"] and
        cond_bpbs["small_world"] < cond_bpbs.get("baseline", [9999])
        for p in pairwise
    )
    # More direct check:
    sw_m = np.mean(cond_bpbs["small_world"])
    rg_m = np.mean(cond_bpbs["random_graph"])
    sf_m = np.mean(cond_bpbs["scale_free"])
    bl_m = np.mean(cond_bpbs["baseline"])

    sw_vs_bl = next((p for p in pairwise if set(p["pair"].split("_vs_")) == {"small_world", "baseline"}), None)
    sw_vs_rg = next((p for p in pairwise if set(p["pair"].split("_vs_")) == {"small_world", "random_graph"}), None)
    sw_vs_sf = next((p for p in pairwise if set(p["pair"].split("_vs_")) == {"small_world", "scale_free"}), None)

    topology_specific = (
        sw_m < bl_m and
        (sw_vs_rg is None or sw_m < rg_m) and
        (sw_vs_sf is None or sw_m < sf_m)
    )

    summary = {
        "conditions": cond_summary,
        "anova_f":    float(f_stat),
        "anova_p":    float(p_anova),
        "pairwise":   pairwise,
        "topology_specific_verdict": topology_specific,
        "elapsed_s":  time.time() - t0,
    }

    out_path = os.path.join(RESULTS_DIR, "specificity_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Phase 2 Results ──────────────────────────────────────")
    print(f"  One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
    print()
    rank = sorted(cond_summary.items(), key=lambda x: x[1]["mean"])
    for name, s in rank:
        print(f"  {name:14s}: {s['mean']:.4f} ± {s['sd']:.4f}")
    print()
    print("  Critical pairwise (Holm-Bonferroni corrected):")
    for p in pairwise:
        if "small_world" in p["pair"]:
            sig = "***" if p["significant"] else "ns"
            print(f"    {p['pair']:40s}  Δ={p['diff_a_minus_b']:+.4f}  p_corr={p['p_corrected']:.4f}  {sig}")
    print()
    print(f"  Topology-specific verdict: {'YES — SW specifically better' if topology_specific else 'NO — generic regularization'}")
    print(f"  Saved → {out_path}")
    return summary


if __name__ == "__main__":
    main()
