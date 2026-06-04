"""
Master pipeline orchestrator.

Runs all phases in dependency order, applying decision gates,
and produces a unified results summary with paper-ready tables.

Usage:
    python run_pipeline.py [--phases 1a 1b 2 3 4 5] [--skip-gates]

Default: runs all phases in order.
"""

import os, sys, json, time, argparse
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)


def load_result(phase_name: str):
    paths = {
        "1a": "results/phase1a/summary.json",
        "1b": "results/phase1b/threshold_sensitivity.json",
        "2":  "results/phase2/specificity_results.json",
        "3":  "results/phase3/rsa_results.json",
        "4":  "results/phase4/crossmodel_results.json",
        "5":  "results/phase5/mechanistic_summary.json",
    }
    p = os.path.join(PIPELINE_DIR, paths[phase_name])
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_phase(name, module_path, skip_gates, gate_fn=None, gate_msg=""):
    separator(f"PHASE {name.upper()}")
    existing = load_result(name)
    if existing:
        print(f"  Cached result found — loading (delete to re-run)")
        return existing

    import importlib.util
    spec = importlib.util.spec_from_file_location(f"phase{name}", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.main()

    if gate_fn and not skip_gates:
        passed = gate_fn(result)
        if not passed:
            print(f"\n  ⚠ GATE FAILED: {gate_msg}")
            print(f"  Downstream phases may still run (use --skip-gates to suppress warning)")
    return result


def gate_1a(result):
    """Phase 1A gate: effect must be real before running Phase 2."""
    return result.get("significant", False) or result.get("p_one_tailed", 1.0) < 0.10


def gate_1b(result):
    """Phase 1B gate: topological split must be at least partially robust."""
    return result.get("verdict") in ("ROBUST", "PARTIAL")


def build_summary_report(results: dict):
    """Generate a structured summary with paper-ready tables."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("  PIPELINE SUMMARY REPORT")
    lines.append("="*70)

    # ── Phase 1A ──────────────────────────────────────────────────────────
    r1a = results.get("1a")
    if r1a:
        lines.append("\n## Phase 1A: Multi-seed validation")
        lines.append(f"  Baseline:    {r1a['baseline_mean']:.4f} ± {r1a['baseline_sd']:.4f}")
        lines.append(f"  Small-world: {r1a['sw_mean']:.4f} ± {r1a['sw_sd']:.4f}")
        lines.append(f"  Δ (b-sw):    {r1a['mean_diff']:+.4f}  95% CI [{r1a['ci_95_lo']:+.4f}, {r1a['ci_95_hi']:+.4f}]")
        lines.append(f"  Cohen's d:   {r1a['cohens_d']:.3f}")
        lines.append(f"  p (1-tail):  {r1a['p_one_tailed']:.4f}")
        verdict = "REAL EFFECT" if r1a.get("significant") else "NOT SIGNIFICANT"
        lines.append(f"  Verdict:     {verdict}")

    # ── Phase 1B ──────────────────────────────────────────────────────────
    r1b = results.get("1b")
    if r1b:
        lines.append("\n## Phase 1B: Threshold sensitivity")
        lines.append(f"  Thresholds tested:        {len(r1b['thresholds_tested'])}")
        lines.append(f"  Significant splits (p<.05): {r1b['n_significant_splits']}/{len(r1b['thresholds_tested'])}")
        lines.append(f"  Layer 3 is top at:        {r1b['layer3_is_top_n_thresholds']}/{len(r1b['thresholds_tested'])} thresholds")
        taus = [t['tau'] for t in r1b['kendall_taus']]
        lines.append(f"  Kendall τ (adj. thresh):  mean={np.mean(taus):.3f}  min={min(taus):.3f}")
        lines.append(f"  Verdict:                  {r1b['verdict']}")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    r2 = results.get("2")
    if r2:
        lines.append("\n## Phase 2: Topology Specificity Ablation")
        lines.append(f"  One-way ANOVA: F={r2['anova_f']:.3f}  p={r2['anova_p']:.4f}")
        lines.append("")
        lines.append(f"  {'Condition':<16} {'Mean':>8} {'SD':>8}")
        lines.append(f"  {'-'*34}")
        conds = r2["conditions"]
        for name, s in sorted(conds.items(), key=lambda x: x[1]["mean"]):
            lines.append(f"  {name:<16} {s['mean']:>8.4f} {s['sd']:>8.4f}")
        lines.append("")
        lines.append("  Critical pairwise (Holm-Bonferroni):")
        for p in r2["pairwise"]:
            if "small_world" in p["pair"]:
                sig = "***" if p["significant"] else "ns"
                lines.append(f"    {p['pair']:42s} Δ={p['diff_a_minus_b']:+.4f}  p_corr={p['p_corrected']:.4f}  {sig}")
        spec = r2["topology_specific_verdict"]
        lines.append(f"\n  Specificity verdict: {'BRAIN-LIKE TOPOLOGY IS SPECIFICALLY BETTER' if spec else 'GENERIC REGULARIZATION EFFECT'}")

    # ── Phase 3 ───────────────────────────────────────────────────────────
    r3 = results.get("3")
    if r3:
        conf = r3.get("confirmatory", {})
        lines.append("\n## Phase 3: Real-data RSA (preregistered)")
        if "rho_partial" in conf:
            sig = "SIGNIFICANT" if conf["significant_p05"] else "not significant"
            lines.append(f"  Layer 3 × frontoparietal: ρ={conf['rho_partial']:.4f}  p={conf['p_permutation']:.4f}  [{sig}]")
            lines.append(f"  95% CI: [{conf['ci_95_lo']:.4f}, {conf['ci_95_hi']:.4f}]")
            lines.append(f"  Noise ceiling: [{conf['noise_ceiling_lo']:.3f}, {conf['noise_ceiling_hi']:.3f}]")
            lines.append(f"  ρ/NC: {conf['rho_over_nc']:.3f}")
        else:
            lines.append(f"  {conf.get('error', 'No result')}")
        lines.append(f"  Peak layer (exploratory): {r3.get('peak_layer_exploratory', '?')}")
        lines.append(f"  Preregistered layer is peak: {r3.get('preregistered_layer_is_peak', '?')}")
        lines.append(f"  Data: {r3.get('data_source', 'unknown')}")

    # ── Phase 4 ───────────────────────────────────────────────────────────
    r4 = results.get("4")
    if r4:
        lines.append("\n## Phase 4: Cross-model correlation")
        for fam in ["pythia", "gpt2"]:
            a = r4.get(fam, {}).get("analysis", {})
            if "error" in a:
                lines.append(f"  {fam}: {a['error']}")
                continue
            ea = a.get("early_vs_ppl", {})
            la = a.get("late_vs_ppl",  {})
            lines.append(f"  {fam} ({a.get('n_models','?')} models):")
            lines.append(f"    Early layers ↔ log-PPL: ρ={ea.get('rho',0):+.3f}  p={ea.get('p',1):.4f}")
            lines.append(f"    Late  layers ↔ log-PPL: ρ={la.get('rho',0):+.3f}  p={la.get('p',1):.4f}")
            lines.append(f"    Prediction correct:     {a.get('prediction_correct', '?')}")

    # ── Phase 5 ───────────────────────────────────────────────────────────
    r5 = results.get("5")
    if r5:
        lines.append("\n## Phase 5: Mechanistic analysis")
        abl = r5.get("5b_ablation", {})
        if "baseline_bpb" in abl:
            lines.append(f"  5B Ablation:")
            lines.append(f"    Baseline:                 {abl['baseline_bpb']:.4f}")
            lines.append(f"    Brain-like heads ablated: {abl['top_brain_like_ablated_bpb']:.4f}  (Δ={abl['top_cost']:+.4f})")
            lines.append(f"    Non-brain heads ablated:  {abl['bottom_brain_like_ablated_bpb']:.4f}  (Δ={abl['bottom_cost']:+.4f})")
            lines.append(f"    Brain-like heads more important: {abl['brain_like_heads_more_important']}")
        bsm = r5.get("5c_brain_specific", {})
        if "early_mean_hier_levels" in bsm:
            lines.append(f"  5C Hierarchical modularity:")
            lines.append(f"    Early layers mean levels: {bsm['early_mean_hier_levels']:.2f}")
            lines.append(f"    Late  layers mean levels: {bsm['late_mean_hier_levels']:.2f}")

    # ── Overall claim level ───────────────────────────────────────────────
    lines.append("\n" + "="*70)
    lines.append("  CLAIM LEVEL ASSESSMENT")
    lines.append("="*70)

    claim_level = 1
    reasons = []

    if r1a and r1a.get("significant"):
        claim_level = max(claim_level, 2)
        reasons.append("Phase 1A: Multi-seed effect confirmed")
    if r2 and r2.get("topology_specific_verdict"):
        claim_level = max(claim_level, 3)
        reasons.append("Phase 2: Brain-like topology specifically better (not generic regularization)")
    if r3:
        conf = r3.get("confirmatory", {})
        if conf.get("significant_p05"):
            claim_level = max(claim_level, 4)
            reasons.append("Phase 3: Preregistered RSA survives with real data")
    if r4:
        pa = r4.get("pythia", {}).get("analysis", {})
        if pa.get("prediction_correct"):
            claim_level = max(claim_level, 3)
            reasons.append("Phase 4: Cross-model correlation holds in Pythia family")

    claim_labels = {
        1: "STRUCTURAL DESCRIPTION ONLY — topology difference observed, no performance link",
        2: "MULTI-SEED PERFORMANCE EFFECT — brain-like topology improves training (generic or specific TBD)",
        3: "TOPOLOGY-SPECIFIC PERFORMANCE EFFECT — brain-like structure specifically beneficial",
        4: "FULL CONVERGENT CLAIM — structural + representational + performance evidence",
    }
    lines.append(f"\n  Level {claim_level}: {claim_labels[claim_level]}")
    for r in reasons:
        lines.append(f"    ✓ {r}")

    report = "\n".join(lines)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="*", default=["1a","1b","2","3","4","5"],
                        help="Which phases to run")
    parser.add_argument("--skip-gates", action="store_true")
    args = parser.parse_args()

    t_total = time.time()
    results = {}

    phase_configs = {
        "1a": ("phase1a_multiseed.py",  gate_1a, "Effect not significant — SW topology may not help"),
        "1b": ("phase1b_threshold.py",  gate_1b, "Topology split is threshold-dependent"),
        "2":  ("phase2_specificity.py", None,    ""),
        "3":  ("phase3_real_rsa.py",    None,    ""),
        "4":  ("phase4_crossmodel.py",  None,    ""),
        "5":  ("phase5_mechanistic.py", None,    ""),
    }

    for phase in args.phases:
        if phase not in phase_configs:
            print(f"Unknown phase: {phase}")
            continue
        fname, gate_fn, gate_msg = phase_configs[phase]
        module_path = os.path.join(PIPELINE_DIR, fname)
        result = run_phase(phase, module_path, args.skip_gates, gate_fn, gate_msg)
        results[phase] = result

    report = build_summary_report(results)
    print(report)

    report_path = os.path.join(PIPELINE_DIR, "results", "PIPELINE_REPORT.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    summary_path = os.path.join(PIPELINE_DIR, "results", "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s")
    print(f"  Report → {report_path}")
    print(f"  JSON   → {summary_path}")


if __name__ == "__main__":
    main()
