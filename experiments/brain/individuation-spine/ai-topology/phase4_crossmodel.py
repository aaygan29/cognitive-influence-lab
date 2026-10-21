"""
Phase 4: Cross-model correlation study.

Tests whether better language models have more brain-like early-layer topology.
Uses the Pythia model family (controlled training, same data, different scale)
plus GPT-2 variants.

Prediction (preregistered): across Pythia 70m→6.9b, early-layer brain-similarity
correlates negatively with log-perplexity (better models = more brain-like).

Statistics:
  - Spearman ρ with bootstrap 95% CI
  - Separate Pythia-only and GPT-2-only analyses
  - Partial correlation controlling for log(parameter count)
  - Control: same correlation for late-layer scores (should be weaker)
"""

import os, sys, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology"))
from attention_topology import extract_attention_graphs, _default_corpus
from topo_metrics import compute_topo_profile

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase4")

# Models to analyze: (name, hf_id, n_params_M, known_ppl_wikitext103)
# Perplexity values from published EleutherAI/Pythia and GPT-2 papers
PYTHIA_MODELS = [
    ("pythia-70m",   "EleutherAI/pythia-70m",   70,    29.0),
    ("pythia-160m",  "EleutherAI/pythia-160m",  160,   18.2),
    ("pythia-410m",  "EleutherAI/pythia-410m",  410,   13.9),
    ("pythia-1b",    "EleutherAI/pythia-1b",    1000,  11.4),
    ("pythia-1.4b",  "EleutherAI/pythia-1.4b",  1400,  10.6),
    ("pythia-2.8b",  "EleutherAI/pythia-2.8b",  2800,  9.3),
    ("pythia-6.9b",  "EleutherAI/pythia-6.9b",  6900,  8.3),
]

GPT2_MODELS = [
    ("gpt2-small",  "gpt2",        117,   29.4),
    ("gpt2-medium", "gpt2-medium", 345,   22.8),
    ("gpt2-large",  "gpt2-large",  774,   19.4),
    ("gpt2-xl",     "gpt2-xl",    1558,   17.5),
]

HUMAN_FC_SIGMA = 2.73
HUMAN_FC_Q     = 0.60


def brain_similarity_score(profile) -> float:
    if profile.small_worldness_sigma <= 0 or profile.char_path_length <= 0:
        return 0.0
    sw_ratio = min(profile.small_worldness_sigma / HUMAN_FC_SIGMA,
                   HUMAN_FC_SIGMA / profile.small_worldness_sigma)
    q_sim    = 1.0 - abs(profile.modularity - HUMAN_FC_Q) / HUMAN_FC_Q
    eff      = profile.global_efficiency
    return float(np.clip(np.mean([sw_ratio, max(q_sim, 0), eff]), 0, 1))


def analyse_model(name, hf_id, n_params_m, ppl, corpus, density=0.15):
    """Extract topology for one model. Returns dict with per-layer scores."""
    print(f"  Extracting: {name}...", flush=True)
    try:
        attn_data = extract_attention_graphs(hf_id, sentences=corpus, max_len=32)
    except Exception as e:
        print(f"    FAILED: {e}")
        return None

    mats     = attn_data["attention_matrices"]
    n_layers = attn_data["n_layers"]
    n_heads  = attn_data["n_heads"]

    early_cutoff = max(1, int(n_layers * 0.25))  # first 25% of layers
    late_cutoff  = int(n_layers * 0.50)           # last 50% of layers

    layer_scores = []
    for layer in range(n_layers):
        head_scores = []
        for head in range(n_heads):
            A = mats[layer, head]
            A = (A + A.T) / 2
            np.fill_diagonal(A, 0)
            try:
                profile = compute_topo_profile(
                    A, label=f"L{layer:02d}H{head:02d}",
                    density_threshold=density, symmetrize=False
                )
                score = brain_similarity_score(profile)
            except Exception:
                score = 0.0
            head_scores.append(score)
        layer_scores.append(float(np.mean(head_scores)))

    early_mean = float(np.mean(layer_scores[:early_cutoff]))
    late_mean  = float(np.mean(layer_scores[late_cutoff:]))
    global_mean = float(np.mean(layer_scores))

    return {
        "name":          name,
        "hf_id":         hf_id,
        "n_params_m":    n_params_m,
        "log_ppl":       float(np.log(ppl)),
        "ppl":           float(ppl),
        "n_layers":      n_layers,
        "n_heads":       n_heads,
        "layer_scores":  layer_scores,
        "early_mean":    early_mean,
        "late_mean":     late_mean,
        "global_mean":   global_mean,
        "early_cutoff_layers": early_cutoff,
    }


def spearman_with_ci(x, y, n_boot=10_000, seed=0):
    rho, p = stats.spearmanr(x, y)
    rng    = np.random.default_rng(seed)
    boot   = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        r, _ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
        boot.append(r)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return float(rho), float(p), ci_lo, ci_hi


def partial_spearman_control_params(x, y, control):
    """Partial Spearman ρ between x and y, controlling for control variable."""
    from scipy.stats import spearmanr, rankdata
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(control)
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(rc)), rc])
    rx_resid = rx - X @ lstsq(X, rx, rcond=None)[0]
    ry_resid = ry - X @ lstsq(X, ry, rcond=None)[0]
    rho, p = stats.pearsonr(rx_resid, ry_resid)
    return float(rho), float(p)


def analyse_family(model_results, family_name):
    """Run correlation analysis for a model family."""
    valid = [r for r in model_results if r is not None]
    if len(valid) < 3:
        return {"error": f"Too few valid models in {family_name}"}

    early = [r["early_mean"] for r in valid]
    late  = [r["late_mean"]  for r in valid]
    log_ppl = [r["log_ppl"] for r in valid]
    log_params = [np.log(r["n_params_m"]) for r in valid]

    rho_e, p_e, ci_lo_e, ci_hi_e = spearman_with_ci(early, log_ppl)
    rho_l, p_l, ci_lo_l, ci_hi_l = spearman_with_ci(late,  log_ppl)

    part_rho_e, part_p_e = partial_spearman_control_params(early, log_ppl, log_params)
    part_rho_l, part_p_l = partial_spearman_control_params(late,  log_ppl, log_params)

    return {
        "family":        family_name,
        "n_models":      len(valid),
        "models":        [r["name"] for r in valid],
        "early_vs_ppl":  {"rho": rho_e, "p": p_e, "ci_lo": ci_lo_e, "ci_hi": ci_hi_e},
        "late_vs_ppl":   {"rho": rho_l, "p": p_l, "ci_lo": ci_lo_l, "ci_hi": ci_hi_l},
        "early_partial_vs_ppl_controlling_params": {"rho": part_rho_e, "p": part_p_e},
        "late_partial_vs_ppl_controlling_params":  {"rho": part_rho_l, "p": part_p_l},
        "prediction_correct": bool(rho_e < 0 and p_e < 0.05),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Phase 4: Cross-model correlation (Pythia + GPT-2 families)")
    t0 = time.time()

    corpus = _default_corpus()

    pythia_results = []
    for name, hf_id, npar, ppl in PYTHIA_MODELS:
        r = analyse_model(name, hf_id, npar, ppl, corpus)
        pythia_results.append(r)

    gpt2_results = []
    for name, hf_id, npar, ppl in GPT2_MODELS:
        r = analyse_model(name, hf_id, npar, ppl, corpus)
        gpt2_results.append(r)

    pythia_analysis = analyse_family(pythia_results, "pythia")
    gpt2_analysis   = analyse_family(gpt2_results,   "gpt2")

    # Combined analysis
    all_results = [r for r in (pythia_results + gpt2_results) if r is not None]
    combined_analysis = analyse_family(all_results, "combined") if len(all_results) >= 4 else {}

    summary = {
        "pythia":   {"model_results": pythia_results, "analysis": pythia_analysis},
        "gpt2":     {"model_results": gpt2_results,   "analysis": gpt2_analysis},
        "combined": combined_analysis,
        "elapsed_s": time.time() - t0,
    }

    out_path = os.path.join(RESULTS_DIR, "crossmodel_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── Phase 4 Results ─────────────────────────────────────")
    for fam_name, analysis in [("Pythia", pythia_analysis), ("GPT-2", gpt2_analysis)]:
        if "error" not in analysis:
            print(f"  {fam_name} ({analysis['n_models']} models):")
            ea = analysis["early_vs_ppl"]
            la = analysis["late_vs_ppl"]
            print(f"    Early layers ↔ log-PPL: ρ={ea['rho']:+.3f}  p={ea['p']:.4f}  CI [{ea['ci_lo']:+.3f}, {ea['ci_hi']:+.3f}]")
            print(f"    Late  layers ↔ log-PPL: ρ={la['rho']:+.3f}  p={la['p']:.4f}  CI [{la['ci_lo']:+.3f}, {la['ci_hi']:+.3f}]")
            print(f"    Prediction correct: {analysis['prediction_correct']}")
    print(f"  Saved → {out_path}")
    return summary


if __name__ == "__main__":
    main()
