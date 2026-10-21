"""
Phase 1B: Threshold sensitivity analysis.

Re-runs the full GPT-2 topology analysis at 7 density thresholds
(5%, 10%, 15%, 20%, 25%, 30%, weighted) and tests whether the
early/late layer topological split is robust.

For each threshold:
  - Computes topology profile for all 144 heads
  - Tests layers 0-3 vs 4-11 brain-similarity with Mann-Whitney U
  - Records which layer contains the top-ranked head
  - Computes Kendall's τ between adjacent-threshold rankings
"""

from __future__ import annotations
import os, sys, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology"))
from attention_topology import extract_attention_graphs, _default_corpus
from topo_metrics import compute_topo_profile, threshold_and_binarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase1b")

THRESHOLDS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EARLY_LAYERS = list(range(4))    # layers 0-3
LATE_LAYERS  = list(range(4, 12)) # layers 4-11

HUMAN_FC_SIGMA = 2.73
HUMAN_FC_Q     = 0.60


def brain_similarity_score(profile) -> float:
    """Same composite metric as Experiment 2."""
    if profile.small_worldness_sigma <= 0 or profile.char_path_length <= 0:
        return 0.0
    sw_ratio  = min(profile.small_worldness_sigma / HUMAN_FC_SIGMA, HUMAN_FC_SIGMA / profile.small_worldness_sigma)
    q_sim     = 1.0 - abs(profile.modularity - HUMAN_FC_Q) / HUMAN_FC_Q
    eff_score = profile.global_efficiency
    return float(np.clip(np.mean([sw_ratio, max(q_sim, 0), eff_score]), 0, 1))


def weighted_brain_similarity(A: np.ndarray) -> float:
    """
    Brain similarity on weighted adjacency matrix using weighted graph metrics.
    Avoids arbitrary threshold entirely.
    """
    import networkx as nx
    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return 0.0
    try:
        C_w = nx.average_clustering(G, weight="weight")
        # Weighted path length via reciprocal weights
        Gw = nx.from_numpy_array(1.0 / (A + 1e-6) - (1.0 / 1e-6) * (A < 1e-6))
        if nx.is_connected(G):
            L_w = nx.average_shortest_path_length(G)
        else:
            Gcc = max(nx.connected_components(G), key=len)
            L_w = nx.average_shortest_path_length(G.subgraph(Gcc))
        # Compare to human FC targets
        target_C, target_L = 0.45, 2.3
        c_sim = 1.0 - abs(C_w - target_C) / target_C
        l_sim = 1.0 - abs(L_w - target_L) / target_L
        return float(np.clip(np.mean([c_sim, l_sim]), 0, 1))
    except Exception:
        return 0.0


def analyse_threshold(attn_mats, threshold, n_layers=12, n_heads=12):
    """Run topology analysis for one density threshold."""
    scores_by_head = {}  # label -> score
    early_scores, late_scores = [], []

    for layer in range(n_layers):
        for head in range(n_heads):
            A = attn_mats[layer, head]
            A = (A + A.T) / 2
            np.fill_diagonal(A, 0)
            label = f"L{layer:02d}H{head:02d}"

            if threshold == "weighted":
                score = weighted_brain_similarity(A)
            else:
                profile = compute_topo_profile(
                    A, label=label, density_threshold=threshold, symmetrize=False
                )
                score = brain_similarity_score(profile)

            scores_by_head[label] = score
            if layer in EARLY_LAYERS:
                early_scores.append(score)
            else:
                late_scores.append(score)

    # Mann-Whitney U: early vs late
    u_stat, p_mw = stats.mannwhitneyu(early_scores, late_scores, alternative="greater")
    # rank-biserial correlation as effect size
    n1, n2 = len(early_scores), len(late_scores)
    rb = 1 - (2 * u_stat) / (n1 * n2)

    top_head = max(scores_by_head, key=scores_by_head.get)
    top_layer = int(top_head[1:3])
    top_score = scores_by_head[top_head]

    return {
        "threshold":    threshold,
        "early_mean":   float(np.mean(early_scores)),
        "late_mean":    float(np.mean(late_scores)),
        "u_stat":       float(u_stat),
        "p_mannwhitney": float(p_mw),
        "rank_biserial": float(rb),
        "top_head":     top_head,
        "top_layer":    top_layer,
        "top_score":    float(top_score),
        "scores_by_head": {k: float(v) for k, v in scores_by_head.items()},
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Phase 1B: Threshold sensitivity analysis")
    t0 = time.time()

    print("  Extracting GPT-2 attention matrices...")
    corpus = _default_corpus()
    attn_data = extract_attention_graphs("gpt2", sentences=corpus, max_len=32)
    attn_mats = attn_data["attention_matrices"]  # (12, 12, seq, seq)
    n_layers  = attn_data["n_layers"]
    n_heads   = attn_data["n_heads"]
    print(f"  Got attention matrices: {attn_mats.shape}")

    all_results = []
    threshold_list = THRESHOLDS + ["weighted"]

    for thr in threshold_list:
        print(f"  Threshold {thr}...", end=" ", flush=True)
        res = analyse_threshold(attn_mats, thr, n_layers, n_heads)
        all_results.append(res)
        print(f"early={res['early_mean']:.3f} late={res['late_mean']:.3f} "
              f"p={res['p_mannwhitney']:.4f} top={res['top_head']}")

    # Kendall's τ between adjacent-threshold rankings
    kendall_taus = []
    for i in range(len(THRESHOLDS) - 1):
        s1 = all_results[i]["scores_by_head"]
        s2 = all_results[i+1]["scores_by_head"]
        heads = sorted(s1.keys())
        v1 = [s1[h] for h in heads]
        v2 = [s2[h] for h in heads]
        tau, p_tau = stats.kendalltau(v1, v2)
        kendall_taus.append({
            "pair": f"{THRESHOLDS[i]}_vs_{THRESHOLDS[i+1]}",
            "tau": float(tau),
            "p":   float(p_tau),
        })

    # How many thresholds show significant early > late split?
    n_significant = sum(1 for r in all_results if r["p_mannwhitney"] < 0.05)
    top_layers = [r["top_layer"] for r in all_results]
    layer3_tops = sum(1 for l in top_layers if l == 3)

    summary = {
        "thresholds_tested": threshold_list,
        "n_significant_splits": n_significant,
        "top_layers_by_threshold": top_layers,
        "layer3_is_top_n_thresholds": layer3_tops,
        "kendall_taus": kendall_taus,
        "per_threshold": all_results,
        "elapsed_s": time.time() - t0,
        "verdict": (
            "ROBUST" if n_significant >= 5 else
            "PARTIAL" if n_significant >= 3 else
            "THRESHOLD_DEPENDENT"
        ),
    }

    out_path = os.path.join(RESULTS_DIR, "threshold_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Phase 1B Results ─────────────────────────────")
    print(f"  Significant early>late splits: {n_significant}/{len(threshold_list)}")
    print(f"  Layer 3 is top head at: {layer3_tops}/{len(threshold_list)} thresholds")
    print(f"  Verdict: {summary['verdict']}")
    taus = [t['tau'] for t in kendall_taus]
    print(f"  Kendall τ (adjacent thresholds): mean={np.mean(taus):.3f} min={min(taus):.3f}")
    print(f"  Saved → {out_path}")
    return summary


if __name__ == "__main__":
    main()
