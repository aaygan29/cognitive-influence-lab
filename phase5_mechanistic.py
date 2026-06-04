"""
Phase 5: Mechanistic analysis.

5A: Training trajectory — does brain-like topology develop naturally?
    Tracks val_bpb and brain-similarity per layer at each checkpoint.
    Uses saved checkpoints from Phase 2 (baseline + small_world seeds 0-2).

5B: Head ablation — do brain-like heads carry more functional load?
    Ablates top-10 vs bottom-10 brain-like heads; measures val_bpb cost.

5C: Rich-club and hierarchical modularity — brain-specific signatures.
    Adds metrics that distinguish brains from other small-world systems.
"""

import os, sys, json, time, math
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology"))

from train_core import train, get_data, get_batch, eval_val_bpb, DEVICE, NanoGPT
from topo_metrics import compute_topo_profile

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results", "phase5")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

HUMAN_FC_SIGMA = 2.73
HUMAN_FC_Q     = 0.60


# ── Brain similarity ──────────────────────────────────────────────────────────

def brain_score(profile) -> float:
    if profile.small_worldness_sigma <= 0:
        return 0.0
    sw  = min(profile.small_worldness_sigma / HUMAN_FC_SIGMA,
              HUMAN_FC_SIGMA / profile.small_worldness_sigma)
    q   = 1.0 - abs(profile.modularity - HUMAN_FC_Q) / HUMAN_FC_Q
    eff = profile.global_efficiency
    return float(np.clip(np.mean([sw, max(q, 0), eff]), 0, 1))


def layer_brain_scores(model, sentences, cfg, density=0.15):
    """Compute mean brain-similarity score per layer for a trained model."""
    import torch.nn.functional as F
    model.eval()
    scores_by_layer = []

    # Build a simple tokenizer from Shakespeare data
    from train_core import get_data as gd
    _, _, vocab = gd()
    path = os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology",
                        "data", "train_data", "shakespeare.txt")
    text  = open(path).read()
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}

    n_layers = cfg["n_layer"]
    n_heads  = cfg["n_head"]
    accum    = None
    n        = 0

    with torch.no_grad():
        for sent in sentences[:20]:  # use 20 sentences for speed
            tokens = [stoi.get(c, 0) for c in sent[:cfg["block_size"]]]
            if len(tokens) < 2:
                continue
            idx = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
            T   = idx.size(1)
            pos = torch.arange(T, dtype=torch.long, device=DEVICE)
            x   = model.drop(model.wte(idx) + model.wpe(pos))

            head_mats_per_layer = []
            for block in model.blocks:
                B_, T_, C = x.size()
                h_states  = block.ln1(x)
                q, k, v   = block.attn.c_attn(h_states).split(cfg["n_embd"], dim=2)
                nh  = cfg["n_head"]
                k_  = k.view(B_,T_,nh,C//nh).transpose(1,2)
                q_  = q.view(B_,T_,nh,C//nh).transpose(1,2)
                att = (q_ @ k_.transpose(-2,-1)) / math.sqrt(k_.size(-1))
                att = att.masked_fill(block.attn.bias[:,:,:T_,:T_]==0, float("-inf"))
                att = F.softmax(att, dim=-1)   # (1, nh, T, T)
                head_mats_per_layer.append(att[0].cpu().numpy())  # (nh, T, T)
                x = block(x)

            mat = np.stack(head_mats_per_layer)  # (n_layer, nh, T, T)
            s   = min(mat.shape[-1], accum.shape[-1]) if accum is not None else mat.shape[-1]
            if accum is None:
                accum = mat[..., :s, :s]
            else:
                accum = accum[..., :s, :s] + mat[..., :s, :s]
            n += 1

    if n == 0:
        return [0.0] * n_layers

    accum /= n

    for layer in range(n_layers):
        head_scores = []
        for head in range(n_heads):
            A = accum[layer, head]
            A = (A + A.T) / 2
            np.fill_diagonal(A, 0)
            try:
                p = compute_topo_profile(A, label=f"L{layer}H{head}",
                                         density_threshold=density, symmetrize=False)
                head_scores.append(brain_score(p))
            except Exception:
                head_scores.append(0.0)
        scores_by_layer.append(float(np.mean(head_scores)))

    return scores_by_layer


# ── Phase 5A: Training trajectory ────────────────────────────────────────────

def phase5a_trajectory():
    print("  5A: Training trajectory analysis (3 seeds × 2 conditions)...")
    from attention_topology import _default_corpus
    sentences = _default_corpus()

    trajectory_data = []
    cfg_base = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.1,
                    batch_size=32, attn_top_k=0)

    for condition, lam, tgt in [("baseline", 0.0, "none"), ("small_world", 0.10, "small_world")]:
        for seed in [42, 43, 44]:
            print(f"    {condition} seed={seed}...")
            ckpt_dir = os.path.join(CHECKPOINT_DIR, f"{condition}_s{seed}")

            result = train(
                seed=seed, lambda_topo=lam, topo_target=tgt,
                checkpoint_every=50, checkpoint_dir=ckpt_dir,
                max_iters=500, verbose=False,
            )

            # Load each checkpoint and compute brain scores
            traj = []
            for step, val_bpb in result["trajectory"]:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:04d}.pt")
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=DEVICE)
                    cfg  = ckpt["cfg"]
                    _, _, vocab = get_data()
                    model = NanoGPT(vocab, cfg).to(DEVICE)
                    model.load_state_dict(ckpt["model"])
                    scores = layer_brain_scores(model, sentences, cfg)
                    traj.append({
                        "step": step, "val_bpb": val_bpb,
                        "early_brain_score": float(np.mean(scores[:2])),
                        "late_brain_score":  float(np.mean(scores[2:])),
                        "all_layer_scores":  scores,
                    })

            trajectory_data.append({
                "condition": condition, "seed": seed,
                "trajectory": traj,
                "final_val_bpb": result["final_val_bpb"],
            })

    return trajectory_data


# ── Phase 5B: Head ablation ───────────────────────────────────────────────────

def phase5b_ablation():
    print("  5B: Head ablation study...")
    from attention_topology import _default_corpus, extract_attention_graphs
    sentences = _default_corpus()

    # Train a baseline model for ablation
    result = train(seed=42, lambda_topo=0.0, topo_target="none",
                   checkpoint_every=500, verbose=False)
    _, _, vocab = get_data()
    cfg = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.1,
               batch_size=32, attn_top_k=0, lambda_topo=0.0, topo_target="none")
    model = NanoGPT(vocab, cfg).to(DEVICE)

    # Retrain to get the model state
    torch.manual_seed(42)
    train_data, val_data, _ = get_data()

    # Score all heads
    attn_data = extract_attention_graphs("gpt2", sentences, max_len=32)
    from attention_topology import build_head_adjacency
    head_scores = {}
    mats = attn_data["attention_matrices"]
    for layer in range(4):
        for head in range(4):
            A = build_head_adjacency(mats, layer, head)
            try:
                p = compute_topo_profile(A, label=f"L{layer}H{head}",
                                         density_threshold=0.15, symmetrize=False)
                head_scores[(layer, head)] = brain_score(p)
            except Exception:
                head_scores[(layer, head)] = 0.0

    ranked = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    top_heads    = [(l, h) for (l, h), _ in ranked[:10]]
    bottom_heads = [(l, h) for (l, h), _ in ranked[-10:]]

    # Use GPT-2's attention (more meaningful than nanoGPT for ablation)
    # We ablate by zeroing the output projection of specific heads
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch.nn.functional as F

    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def eval_gpt2_ppl(model_, sentences_):
        losses = []
        with torch.no_grad():
            for sent in sentences_:
                toks = tokenizer.encode(sent, return_tensors="pt")
                out  = model_(toks, labels=toks)
                losses.append(out.loss.item())
        return float(np.mean(losses)) / math.log(2)

    baseline_bpb = eval_gpt2_ppl(gpt2, sentences)

    def ablate_heads(model_, head_list):
        """Zero out the output of specific (layer, head) pairs."""
        import copy
        m = copy.deepcopy(model_)
        for layer, head in head_list:
            if layer < len(m.transformer.h):
                blk = m.transformer.h[layer]
                n_embd = m.config.n_embd
                n_head  = m.config.n_head
                head_dim = n_embd // n_head
                # Zero the head's slice of c_proj weight
                with torch.no_grad():
                    blk.attn.c_proj.weight[:, head * head_dim:(head + 1) * head_dim] = 0
        return m

    top_ablated    = ablate_heads(gpt2, top_heads)
    bottom_ablated = ablate_heads(gpt2, bottom_heads)

    top_bpb    = eval_gpt2_ppl(top_ablated,    sentences)
    bottom_bpb = eval_gpt2_ppl(bottom_ablated, sentences)

    return {
        "baseline_bpb":        baseline_bpb,
        "top_brain_like_ablated_bpb":    top_bpb,
        "bottom_brain_like_ablated_bpb": bottom_bpb,
        "top_cost":    float(top_bpb    - baseline_bpb),
        "bottom_cost": float(bottom_bpb - baseline_bpb),
        "brain_like_heads_more_important": bool(top_bpb > bottom_bpb),
        "top_heads":    [(int(l), int(h), float(head_scores[(l,h)])) for l, h in top_heads],
        "bottom_heads": [(int(l), int(h), float(head_scores[(l,h)])) for l, h in bottom_heads],
    }


# ── Phase 5C: Rich-club and hierarchical modularity ──────────────────────────

def rich_club_curve(A: np.ndarray, k_range=None):
    """Compute rich-club coefficient φ(k) for a range of degree thresholds."""
    import networkx as nx
    G = nx.from_numpy_array(A)
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    if k_range is None:
        k_range = range(1, min(max_deg, 10) + 1)

    phi_curve = {}
    for k in k_range:
        rich_nodes = [n for n, d in degrees.items() if d > k]
        N_k = len(rich_nodes)
        if N_k < 2:
            phi_curve[k] = 0.0
            continue
        subgraph  = G.subgraph(rich_nodes)
        E_k       = subgraph.number_of_edges()
        max_edges = N_k * (N_k - 1) / 2
        phi_curve[k] = float(E_k / max_edges) if max_edges > 0 else 0.0

    return phi_curve


def hierarchical_modularity(A: np.ndarray, n_levels=3):
    """Recursive community detection to detect hierarchical modularity."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    G = nx.from_numpy_array(A)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_nodes() < 4:
        return {"n_levels": 0, "level_modularities": []}

    level_mods = []
    current_communities = list(greedy_modularity_communities(G))
    try:
        import networkx.algorithms.community as nxc
        Q = nxc.modularity(G, current_communities)
    except Exception:
        Q = 0.0
    level_mods.append(float(Q))

    for level in range(1, n_levels):
        sub_communities = []
        for comm in current_communities:
            if len(comm) < 4:
                continue
            subG = G.subgraph(comm).copy()
            try:
                sub_comms = list(greedy_modularity_communities(subG))
                if len(sub_comms) > 1:
                    sub_communities.extend(sub_comms)
                    Q_sub = nxc.modularity(subG, sub_comms)
                    level_mods.append(float(Q_sub))
            except Exception:
                pass
        if not sub_communities:
            break
        current_communities = sub_communities

    return {
        "n_levels":           len(level_mods),
        "level_modularities": level_mods,
        "mean_modularity":    float(np.mean(level_mods)) if level_mods else 0.0,
    }


def phase5c_brain_specific_metrics():
    print("  5C: Rich-club and hierarchical modularity analysis...")
    from attention_topology import _default_corpus, extract_attention_graphs, build_head_adjacency
    from topo_metrics import threshold_and_binarize

    sentences = _default_corpus()
    attn_data = extract_attention_graphs("gpt2", sentences, max_len=32)
    mats      = attn_data["attention_matrices"]

    # Also load human FC matrix
    fc_path = os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology",
                           "data", "cached_connectomes")

    results = {"heads": [], "human_fc": None}

    # Human FC rich-club (load from cached connectomes)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "neuro-ai-topology"))
        from data.connectomes import load_connectomes
        connectomes = load_connectomes()
        human_fc = connectomes.get("human_fc")
        if human_fc is not None:
            A_fc = threshold_and_binarize(human_fc, threshold=0.15)
            results["human_fc"] = {
                "rich_club": rich_club_curve(A_fc),
                "hierarchical": hierarchical_modularity(A_fc),
            }
    except Exception as e:
        print(f"    Human FC load failed: {e}")

    # Analyze representative heads: top 5 brain-like (early) + bottom 5 (late)
    from topo_metrics import compute_topo_profile

    for layer in range(12):
        for head in [0, 3, 6, 9, 11]:  # sample 5 heads per layer
            A = build_head_adjacency(mats, layer, head)
            A_bin = threshold_and_binarize(A, threshold=0.15, symmetrize=True)
            try:
                rc   = rich_club_curve(A_bin)
                hier = hierarchical_modularity(A_bin)
                p    = compute_topo_profile(A, label=f"L{layer:02d}H{head:02d}",
                                            density_threshold=0.15, symmetrize=False)
                results["heads"].append({
                    "label":         f"L{layer:02d}H{head:02d}",
                    "layer":         layer,
                    "head":          head,
                    "is_early":      layer < 4,
                    "brain_score":   float(brain_score(p)),
                    "sw_sigma":      float(p.small_worldness_sigma),
                    "rich_club":     {str(k): float(v) for k, v in rc.items()},
                    "hierarchical":  hier,
                })
            except Exception as ex:
                print(f"    L{layer}H{head} failed: {ex}")

    # Compare early vs late rich-club curves
    early_heads = [h for h in results["heads"] if h["is_early"] and h["brain_score"] > 0.5]
    late_heads  = [h for h in results["heads"] if not h["is_early"]]

    def mean_rc_curve(heads):
        if not heads:
            return {}
        all_k = set()
        for h in heads:
            all_k.update(h["rich_club"].keys())
        out = {}
        for k in sorted(all_k):
            vals = [h["rich_club"].get(k, 0) for h in heads]
            out[k] = float(np.mean(vals))
        return out

    results["early_mean_rich_club"] = mean_rc_curve(early_heads)
    results["late_mean_rich_club"]  = mean_rc_curve(late_heads)
    results["early_mean_hier_levels"] = float(np.mean([h["hierarchical"]["n_levels"] for h in early_heads])) if early_heads else 0
    results["late_mean_hier_levels"]  = float(np.mean([h["hierarchical"]["n_levels"] for h in late_heads]))  if late_heads  else 0

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Phase 5: Mechanistic analysis")
    t0 = time.time()

    all_results = {}

    # 5A
    try:
        traj = phase5a_trajectory()
        all_results["5a_trajectory"] = traj
        out5a = os.path.join(RESULTS_DIR, "trajectory.json")
        with open(out5a, "w") as f:
            json.dump(traj, f, indent=2)
        print(f"  5A saved → {out5a}")
    except Exception as e:
        print(f"  5A FAILED: {e}")
        all_results["5a_trajectory"] = {"error": str(e)}

    # 5B
    try:
        ablation = phase5b_ablation()
        all_results["5b_ablation"] = ablation
        out5b = os.path.join(RESULTS_DIR, "ablation.json")
        with open(out5b, "w") as f:
            json.dump(ablation, f, indent=2)
        print(f"\n── Phase 5B Results ─────────────────────────────────")
        print(f"  Baseline bpb:              {ablation['baseline_bpb']:.4f}")
        print(f"  Top (brain-like) ablated:  {ablation['top_brain_like_ablated_bpb']:.4f}  (cost={ablation['top_cost']:+.4f})")
        print(f"  Bottom (non-brain) ablated:{ablation['bottom_brain_like_ablated_bpb']:.4f}  (cost={ablation['bottom_cost']:+.4f})")
        print(f"  Brain-like heads more important: {ablation['brain_like_heads_more_important']}")
        print(f"  5B saved → {out5b}")
    except Exception as e:
        print(f"  5B FAILED: {e}")
        all_results["5b_ablation"] = {"error": str(e)}

    # 5C
    try:
        metrics = phase5c_brain_specific_metrics()
        all_results["5c_brain_specific"] = metrics
        out5c = os.path.join(RESULTS_DIR, "brain_specific_metrics.json")
        with open(out5c, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n── Phase 5C Results ─────────────────────────────────")
        print(f"  Early heads mean hierarchical levels: {metrics.get('early_mean_hier_levels', '?'):.2f}")
        print(f"  Late  heads mean hierarchical levels: {metrics.get('late_mean_hier_levels', '?'):.2f}")
        print(f"  5C saved → {out5c}")
    except Exception as e:
        print(f"  5C FAILED: {e}")
        all_results["5c_brain_specific"] = {"error": str(e)}

    all_results["elapsed_s"] = time.time() - t0
    out_all = os.path.join(RESULTS_DIR, "mechanistic_summary.json")
    with open(out_all, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full summary → {out_all}")
    return all_results


if __name__ == "__main__":
    main()
