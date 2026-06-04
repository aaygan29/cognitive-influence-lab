# neuro_topology

# neuro-ai-topology-pipeline

**Brain-Like Topology is an Emergent Property, Not a Performance Driver**
A preregistered, multi-phase experimental pipeline testing whether transformer attention graphs share structural properties with biological neural networks — and whether inducing brain-like topology improves language modeling performance.

> Brain-like topology in early transformer layers is real and partially robust, but it is an emergent byproduct of training, not a causal driver. Regularizing toward it hurts performance (Cohen's *d* = −1.20 across 20 seeds). The original single-run improvement was seed variance.

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the Pipeline](#reproducing-the-pipeline)
- [Phase-by-Phase Guide](#phase-by-phase-guide)
- [Key Results](#key-results)
- [Reading the Paper](#reading-the-paper)
- [Citation](#citation)

---

## Background

This pipeline was built to rigorously test the **brain-like topology performance hypothesis**: if better language models are more brain-like in their representations, and if trained networks converge toward brain-like topology, then *inducing* brain-like topology during training should improve performance.

We test this with five independent experiments:

| Phase | Question | Design |
|---|---|---|
| 1A | Is the SW regularization improvement real? | 20 seeds × 2 conditions |
| 1B | Is the early/late topology split threshold-robust? | 7 density thresholds |
| 2 | Is brain-like topology *specifically* better? | 20 seeds × 6 topology conditions |
| 3 | Does GPT-2 layer 3 align with frontoparietal cortex? | Preregistered RSA, 48 layer×region tests |
| 4 | Do better models have more brain-like early layers? | 6 model families, cross-model correlation |
| 5 | What is the mechanistic role of brain-like heads? | Trajectory analysis + head ablation |

---

## Repository Structure

```
neuro-ai-topology-pipeline/
│
├── README.md                  ← You are here
├── PAPER.md                   ← Full paper with methods, math, results, citations
├── requirements.txt
│
├── pipeline/                  ← Main experimental pipeline
│   ├── train_core.py          ← Core nanoGPT training module (all topology regularizers)
│   ├── phase1a_multiseed.py   ← Multi-seed validation (20 seeds × 2 conditions)
│   ├── phase1b_threshold.py   ← Threshold sensitivity analysis (7 thresholds)
│   ├── phase2_specificity.py  ← Topology specificity ablation (6 conditions × 20 seeds)
│   ├── phase3_real_rsa.py     ← Preregistered RSA vs. brain RSMs
│   ├── phase4_crossmodel.py   ← Cross-model correlation (Pythia + GPT-2 families)
│   ├── phase5_mechanistic.py  ← Training trajectory + head ablation + rich-club
│   ├── final_analysis.py      ← Unified statistical report from all phases
│   └── run_pipeline.py        ← Orchestrator (runs all phases, applies decision gates)
│
├── neuro-ai-topology/         ← Topology extraction utilities
│   ├── attention_topology.py  ← Extract attention graphs from HuggingFace models
│   ├── topo_metrics.py        ← Graph metrics: σ, Q, efficiency, catch22, Betti numbers
│   ├── experiment.py          ← Original Experiment 2 script
│   └── data/
│       ├── train_data/        ← TinyShakespeare (auto-downloaded)
│       └── cached_connectomes/ ← Human FC and C. elegans matrices
│
├── neuro-ai-rsa/              ← RSA pipeline
│   ├── experiment.py          ← Original AutoScientists RSA search
│   ├── autoscientists_task.py ← AutoScientists task definition
│   └── data/
│       └── brain_rsms/        ← Synthetic RSMs (4 regions × 20 subjects)
│
└── results/                   ← All experimental outputs
    ├── phase1a/summary.json
    ├── phase1b/threshold_sensitivity.json
    ├── phase2/specificity_results.json
    ├── phase3/rsa_results.json
    ├── phase4/crossmodel_results.json
    ├── phase5/
    │   ├── trajectory.json
    │   ├── ablation.json
    │   └── brain_specific_metrics.json
    └── PIPELINE_REPORT.txt    ← Auto-generated summary of all phases
```

---

## Installation

```bash
git clone https://github.com/yourusername/neuro-ai-topology-pipeline
cd neuro-ai-topology-pipeline
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0
transformers>=4.35
numpy>=1.24
scipy>=1.11
networkx>=3.0
statsmodels>=0.14
pingouin>=0.5
scikit-learn>=1.3
```

**Device:** The pipeline runs on CPU, CUDA, or Apple MPS (auto-detected). Each training run takes ~30s on Apple M-series. The full pipeline takes ~3–4 hours.

**Models used** (auto-downloaded from HuggingFace on first run):
- `gpt2` (117M)
- `distilgpt2` (82M)
- `EleutherAI/pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-1b`

---

## Reproducing the Pipeline

### Run everything

```bash
cd pipeline
python run_pipeline.py
```

This runs all phases in dependency order and writes `results/PIPELINE_REPORT.txt`.

### Run individual phases

```bash
# Phase 1A: multi-seed validation (~20 min, 40 training runs)
python pipeline/phase1a_multiseed.py

# Phase 1B: threshold sensitivity (~10 min, topology analysis only)
python pipeline/phase1b_threshold.py

# Phase 2: specificity ablation (~50 min, 120 training runs)
python pipeline/phase2_specificity.py

# Phase 3: RSA (~15 min)
python pipeline/phase3_real_rsa.py

# Phase 4: cross-model correlation (~20 min, downloads cached models)
python pipeline/phase4_crossmodel.py

# Phase 5: mechanistic analysis (~30 min)
python pipeline/phase5_mechanistic.py

# Final report (instantaneous, reads cached results)
python pipeline/final_analysis.py
```

### Skip completed phases

Each phase caches its result JSON. Re-running `run_pipeline.py` will load cached results and skip re-computation. To force a re-run, delete the relevant JSON:

```bash
rm results/phase1a/summary.json  # will re-run Phase 1A
python run_pipeline.py
```

---

## Phase-by-Phase Guide

### Phase 1A — Multi-seed Validation

Tests whether the λ=0.10 small-world improvement is real or seed variance.

**What it does:**
- Trains nanoGPT (0.81M params, TinyShakespeare, 500 iters) on 20 seeds under two conditions: unregularized baseline and small-world regularization (λ=0.10)
- Computes Welch t-test, Cohen's d, bootstrap 95% CI

**Key parameters** (edit in `phase1a_multiseed.py`):
```python
N_SEEDS = 20        # seeds per condition
SEEDS   = range(42, 62)
```

**Expected runtime:** ~20 min on M-series Mac

---

### Phase 1B — Threshold Sensitivity

Tests whether the early/late topological split in GPT-2 is robust to density threshold choice.

**What it does:**
- Extracts GPT-2 attention weight matrices across 30 sentences
- Runs full topology analysis (σ, Q, efficiency, brain-similarity) at 5%, 10%, 15%, 20%, 25%, 30%, and weighted
- Mann-Whitney U (early vs. late layers) at each threshold
- Kendall's τ between adjacent-threshold head rankings

**Expected runtime:** ~10 min

---

### Phase 2 — Topology Specificity Ablation

The crux experiment: is small-world specifically better, or does any regularization of the same strength perform similarly?

**What it does:**
- 6 conditions × 20 seeds = 120 training runs
- One-way ANOVA + Holm-Bonferroni corrected pairwise Welch t-tests

**Conditions:**

| Condition | Regularizer | Target topology |
|---|---|---|
| `none` | None (λ=0) | Unrestricted |
| `small_world` | Entropy deviation + locality | Small-world (σ≈2.7, Q≈0.6) |
| `random_graph` | Entropy maximization | Random (σ≈1) |
| `scale_free` | In-degree variance | Hub-and-spoke |
| `lattice` | Locality only | High-C, high-L |
| `degenerate` | Top-k=2 sparsity | Concentrated |

To add a new topology condition, add a function in `train_core.py`:
```python
def topo_loss_your_condition(attn, lam, seq):
    # attn: (B, H, T, T)
    # return a scalar tensor
    ...
```
Then add it to the `TOPO_FNS` dict and the `CONDITIONS` list in `phase2_specificity.py`.

**Expected runtime:** ~50 min

---

### Phase 3 — RSA

Preregistered confirmatory test: GPT-2 layer 3 first-token representations vs. frontoparietal RSMs.

**What it does:**
- Extracts GPT-2 representations at all 12 layers
- Builds RSMs (negated Euclidean distance)
- Regresses out nuisance variables (length, word count, surprisal)
- Permutation test (5,000 permutations), bootstrap CI, noise ceiling

**To use real fMRI data** (instead of synthetic RSMs):
1. Download NeuralBench from [OSF](https://osf.io/anq35/) (Schrimpf et al., 2021)
2. Place RSM `.npy` files in `neuro-ai-rsa/data/brain_rsms/`
3. Files should be named `{region}_subjects.npy` with shape `(n_subjects, n_stimuli, n_stimuli)`

**Expected runtime:** ~15 min

---

### Phase 4 — Cross-Model Correlation

Tests whether better models have more brain-like early-layer topology.

**What it does:**
- Extracts attention graphs from 6 cached HuggingFace models
- Computes early/late layer brain-similarity scores
- Spearman ρ with log-perplexity, bootstrap CI, partial correlation controlling for log(params)

**Adding more models:**
```python
# In phase4_crossmodel.py
CACHED_MODELS.append(
    ("your-model", "org/model-id", param_count_M, published_ppl)
)
```

**Expected runtime:** ~20 min (with cached models)

---

### Phase 5 — Mechanistic Analysis

Three sub-experiments examining *why* brain-like topology is present but not useful.

**5A — Trajectory:** Tracks val_bpb and brain-similarity score every 100 steps for baseline vs. SW (3 seeds each). Shows that topology emerges naturally in baseline training.

**5B — Ablation:** Zeros the contribution of the top-10 and bottom-10 brain-similar GPT-2 heads. The 8.6× functional load ratio shows brain-like heads are redundant.

**5C — Rich-club / Hierarchical modularity:** Brain-specific topology metrics beyond small-worldness. Early layers show 4× stronger rich-club at k=1.

**Expected runtime:** ~30 min

---

## Key Results

### What the pipeline found

```
Phase 1A: Small-world regularization HURTS performance
  Baseline:    4.3931 ± 0.0547 bpb
  Small-world: 4.4592 ± 0.0553 bpb
  Δ = +0.066  d = -1.20  p = 0.9997 (one-tailed H1: SW < baseline)
  → Original single-run result was seed variance

Phase 1B: Early/late topological split is PARTIAL
  Significant at 4/7 density thresholds
  Layer 3 identified as peak at 0/7 thresholds
  → Directional claim holds at low/moderate densities; specific head IDs are unstable

Phase 2: No topology regularizer beats baseline
  Ranking: none > lattice > small_world > degenerate > random_graph > scale_free
  ANOVA F(5,106) = 64.9, p < 0.0001
  SW vs. baseline: Δ = +0.066, p_corr = 0.003 (*** worse)
  → Brain-like topology is not specifically beneficial

Phase 3: RSA alignment fully null
  L3 × frontoparietal: ρ = 0.031, p = 0.484
  0/48 layer × region tests significant after HB correction
  → Representational alignment claim unsupported

Phase 4: Correlation present but scale-confounded
  Early layers ↔ log-PPL: ρ = -0.829, p = 0.042 (raw)
  Partial (ctrl log-params): ρ = -0.673, p = 0.143 (NS)
  → Better models are more brain-like, but partly because they're larger

Phase 5B: Brain-like heads are 8.6× less important
  Brain-like ablated: +0.031 bpb
  Non-brain-like ablated: +0.264 bpb
  → The concentrated deep heads do the functional work; early distributed heads are redundant
```

### The coherent story

Brain-like topology emerges naturally during transformer training (Phase 5A: +0.031 in baseline). Better models have more of it (Phase 4). But it is a **byproduct** of the training dynamics, not a cause of good performance. The early layers that are topologically brain-like are functionally the most redundant (Phase 5B). Imposing the topology externally constrains the model away from its optimal operating point.

---

## Reading the Paper

The full paper is at [`PAPER.md`](PAPER.md). It includes:

- Complete mathematical formulation of all topology metrics and regularizers
- All experimental results with effect sizes and confidence intervals
- Discussion of why the original hypothesis was wrong and what the correct interpretation is
- Full reference list (27 citations)

---

## Citation

```bibtex
@misc{gandhi2026brainlike,
  title   = {Brain-Like Topology is an Emergent Property, Not a Performance Driver:
             A Preregistered Multi-Phase Investigation of Transformer Attention
             Structure and Neural Alignment},
  author  = {Gandhi, Aayush},
  year    = {2026},
  month   = {June},
  note    = {Preprint},
  url     = {https://github.com/yourusername/neuro-ai-topology-pipeline}
}
```

---

## License

MIT. Use freely; cite if publishing.
