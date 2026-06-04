# Digital Brains: Geometry-Aware Encoding Models for Individual Neural Organization

**Aayush Gandhi**

> *"A digital brain that captures only amplitude without geometry is an impoverished proxy — correct about how loudly a person's brain speaks, but not about what it is saying."*

---

## Overview

This repository contains the full experimental pipeline for the paper:

**"Digital Brains: Geometry-Aware Encoding Models Capture Individual Neural Organization in Human Visual Cortex"**

We introduce and validate the concept of a **digital brain** — a subject-specific encoding model that maps visual stimuli to predicted 7T fMRI responses — and evaluate whether such models capture the unique representational organization of individual subjects across five levels of validation.

### Key Results (N=4 subjects, Algonauts 2023 NSD, 260 shared images)

| Metric | Result |
|--------|--------|
| Encoding accuracy | r = 0.183–0.258 (median Pearson r, across ROIs) |
| RSA self-advantage | Δ = +0.043 to +0.094 (all positive; geometry-aware > amplitude-only) |
| Subject fingerprinting | 6/10 ROIs significant (p < 0.05, chance = 25%) |
| — 100% accuracy | FFA-1 (p=0.041), OFA (p=0.044), PPA (p=0.032), V1d (p=0.043) |
| — 75% accuracy | EBA (p=0.032), FBA-2 (p=0.026), OPA (p=0.039) |
| Counterfactual r | 0.159–0.219 across 6 subject pairs (all positive) |

**Scaling path:** N=8 subjects yields p < 5×10⁻⁸ under equivalent accuracy. No code changes required — see `scripts/build_algonauts_dataset.py`.

---

## Core Contribution: The RSA Paradox and Representational Asymmetry

Standard ridge regression encoding models exhibit an **RSA paradox**: they fingerprint subjects correctly via amplitude matching, yet sometimes match *other* subjects' representational geometry better than their own. We resolve this with a dual-objective loss:

```
L = α·L_MSE  +  β·L_RDM  +  γ·L_rank
    (amplitude)  (geometry   (rank-order
                  magnitude)  geometry)
```

With α=1.0, β=0.6, γ=0.3, our geometry-aware encoder achieves positive RSA self-advantage in **all 8 tested ROIs** — a complete resolution of the paradox at the directional level.

However, we reveal a **representational asymmetry**: amplitude-level individuation (fingerprinting) is readily achievable with current CLIP features, while geometry-level individuation (full RDM matching) remains partially elusive. This asymmetry points to CLIP's population-level training as the architectural ceiling — not the training objective. Brain-optimized encoders (Brain-JEPA, MindEye) are the natural next step.

---

## Dataset

**Algonauts 2023 Challenge** — Natural Scenes Dataset (NSD) subset:
- 7T fMRI, z-scored within sessions, averaged across 3 image repeats
- 4 subjects: subj05, subj06, subj07, subj08 (pilot; pipeline supports all 8)
- 260 shared NSD images (present in all subjects' training sets)
- 25 ROIs across 6 functional classes
- Authoritative noise ceilings from test-retest reliability

**Noise ceilings (median LH):** subj05=0.473, subj07=0.296, subj06=0.243, subj08=0.139

Download: https://algonautsproject.com/2023

---

## Architecture

### Digital Brain Model

For each subject × ROI, a subject-specific MLP:
```
Input (1024-dim CLIP ViT-L/14 CLS token)
  → Linear(512) → LayerNorm → GELU → Dropout(0.3)
  → Linear(256) → LayerNorm → GELU → Dropout(0.3)
  → Linear(100)  [voxel PCA components]
  → PCA⁻¹       [reconstructed voxel space]
```

### Dual-Objective Loss

```python
L_total = 1.0 * L_MSE      # amplitude fidelity
        + 0.6 * L_RDM      # RDM Frobenius distance (geometry magnitude)
        + 0.3 * L_rank     # differentiable rank correlation (geometry order)
```

`L_rank` uses pairwise sigmoid rank approximation (Blondel et al., 2020) with τ=0.05 — fully differentiable, encourages ordinal preservation of pairwise distances.

**Training:** AdamW, lr=3×10⁻⁴, weight decay=10⁻⁴, cosine annealing, 200 epochs, batch=64.  
**Split:** 208 train / 52 test (fixed seed, identical across all subjects).

---

## Evaluation: Five-Level Validation Protocol

| Level | Metric | What it tests |
|-------|--------|---------------|
| 1 | Encoding accuracy (median Pearson r) | Stimulus-driven amplitude prediction |
| 2 | RSA self-advantage (Δ) | Individual geometry vs. population |
| 3 | RSA identity matrix diagonal dominance | Cross-subject geometric specificity |
| 4 | Subject fingerprinting (permutation test) | Biometric identification from predictions |
| 5 | Counterfactual consistency (r) | Generalization of subject differences to novel stimuli |

---

## Project Structure

```
digital-brain/
├── src/
│   ├── geometry_aware_encoder.py   # Dual-objective MLP + differentiable rank loss
│   ├── evaluation.py               # All 5 validation levels
│   ├── visualization.py            # Publication figures
│   └── data_loader.py
├── scripts/
│   ├── build_algonauts_dataset.py  # Dataset preparation (supports N=1–8)
│   ├── run_algonauts_experiment.py # Full 5-level experiment
│   ├── compare_architectures.py    # Ridge vs. geometry-aware comparison
│   └── extract_bold5000_features.py
├── results/
│   ├── algonauts2023/
│   │   ├── figures/                # main_results_N4.pdf + RSA matrices
│   │   ├── models/                 # Cached geometry-aware digital brains
│   │   └── all_results_N4.json     # Full numerical results
│   └── comparison/                 # Ridge vs. geometry-aware figures
└── Digital_Brain_Paper.pdf         # Full paper
```

---

## Reproduction

```bash
# 1. Install dependencies
pip install torch transformers scikit-learn scipy matplotlib seaborn numpy

# 2. Download Algonauts 2023 data → place in:
#    ~/Desktop/Delete\ Later/Train\ Data/subj0X/
#    ~/Desktop/Delete\ Later/Test\ Data/subj0X/

# 3. Build dataset + extract features
python scripts/build_algonauts_dataset.py

# 4. Run full experiment (all 5 levels)
python scripts/run_algonauts_experiment.py

# 5. Architecture comparison (ridge vs. geometry-aware)
python scripts/compare_architectures.py
```

**To scale to N=8:** ensure all 8 subjects' data is present in the data directory. The pipeline detects available subjects automatically.

---

## Citation

```bibtex
@article{gandhi2026digitalbrain,
  title={Digital Brains: Geometry-Aware Encoding Models Capture Individual
         Neural Organization in Human Visual Cortex},
  author={Gandhi, Aayush},
  year={2026}
}
```

---

## References

- Allen et al. (2022). A massive 7T fMRI dataset. *Nature Neuroscience*.
- Gifford et al. (2023). The Algonauts Project 2023. *arXiv:2301.03198*.
- Radford et al. (2021). CLIP. *ICML*.
- Kriegeskorte et al. (2008). RSA. *Frontiers in Systems Neuroscience*.
- Blondel et al. (2020). Fast differentiable sorting and ranking. *ICML*.
