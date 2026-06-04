# Digital Brains: Geometry-Aware Encoding Models for Individual Neural Organization

**Aayush Gandhi**

> *"A digital brain that captures only amplitude without geometry is an impoverished proxy: correct about how loudly a person's brain speaks, but not about what it is saying."*

---

## Overview

This repository contains the full experimental pipeline for the paper:

**"Digital Brains: Geometry-Aware Encoding Models Capture Individual Neural Organization in Human Visual Cortex"**

We introduce and validate the concept of a **digital brain**: a subject-specific encoding model that maps visual stimuli to predicted 7T fMRI responses, evaluated across five levels of individual-specificity validation.

### Key Results (N=4 subjects, Algonauts 2023 NSD, 260 shared images)

| Metric | Result |
|--------|--------|
| Encoding accuracy | r = 0.105 to 0.264 (median Pearson r across 25 ROIs, all positive) |
| RSA self-advantage | Delta = +0.001 to +0.121 (positive in all 25 ROIs; geometry-aware > amplitude-only) |
| Subject fingerprinting | **21/25 ROIs significant** (p < 0.05, chance = 25%) |
| 100% accuracy ROIs | FFA-1 (p=0.041), OFA (p=0.044), PPA (p=0.032), V1v (p=0.048), V1d (p=0.043), V2v (p=0.042), V2d (p=0.033), V3v (p=0.036), VWFA-1 (p=0.043), VWFA-2 (p=0.037), midlateral (p=0.039), midventral (p=0.034), ventral (p=0.034) |
| 75% accuracy ROIs | EBA (p=0.032), FBA-2 (p=0.026), OPA (p=0.039), V3d (p=0.040), hV4 (p=0.039), lateral (p=0.028), midparietal (p=0.043), parietal (p=0.033) |
| Counterfactual r | 0.109 to 0.219 across all 25 ROIs and 6 subject pairs (all positive, held-out stimuli) |

**Scaling path:** N=8 subjects yields p < 5x10^-8 under equivalent accuracy. No code changes required. See `scripts/build_algonauts_dataset.py`.

---

## Core Contribution: The RSA Paradox and Representational Asymmetry

Standard ridge regression encoding models exhibit an **RSA paradox**: they fingerprint subjects correctly via amplitude matching, yet sometimes match *other* subjects' representational geometry better than their own. We resolve this with a dual-objective loss:

```
L = (alpha) * L_MSE  +  (beta) * L_RDM  +  (gamma) * L_rank
       amplitude          geometry             rank-order
                          magnitude            geometry
```

With alpha=1.0, beta=0.6, gamma=0.3, our geometry-aware encoder achieves positive RSA self-advantage in **all 25 tested ROIs**.

However, we reveal a **representational asymmetry**: amplitude-level individuation (fingerprinting) is readily achievable with current CLIP features, while geometry-level individuation (full RDM matching) remains partially elusive. This asymmetry points to CLIP's population-level training as the architectural ceiling, not the training objective. Brain-optimized encoders (Brain-JEPA, MindEye) are the natural next step.

---

## Dataset

**Algonauts 2023 Challenge** (Natural Scenes Dataset subset):
- 7T fMRI, z-scored within sessions, averaged across 3 image repeats
- 4 subjects: subj05, subj06, subj07, subj08 (pilot; pipeline supports all 8)
- 260 shared NSD images (present in all subjects' training sets)
- 25 ROIs across 6 functional classes via challenge-space vertex masks
- Authoritative noise ceilings from test-retest reliability

**Noise ceilings (median LH):** subj05=0.473, subj07=0.296, subj06=0.243, subj08=0.139

Download: https://algonautsproject.com/2023

---

## Architecture

### Digital Brain Model

For each subject x ROI, a subject-specific MLP:
```
Input (1024-dim CLIP ViT-L/14 CLS token)
  -> Linear(512) -> LayerNorm -> GELU -> Dropout(0.3)
  -> Linear(256) -> LayerNorm -> GELU -> Dropout(0.3)
  -> Linear(100)  [voxel PCA components]
  -> PCA inverse  [reconstructed voxel space]
```

### Dual-Objective Loss

```python
L_total = 1.0 * L_MSE      # amplitude fidelity
        + 0.6 * L_RDM      # RDM Frobenius distance (geometry magnitude)
        + 0.3 * L_rank     # differentiable rank correlation (geometry order)
```

`L_rank` uses pairwise sigmoid rank approximation (Blondel et al., 2020) with tau=0.05: fully differentiable, encourages ordinal preservation of pairwise representational distances.

**Training:** AdamW, lr=3x10^-4, weight decay=10^-4, cosine annealing, 200 epochs, batch=64.
**Split:** 208 train / 52 test (fixed seed, identical across all subjects).

---

## Evaluation: Five-Level Validation Protocol

| Level | Metric | What it tests |
|-------|--------|---------------|
| 1 | Encoding accuracy (median Pearson r) | Stimulus-driven amplitude prediction |
| 2 | RSA self-advantage (Delta) | Individual geometry vs. population geometry |
| 3 | RSA identity matrix diagonal dominance | Cross-subject geometric specificity |
| 4 | Subject fingerprinting (permutation test) | Biometric identification from predictions |
| 5 | Counterfactual consistency (r) | Generalization of subject differences to novel stimuli |

---

## Results Provenance

The authoritative results file is `results/algonauts2023/all_results_N4.json`.

An earlier synthetic-data pilot (`results/archive/synthetic_pilot_DO_NOT_CITE/`) is archived with a provenance note. That file shows near-chance encoding (r ~ 0.006) with trivially perfect fingerprinting because subjects were defined by different random generators, not biology. It is not cited in the paper and must not be used for comparison.

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
│   ├── build_algonauts_dataset.py  # Dataset preparation (supports N=1-8)
│   ├── run_algonauts_experiment.py # Full 5-level experiment
│   ├── compare_architectures.py    # Ridge vs. geometry-aware comparison
│   └── extract_bold5000_features.py
├── results/
│   ├── algonauts2023/
│   │   ├── figures/                # main_results_N4.pdf + RSA matrices
│   │   ├── models/                 # Cached geometry-aware digital brains
│   │   └── all_results_N4.json     # AUTHORITATIVE results
│   ├── comparison/                 # Ridge vs. geometry-aware figures
│   └── archive/
│       └── synthetic_pilot_DO_NOT_CITE/   # Synthetic data artifact, not for citation
└── Digital_Brain_Paper.pdf
```

---

## Reproduction

```bash
# 1. Install dependencies
pip install torch transformers scikit-learn scipy matplotlib seaborn numpy

# 2. Download Algonauts 2023 data and place under:
#    Train Data/subj0X/training_split/training_fmri/
#    Test Data/subj0X/test_split/test_fmri/

# 3. Build dataset + extract features
python scripts/build_algonauts_dataset.py

# 4. Run full experiment (all 5 levels, 25 ROIs)
python scripts/run_algonauts_experiment.py

# 5. Architecture comparison (ridge vs. geometry-aware)
python scripts/compare_architectures.py
```

**To scale to N=8:** ensure all 8 subjects' data is present. The pipeline detects available subjects automatically and requires no code changes.

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
- Blondel et al. (2020). Fast differentiable sorting and ranking. *ICML*.
- Gifford et al. (2023). The Algonauts Project 2023. *arXiv:2301.03198*.
- Kriegeskorte et al. (2008). RSA. *Frontiers in Systems Neuroscience*.
- Radford et al. (2021). CLIP. *ICML*.
