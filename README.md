# Wiring Is Not Weights

**An in-silico instrument and falsifiable framework for what stores identity: wiring vs weights vs function.**

Aayush Gandhi. Code and preprint for the question: *if you replicated a brain's connectome perfectly, would
you have the person? And if not, what would you have to store?*

## The claim (stated honestly)

"Identity" is three distinct things, and the answer differs for each:

1. **Identification** (telling you apart from others) is a property of the **wiring/topology**. The connectome
   is sufficient here. Even a *binarized* connectome fingerprints people (this repo, real data, N=248).
2. **Reconstruction** (regenerating your individual function/responses) is a property of the **weights** —
   specifically, the *function* the weights compute, robust to any weight change that preserves that function
   and destroyed by one that does not.
3. **Constitution** (whether a copy is literally *you*) is left explicitly unresolved.

The defensible thesis, demonstrated in silico and corroborated by prior real-connectome work, is:

> The connectome is the architecture; identity (in the reconstruction sense) is the **wiring-scaffolded
> weight-function, up to its degeneracy class.** The wiring alone is non-identifying of function; the *exact*
> weights are unnecessary (degenerate); storing identity requires the connectome **plus** enough functional
> constraint to pin the right functional-equivalence class.

**Scope note.** The strong claim is established on **synthetic ground truth** (a validated instrument) and is
consistent with the connectome-constrained-network, degeneracy (Marder), and *C. elegans* literature. On real
human data this repo establishes the *identification* axis (topology suffices) and reports an *underpowered
null* on the reconstruction axis. We do **not** claim to have copied a brain. See `paper/manuscript.md`.

## What is here

| File | What it does | Headline result |
|---|---|---|
| `exp01_apparatus_validation.py` | 4-arm ablation + reconstruction identity metric, R=20 random cohorts | apparatus valid; predicted ordering replicates 20/20 |
| `exp02_stress_and_controls.py` | corrected controls + SNR x signal stress sweep | ordering holds across all 6 regimes |
| `exp03a_identity_sufficiency_curve.py` | the graded identity-sufficiency curve (Fig 1) | identity is distributed; exact-degeneracy control flat |
| `exp03b_nonlinear_degeneracy.py` | nonlinear/compensatory degeneracy, the Marder test (Fig 2) | weights change 1.57x with identity intact; matched functional change destroys it (d=33) |
| `exp04_real_pilot_nsd.py` | real NSD N=8 reconstruction pilot | honest null (N=8 underpowered) |
| `exp05_abide_wiring_vs_weights.py` | powered real connectomes, ABIDE N=248 (Fig 3) | binarized topology fingerprints ~ weighted (identification axis) |
| `exp05_hcp_wiring_vs_weights.py` | HCP variant (ready; needs gated HCP PTN data) | not run |

Pre-registration: `PREREG_exp01_apparatus.md`. Full results narrative with caveats: `RESULTS.md`.
Real-data scope/weight-proxy: `SCOPE_exp04_real_mve.md`, `HCP_DATA_INSTRUCTIONS.md`.
Preprint + figures: `paper/`.

## Reproduce

```bash
pip install -r requirements.txt
python3 exp01_apparatus_validation.py        # ~7s
python3 exp02_stress_and_controls.py         # ~19s
python3 exp03a_identity_sufficiency_curve.py # ~2s  (auto-calibrates difficulty)
python3 exp03b_nonlinear_degeneracy.py       # ~1s
python3 exp05_abide_wiring_vs_weights.py     # fetches ~300 ABIDE subjects from the public bucket, ~2-3 min first run
python3 paper/make_figures.py                # regenerate figures from results/
```

Results are written to `results/*.json` (tracked). Figures to `paper/figures/`.

## Data

No data is redistributed here.
- **Synthetic** experiments (exp01-03b) generate their own ground truth; nothing to download.
- **ABIDE** (exp05) is fetched at runtime from the public `s3://fcp-indi` bucket (anonymous, no credentials),
  CC200 ROI timeseries, via the Preprocessed Connectomes Project.
- **NSD** (exp04) uses derived shared-image response matrices (not included).
- **HCP** (exp05_hcp) requires a ConnectomeDB account + Data Use Terms; see `HCP_DATA_INSTRUCTIONS.md`.

## Citation

If this is useful, cite the preprint in `paper/`. License: MIT.
