# Attributing neural-manifold topology to geometric vs temporal generators

Persistent homology of grid-cell population activity returns a torus (H1 rank 2, H2 rank 1),
and two 2025-2026 readings of that fact appear to disagree:

- **Gardner et al. (2022, Nature)** treat the torus as the intrinsic geometry of the spatial
  code, a continuous-attractor manifold.
- **di Sarra et al. (2025, PLOS Comput. Biol.)** show the *empirically recovered* torus depends
  on neural oscillations: hexagonal spatial tuning alone is "not sufficient," and jittering spike
  times by ~100-500 ms (theta/eta periods) collapses the toroidal barcode.

A single persistence diagram cannot say which generator a homology class comes from. This repo
proposes a **surrogate-based attribution**: manipulate the geometric and temporal generators
independently in a controlled grid-cell simulation, and measure how each persistent-homology
feature responds. The goal is reconciliation (deciding what each finding constrains), not
debunking either paper.

## Method

Controlled model (one grid module, N=500 cells, realistic 0.34 m/s trajectory over 120 s):

```
rate_i(t) = base * grid_i(x(t)) * (1 + m * osc(t)) ,  spikes ~ Poisson
```

- **Geometric generator G** = the phase tiling of the torus (diverse grid phases).
- **Temporal generator T** = oscillatory rate modulation osc(t) = 0.6 cos(2 pi 8 t) + 0.4 cos(2 pi 4 t).

Surrogates:

- **destroy T**: set m = 0 (no oscillation), or jitter spike times by Delta t (di Sarra's manipulation).
- **destroy G**: collapse all grid phases to one value (no torus tiling), oscillation kept.

Readout (ripser, H1 and H2, scale-normalized point clouds):

- **gap significance**: gapH1 = 2nd/3rd longest H1 bar; gapH2 = 1st/2nd H2 bar (> 1 means torus-like).
- **bottleneck toroidality Gamma_k**: bottleneck distance from a diagram to the empirical
  no-structure null (the destroy-G diagrams). Large Gamma means far from "no torus."

The noiseless rate manifold gives a textbook torus (H1 = [4.19, 4.10], H2 = [2.96]); see
`neural_manifold/torus_validation.py`.

## Results (n = 12 seeds, mean +/- sem)

| condition                     | Gamma H1 (loops) | gapH1 | Mann-Whitney vs destroy-G |
|-------------------------------|------------------|-------|---------------------------|
| REAL (geometry + oscillation) | 0.436            | 1.59  | -                         |
| destroy T (no oscillation)    | 0.413            | 1.24  | Gamma H1 n.s. (approx REAL)|
| destroy G (no phase tiling)   | 0.140            | 1.25  | Gamma H1 p < 0.0001; Gamma H2 p < 0.001 |

Jitter sweep, Gamma H1: 0.44 (0 ms) -> 0.35 (100) -> 0.17 (300) -> 0.15 (500).

Figure: `neural_manifold/results/attribution_stats.png`.

### What holds up

1. **Geometry is necessary, and the effect is highly significant.** Destroying the phase tiling
   collapses Gamma H1 from 0.44 to 0.14 (p < 0.0001) and Gamma H2 likewise (p < 0.001). The torus
   is fundamentally geometric. This supports Gardner et al., now with statistics.

2. **Rate oscillation is not necessary.** Removing it (m = 0) leaves Gamma H1 unchanged
   (0.41 vs 0.44, n.s.). In a rate-based model, di Sarra et al.'s "oscillations are required" does
   not reproduce.

3. **Spike-time jitter degrades the torus** (Gamma H1 0.44 -> 0.15), but with no critical timescale,
   and since removing the oscillation itself did not hurt, this is generic position-code smearing
   (a 300 ms jitter at 0.34 m/s misattributes a spike to a position roughly 0.1 m away), not
   oscillation-specificity.

4. **Metric choice matters.** The bottleneck Gamma detects the geometry effect at p < 0.0001 where
   the gap-ratio could not (p = 0.12). Toroidality readouts based on a single bar-length ratio are
   underpowered; a null-referenced bottleneck distance is not.

### What this does and does not claim

- It does **not** reproduce di Sarra et al.'s oscillation-dependence, and it does not refute it.
  The result is that a *rate*-level oscillation cannot be the mechanism, because the smoothing that
  recovers Gardner's torus averages a common-mode rate oscillation away.
- **Localization / prediction:** if di Sarra et al.'s oscillation-dependence is real, it must act
  through **spike timing / phase coding**, not rate modulation. Testing this requires a phase-coded
  spike model (planned; see below), and is the concrete next experiment.
- This is a controlled in-silico model, not an analysis of the Gardner recordings. Magnitudes and
  the exact null depend on the model; the qualitative attribution (geometry necessary; rate
  oscillation not) is the transferable claim.

## Reproduce

```bash
pip install numpy scipy networkx gudhi ripser matplotlib
cd neural_manifold
python3 torus_validation.py      # textbook torus from the noiseless rate manifold
python3 attribution_study.py     # n=12 study -> results/attribution_stats.png, results/stats_results.json (~8 min)
```

## Planned next step

Add a phase-coded spike model (theta phase precession), so oscillation carries topological
information beyond rate, and re-run the attribution to test the localization prediction directly.

## Sources

- R. J. Gardner et al., "Toroidal topology of population activity in grid cells," Nature 602,
  123-128 (2022). doi:10.1038/s41586-021-04268-7
- G. di Sarra et al., "The role of oscillations in grid cells' toroidal topology," PLOS Comput.
  Biol. (2025); arXiv:2501.19262.
- Tools: ripser, GUDHI, SciPy, NumPy, Matplotlib.

## archive/

`archive/cyclomatic-vs-clique/` holds an earlier, unrelated study (graph cyclomatic number vs
clique-complex b1). Its own README documents its scope and the corrections made to it. It is kept
for provenance and is not part of the neural-manifold work above.
