# Reproduce

## Environment
Python 3 with numpy, scipy, scikit-learn, matplotlib.

## Simulation (extraction + manipulation)
```bash
python3 scripts/run_fusion_environment.py
```
Reads `results/narps_lambda.npy` (real NARPS loss-aversion, the calibration prior) and
writes `results/fusion_environment.json`. Expected: fused extraction RMSE ~0.082 (< best
single ~0.088), fused manipulation control error ~0.045 (< best single ~0.052).

The NARPS calibration array is derived from the public NARPS dataset (OpenNeuro
ds001734) behaviour; `narps_lambda.npy` is the per-subject loss-aversion estimate.

## Lean proofs
Lean 4 + Mathlib. The math was verified on 2026-09-28 (Lean v4.34.1, `lake build`
succeeded, 8927 jobs, exit 0, no `sorry`). There is no lakefile committed here (it would
pull all of Mathlib). To check locally, drop `proofs/FusionMath.lean` and
`proofs/FusionDetection.lean` into a Mathlib v4.34.1 lake project and:
```bash
lake exe cache get
lake build FusionMath FusionDetection
```
Theorems: `fused_le_min` (inverse-variance variance inequality), `snr_quadrature`
(SNR^2 add in quadrature), `fused_snr_sq_ge_single` (fused >= best single).
