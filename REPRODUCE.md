# Reproduction guide

Status as of 2026-09-28. This file records exactly what reproduces from the
data shipped in this repo and what needs a re-download, so results are not
overstated.

## Environment
- Python 3 with `torch`, `transformers`, `scikit-learn`, `scipy`, `numpy`,
  `matplotlib` (tested with torch 2.8, transformers 4.57).

## BOLD5000 arm — fully reproducible from cached data (VERIFIED)
The processed per-ROI betas (`data/processed/CSI1..CSI4/`) and CLIP features
(`data/features/bold5000_clip_features.npy`) are in the repo.

```bash
python3 scripts/run_bold5000_experiment.py
```

Re-running on 2026-09-28 reproduced the cached `results/bold5000/all_results.json`
to within ~1e-6 (floating-point nondeterminism only). Encoding r 0.10-0.22 across
5 ROIs, RSA self-advantage positive in all 5 ROIs, subject fingerprinting 100%
(p=0.034) in all 5 ROIs, counterfactual r 0.07-0.12.

## NSD / Algonauts 2023 arm — needs the training bundle
The 2026-06 disk cleanup pruned the raw NSD downloads and the per-subject
image->fMRI alignment (`shared_image_indices`) and full-stimulus CLIP features.
The authoritative results survive at `results/algonauts2023/all_results_N4.json`
and its summary statistics were re-verified on 2026-09-28, but the encoders
cannot be re-run from the cache alone.

To re-run from scratch, download the Algonauts 2023 training data (per-subject
image zips + `training_split/training_fmri/` + `roi_masks/`) from
https://algonautsproject.com/2023, then:

```bash
export ALGONAUTS_RAW="/path/to/train data"   # folder holding subj0X zips + fmri
python3 scripts/run_algonauts_experiment.py
```

The runner auto-detects available subjects and scales from N=1 to N=8 with no
code change. The post-challenge test release (`test_split/` with `test_fmri`,
`test_images`, `noise_ceiling`) alone is **not** sufficient for the per-ROI
analysis because it ships no `roi_masks`.

## Note on prior numbers
An earlier `README.md` results table overstated encoding accuracy (claimed median
r 0.105-0.264, all positive). The true per-subject medians from
`all_results_N4.json` are S1=0.375, S2=0.189, S3=0.015, S4=0.149, and RSA
self-advantage is positive in all 25 ROIs only on the subject-averaged mean
(66/100 at the subject x ROI level). The README and paper now use the verified
values.
