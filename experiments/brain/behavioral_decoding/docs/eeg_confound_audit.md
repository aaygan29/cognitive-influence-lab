# EEG neuroprivacy confound audit (ASZED-153) -- result

Reviewer point 3 ("the EEG diagnosis result needs a confound check") run to a
verdict. Script: [`scripts/eeg_confound_audit.py`](../scripts/eeg_confound_audit.py).
Numbers: [`results/eeg_confound_audit/audit.json`](../results/eeg_confound_audit/audit.json).

## Failure mode named (before the test)
The schizophrenia-vs-control decode in `run_eeg_neuroprivacy.py` may reflect
demographic / acquisition confounds rather than a diagnosis-specific spectral
signature. In the 76-subject feature cohort:
- patients are ~10 yr older (mean 39.8 vs 30.0),
- patients have longer minimum-phase recordings (7.8 s vs 5.3 s),
- the 4-phase cohort cut kept 52/76 patients but only 24/77 controls (so the
  label is imbalanced 52:24, majority-baseline accuracy 0.78),
- site is constant (all OAUTHC_IFE_ILESA) and device is 76/78 Contec, so
  device/site are **not** the confound -- demographics and duration are.

**Rescued** would mean: EEG band-power beats a confound-only baseline and
survives confound regression. **Refuted** would mean: the confounds alone decode
diagnosis as well as the EEG, and the decode collapses once they are removed.

## Field-standard controls wired in
Following **Snoek, Miletić & Scholte (2019), *NeuroImage* 184:741-760,
"How to control for confounds in decoding analyses of neuroimaging data"**
([doi:10.1016/j.neuroimage.2018.09.074](https://doi.org/10.1016/j.neuroimage.2018.09.074)):
a **confound-only baseline** (decode the label from age+sex+duration alone) and
**confound regression performed inside every CV fold** (their key result:
residualizing on the full dataset induces negative bias; per-fold removes it).
Subject-level features (phase means concatenated, 380-d), StratifiedKFold,
class-balanced logistic regression, 20 seeds. Caveat carried forward (NeurIPS
2026 residualization audits): residualization is not a guarantee, so it is
reported next to the confound-only baseline and a synthetic sanity check, not
on its own.

## Result (balanced accuracy, mean +/- sd over 20 seeds)

| arm | balanced accuracy | reads as |
| --- | --- | --- |
| original fused pipeline (run_record.json) | 0.705 (boot CI 0.62-0.80) | the published claim |
| **confound-only (age+sex+duration)** | **0.723 +/- 0.025** | nuisances alone match/beat the brain decode |
| EEG raw (this audit) | 0.637 +/- 0.043 | below the confound baseline |
| EEG + confounds | 0.652 +/- 0.042 | EEG adds nothing over confounds |
| **EEG, per-fold confound regression** | **0.542 +/- 0.040** | collapses toward chance once confounds removed |

Supporting:
- **Permutation null** on the raw EEG decode: observed 0.579, null mean 0.496,
  null p95 0.609, **p = 0.135** -- not significant at a fixed split. (The
  original run's own permutation test was degenerate: observed = null mean,
  p = 1.0.)
- **Synthetic sanity check:** a pure-age feature decodes at 0.634 raw and 0.479
  (chance) after per-fold residualization -- confirming the residualizer strips a
  real confound without over-correcting.

## Verdict: REFUTED
The ASZED schizophrenia-vs-control decode does not survive confound control.
Age+sex+duration alone decode diagnosis at least as well as the full multibagged
EEG pipeline (0.723 vs 0.705); adding EEG to those nuisances does not improve on
them (0.652); and regressing them out per fold collapses the decode to chance
(0.542). What looked like "a few minutes of routine EEG carry a decodable
protected health attribute" is, at this cohort and feature set, largely the
cohort's demographic and acquisition imbalance, not a diagnosis-specific brain
signature.

**Caveats.** n = 76 and the cohort is imbalanced, so power is limited and a
genuine small effect could be masked; this is a refutation of the *claim as
stated*, not proof of zero signal. Residualization is not an oracle (NeurIPS
2026), but here three independent controls -- the confound-only baseline, the
per-fold residualized decode, and the permutation test -- agree. A real test of
the neuroprivacy claim would need an age-, sex-, and duration-matched cohort (or
the full 153 subjects with matched controls) and ideally within-subject
diagnosis-independent acquisition.

## What this changes upstream
README part 3 and `docs/eeg_neuroprivacy.md` previously said the EEG arm "may
carry an undisclosed diagnosis, pending a confound check that is not yet clean."
The check is now done: at this cohort it does not survive. The repository's
honest-reporting stance (nulls reported, not hidden) is preserved by stating the
refutation plainly rather than keeping the uncontrolled 0.705.
