# spikeprint — Preregistration (v0.1, draft)

**Title:** Does a content-level Cognitive Sovereignty Index (CSI) predict behavioral choice
distortion? A secondary-data, in-silico construct-validation study.

**Author:** Aayush Gandhi (ORCID 0009-0003-4649-0367)
**Date drafted:** 2026-06-08 · **Status:** DRAFT — not yet timestamped on OSF.
**OSF registration DOI:** _to be minted before any model is fit on outcome data._

> This document is written to be registered **before** any analysis touches an outcome
> variable. Its purpose is to fix hypotheses, datasets, analyses, and **kill criteria** in
> advance, so that nothing below can be silently changed after seeing results
> (Simmons, Nelson, & Simonsohn, 2011). Deviations will be logged in `docs/DEVIATIONS.md`.

---

## 1. What we claim, and what we do not

This study makes **one** kind of claim, and it is behavioral:

- **Ground truth (the thing we predict):** *choice distortion* — a measured shift in an
  observable decision (accept/reject a gamble, donation amount, opinion change), produced by
  a stimulus, relative to a control or to a reflective baseline.
- **Instrument under test:** the **Cognitive Sovereignty Index (CSI)** — a score computed
  **from the content/stimulus**, not from any person's brain. *(The name is a proper-noun
  instrument label inherited from prior work; it does not assert that the metric measures
  "sovereignty," harm, or any normative property. CSI's meaning is fixed solely by the
  validation tests in §3 — nothing else.)*
- **The brain-encoder** (TRIBE / `neurosignal`) is used only as a *theory-motivated,
  population-average feature generator*. Its activations are inputs to CSI, **not evidence of
  anything by themselves.**

**We do NOT claim** (and the manuscript will state this explicitly):

1. that any signal "detects manipulation." *Manipulation* is a normative, intentional concept;
   it is treated here as an unobserved hypothesis, never as a measured quantity.
2. that we read or track any individual's neural state. Neural representations are individually
   variable; we make no individual-level neural claim. All inference is at the level of the
   **fixed stimulus** and the **observed choice**, both of which are comparable across people.
3. that a high CSI implies harm, intent, or coercion. CSI is a predictor of *choice shift*,
   nothing more, unless and until it survives the validity tests below.

Rationale for grounding in choice rather than affect/neural signal: affective or neural
"appeal" is neither necessary nor sufficient for manipulation, and emotion-from-neural-signal
decoding is known to be fragile and confounded (see Limitations, §9). Choice is observable,
operational, and on the same scale across subjects (Tversky & Kahneman, 1981).

## 2. Background (each major claim cited)

- Reward/value and conflict/control circuitry can bias value-based choice; nucleus
  accumbens/mPFC reward activity and anterior-insula activity have *predicted* purchasing in
  controlled tasks (Knutson, Rick, Wimmer, Prelec, & Loewenstein, 2007), and subjective value
  has a convergent cortical substrate (Bartra, McGuire, & Kable, 2013). Conflict/control is
  associated with ACC/cognitive-control signals (Botvinick, Braver, Barch, Carter, & Cohen,
  2001).
- Framing and presentation systematically distort choice without changing payoffs
  (Tversky & Kahneman, 1981). Large-scale human choice data make such distortions
  model-testable at scale (Peterson, Bourgin, Agrawal, Reichman, & Griffiths, 2021).
- Persuasive language produces *measurable* outcome change in naturalistic corpora:
  opinion change on r/ChangeMyView (Tan, Niculae, Danescu-Niculescu-Mizil, & Lee, 2016) and
  donation behavior in goal-directed dialogue (Wang et al., 2019).
- Construct validation (does CSI measure what it claims?) follows the convergent/discriminant
  framework of Cronbach & Meehl (1955).
- Where we use language models as synthetic respondents, we follow and *bound* a method with
  documented limits (Horton, 2023; Aher, Arriaga, & Kalai, 2023; Argyle et al., 2023;
  Binz & Schulz, 2023); synthetic data are admitted only after passing the human-calibration
  gate in §6.

## 3. Hypotheses and kill criteria

All tests use **held-out** data (train/val/test fixed in §5) and are reported with effect
sizes and 95% bootstrap CIs. Primary inference is preregistered as **confirmatory**; anything
else is labeled exploratory.

| ID | Hypothesis | Directional prediction | **Kill criterion (preregistered)** |
|----|------------|------------------------|------------------------------------|
| **H1 Predictive** | CSI(stimulus) predicts choice distortion | β > 0; out-of-sample AUC/ρ above chance | If 95% CI of the effect includes the null on the held-out test set → H1 **fails**, reported as such. |
| **H2 Incremental** | CSI adds predictive value over simple text baselines | ΔAUC > 0 vs. {sentiment, length, readability, TF-IDF+logistic} | If CSI does **not** beat the best baseline (CI of ΔAUC includes 0) → CSI is **not** incrementally valid. |
| **H3 Discriminant** | CSI is not merely sentiment/arousal/length | Effect survives covarying out sentiment, arousal, length, readability | If the H1 effect disappears after these covariates → CSI is **confounded**, reported as such. |
| **H4 Convergent** | CSI aligns with established persuasion signals | ρ > 0 with Persuasion-for-Good strategy annotations / CMV winning-argument features | If no positive association → convergent validity **not supported**. |

**Power / minimum detectable effect (registered).** α = .05 two-sided; target power = .80.
Using the Fisher-z approximation (`scripts/power.py`), the minimum detectable correlation is
**|ρ| ≈ 0.089 at n = 1,000** and **|ρ| ≈ 0.025 at the choices13k problem count (~13,000)**.
Effects smaller than the registered MDES are treated as null regardless of p-value. The
confirmatory test set is evaluated **exactly once**. (Per-dataset n and the mixed-model cluster
structure / assumed ICC are recorded in §6 and finalized before the OSF timestamp.)

A result in which CSI fails H2 or H3 is a **publishable negative result** and will be reported
without rescue analyses.

## 4. Datasets (secondary, public, no new subjects)

No human or animal subjects are collected or run. All data are pre-existing, public, and
de-identified at source. Per-dataset provenance, license, version, and access date are in
`docs/DATA.md` (datasheet; Gebru et al., 2021). Summary:

| Dataset | Stimulus | Outcome (choice distortion) | License (verify) | Role |
|---|---|---|---|---|
| **NARPS `ds001734`** (Botvinik-Nezer et al., 2020) | gain/loss framing | accept/reject (+fMRI optional) | **CC0 (confirmed)** | **primary confirmatory (H1/H2)** |
| choices13k (Peterson et al., 2021) | gamble description | human choice rate (~1M choices) | **no license — deferred** | replication; **separate paper, pending author permission** |
| IBL decision task (IBL et al., 2021) | 0.2/0.8 prior block | prior-induced choice shift | CC-BY (verify) | H1 perceptual-value |
| Winning Arguments / CMV (Tan et al., 2016) | argument text | opinion change (Δ) | see ConvoKit/DATA.md | H1/H2/H4 text |
| Persuasion-for-Good (Wang et al., 2019) | persuasion dialogue | donation amount ($) | see ConvoKit/DATA.md | H1/H4 behavioral $ |

The **primary confirmatory test** is H1/H2 on **NARPS `ds001734`** (CC0; its gain/loss framing →
accept/reject is a clean, license-clear choice-distortion target). **choices13k is deferred to a
separate paper** pending written permission from its authors (no license file at source). All
other datasets are pre-specified replication/extension targets, analyzed with the same pipeline.

## 5. Data hygiene, splits, and leakage controls

- **Provenance & integrity:** every raw file is recorded in `data/manifest.csv` with source
  URL, version/commit, license, access date, and **SHA-256**. Analysis refuses to run if a
  checksum mismatches.
- **Instrument pinned:** CSI is computed by `neurosignal` at a fixed commit recorded in
  `csi_provenance.json` (currently `e05bdb2`). The reference encoder is deterministic code (the
  commit pins it). If the learned encoder is used, its weight artifact is SHA-256-hashed and the
  pipeline refuses to run on mismatch — the same discipline as the data manifest.
- **Splits fixed in advance:** train/validation/test partitions are created and hashed
  *before* any outcome is examined; the test set is touched once. Implemented as **group-keyed**
  splits with a frozen split hash (`spikeprint/splits.py`); a leakage test asserts no group id
  (e.g., a choices13k problem id) spans partitions. For sequential/temporal data, splits are
  contiguous (no interleaving) to prevent autocorrelation leakage.
- **No feature/outcome leakage:** CSI and all baselines are computed from the stimulus only,
  with no access to the outcome or to test-set statistics. Text preprocessing parameters are
  fit on train only.
- **Determinism:** all randomness seeded; seeds recorded. Environment pinned (§8).
- **No data redistribution:** raw third-party data are **git-ignored**; only loaders,
  manifests, and derived non-identifying statistics are committed. Each dataset is used under
  its own license (`docs/DATA.md`). Enforced in CI by `scripts/check_no_raw_data.py`, which
  fails the build if anything other than the manifest/.gitkeep is tracked under `data/`.

## 6. Methods

**CSI computation.** CSI is computed by `neurosignal` from stimulus content (reference encoder
by default; learned TRIBE encoder as a pre-registered sensitivity analysis). The exact formula
and weights are version-pinned and reported; no post-hoc reweighting.

**Baselines (H2).** sentiment (lexicon), token length, readability, TF-IDF + L2 logistic
regression, and an LLM-judge baseline. Baselines and CSI are evaluated under identical splits.

**Spiking value model (mechanistic arm).** A spiking drift-diffusion / actor-critic model
(snnTorch; Eshraghian et al., 2023) maps value inputs to a choice + reaction-time
distribution; evaluated on choices13k/IBL with the same held-out protocol. The diffusion
account of choice/RT follows Ratcliff & McKoon (2008). Energy (spikes/decision) is reported
alongside accuracy; a rate-RNN and a small Transformer are controls. The spiking arm is a
*mechanistic complement*; H1–H4 do not depend on it.

**Covariate adjustment (potential outcomes; Rubin, 1974).** Effects are estimated with covariate
adjustment (Rubin, 1974) so that inference does not rely on an *assumed* CSI↔construct
correlation. The conceptual motivation — preferring a direct test over an indirect one that
presupposes such a correlation — follows the indirect-vs-direct test distinction of Kleiner
(2026); the adjustment *method* itself is Rubin's, not Kleiner's. For synthetic respondents,
**substrate (human vs. LLM) is an explicit covariate**, and we report whether the effect
survives adjustment.

**Synthetic respondents (admission gate).** LLM agents are used as additional respondents only
if they reproduce the *human* choice-distortion direction and rough magnitude on overlapping
choices13k problems (pre-set tolerance in `docs/SYNTHETIC.md`). If they fail calibration, they
are excluded from confirmatory analysis and reported as a negative methods result. They are
analyzed as a **separate, clearly-labeled arm and are never pooled into the confirmatory human
estimate.**

**Statistics.** Mixed-effects models with cluster-robust SEs; 95% bootstrap CIs (≥10k resamples;
percentile method, with BCa as a sensitivity check). The confirmatory pass/fail rule is
**Benjamini–Hochberg FDR across the H1–H4 family** (Benjamini & Hochberg, 1995), implemented as
`spikeprint.validate.decide_family` so the code enforces exactly the registered correction. The
power/MDES targets are fixed in §3 before any data are touched.

## 7. Analysis plan, exclusions, stopping rules

- Exclusions (e.g., malformed records, non-finite values) are defined in `docs/EXCLUSIONS.md`
  before analysis and applied identically to CSI and baselines.
- No optional stopping: the test set is evaluated exactly once for the confirmatory estimate.
- All exploratory analyses are clearly labeled and reported separately.

## 8. Reproducibility

- Python ≥3.10; dependencies pinned in `pyproject.toml` + `uv.lock`; seeds fixed.
- Helper scripts: `scripts/power.py` (MDES), `scripts/check_no_raw_data.py` (redistribution
  guard, runs in CI), `scripts/verify_refs.py` (CrossRef citation check). `make repro` (planned)
  will regenerate every number from raw data + manifest.
- Code released under MIT (`LICENSE`); data under their respective licenses.
- Continuous integration runs ruff + tests + the redistribution guard on every commit
  (`.github/workflows/ci.yml`).

## 9. Limitations (stated up front)

- CSI's neural inputs are a **proxy** unless the learned fMRI encoder is used; every output
  states which encoder produced it and flags `cortical_proxy` where applicable.
- Naturalistic text corpora (CMV, Persuasion-for-Good) carry selection effects; associations
  are correlational at the stimulus level, not causal claims about individuals.
- Synthetic (LLM) respondents are **not** humans; they enter only via the §6 calibration gate,
  are always reported separately with the substrate covariate, and are **never pooled into the
  confirmatory human estimate.**
- We deliberately avoid affect/consciousness decoding from neural signal; such decoding is
  known to be fragile and confounded, and is outside the claim space of this study.

## 10. Ethics & scope

Secondary analysis of public, de-identified data; no new data collection. We will record an
IRB/exemption determination in `docs/ETHICS.md`. Intended use is auditing and measurement
(decision-support/research), **not** diagnosis, and not deployment against individuals.

## References

See `REFERENCES.md` for full APA entries and per-citation verification status.
