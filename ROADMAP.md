# Roadmap — six steps forward

Each step names its **goal**, **method**, **data**, **deliverable**, the **gate(s)** it
must pass (see `GUIDELINES.md`), a **kill criterion** committed *before* the run, and a
**checkpoint**. Steps are gated: you do not advance to step N+1 until step N's checkpoint
is green or its kill criterion has fired and been recorded.

The arc is deliberately more than one increment. Step 1 grounds in what is known; step 2
finds the advanced application; step 3 replicates and advances in new data and formats;
step 4 evaluates a genuinely new method; step 5 concludes and extends; step 6 opens future
directions.

---

## Step 1 — Ground in what is known  *(status: ~80% done)*

- **Goal:** a defensible foundation — the AIM math, the neuroforecasting evidence and its
  real weaknesses, the harness, and a synthetic positive control that has the effect built in.
- **Method:** literature sweep (done: Knutson 2008; Carter 2009; Samanez-Larkin & Knutson
  2015 AIM; Genevsky & Knutson 2015; Genevsky/Yoon/Knutson 2017; Knutson & Genevsky 2018;
  Tong 2020; Stallen 2021; Genevsky/Tong/Knutson 2025; Fernandes 2022 AIM-EEG; Falk 2012/15;
  Smith/Montague 2014) + math sweep (Gneiting-Raftery proper scoring; Angelopoulos-Bates
  conformal; DDM; resource-rational) + code audit of `behavioral_decoding`.
- **Data:** none real yet; `synthetic.py` positive control (affective/integrative two-channel).
- **Deliverable:** this repo's `README`, `ROADMAP`, `GUIDELINES`, the design rationale + risk
  register, the literature/math synthesis (in `checkpoints/`).
- **Gate(s):** Gate 0 (is the harness real + leakage-guarded — PASS), Gate T (math sound — PASS).
- **Kill criterion:** if the affective/integrative split has no operational definition
  separable from a generic 2-factor model, the neuro framing is decoration — stop and rescope.
- **Checkpoint C1-DONE:** design rationale + risk register committed; synthetic control runs green.

## Step 2 — Advanced application: the unified AIM-DDM estimator + honesty layer

- **Goal:** prove **C1** (identifiability) and **C2** (honest-by-construction) on synthetic
  ground truth, before any real data can flatter us.
- **Method:**
  - Build `src/phenotype.py`: hierarchical Bayesian estimator of theta =
    (rho_gain, lambda_loss, kappa_risk, delta_time, omega_threat) with a **drift-diffusion**
    choice likelihood; affective channel (rho, lambda, omega) shares a population prior,
    integrative channel (kappa, tau/threshold weighting) is per-subject.
  - Build `src/honesty.py`: strictly-proper scoring (log-score, Brier, CRPS), split-conformal
    prediction sets, and an **MDES gate** that abstains when the identifiable-event count puts
    the minimum detectable effect above the claimed effect.
  - Parameter-recovery experiment: simulate agents at known theta, fit, correlate recovered
    vs true, **20 seeds**, report r and CI per parameter.
- **Data:** synthetic only. Real data is forbidden at this step by design.
- **Deliverable:** `results/e1_recovery.json`, `results/e2_coverage.json`; Figures 2, 3.
- **Gate(s):** Gate 1 (multi-seed), Gate 6 (calibration), Gate T (coverage guarantee holds).
- **Kill criterion (C1):** if recovery r < 0.6 for any affective parameter across 20 seeds,
  the model is not identifiable — fix the likelihood or reduce the parameter set, do not proceed.
- **Kill criterion (C2):** if empirical conformal coverage deviates from nominal by > 3 points
  at n>=200, the calibration is broken — fix before real data.
- **Checkpoint C2:** recovery + coverage green, committed, STATUS updated.

## Step 3 — Replicate and advance in new datasets and new formats

- **Goal:** move off synthetic. Replicate the neuroforecasting transfer direction **inside our
  unified estimator** on real neural data, in two modalities (fMRI *and* EEG), and prove pooling
  buys power. This is **C3** and **C4**.
- **Method:**
  - fMRI: NARPS ds001734 loader already exists in the harness; fit the affective channel from
    NAcc/insula betas, the behavioral channel from choices; forecast held-out choice; compare
    arms with stimulus-grouped CV + permutation null (all present in `evaluation/`).
  - EEG (new format): DEAP via the harness loader; use the AIM-EEG correlates from Fernandes
    2022 (Cue-P3 anticipation, CNV integration) as the affective features. This is the advance:
    the transfer is tested on *cheap, scalable EEG*, not only fMRI.
  - Pooling: hierarchical joint fit across NARPS + DEAP; measure MDES and posterior width vs
    single-dataset fits.
  - **Required ablation (Gate 3):** the affective/integrative split must beat a content-blind
    2-factor PCA of equal capacity, or the neuro label is unearned.
- **Data:** NARPS ds001734 (OpenNeuro, public); DEAP (licensed download). Provenance stamped.
- **Deliverable:** `results/e3_transfer_{narps,deap}.json`, `results/e4_pooling.json`; Figures 4, 5.
- **Gate(s):** Gate 0 (real data, provenance), Gate 3 (specificity ablation), Gate 4 (economic
  baseline partialed out), Gate 7 (holds across two datasets).
- **Kill criterion (C3):** if affective ΔCRPS over behavior has a bootstrap CI including 0 under
  stimulus-grouped CV in *both* datasets, C3 dies; narrow the paper to C1/C2/C4 (methods+honesty).
- **Kill criterion (C4):** if pooled posterior width is not below single-dataset width, drop the
  pooling claim rather than spin it.
- **Checkpoint C3:** at least one real-data transfer result with CI, committed.

## Step 4 — A genuinely new method: calibrated abstention on confounded targets

- **Goal:** the novel contribution — **C5**. Take the framework to targets where ground truth
  is genuinely broken (public figures: no scan, information + attribution confounds) and show
  the system predicts where identifiable and **provably abstains where not**. Evaluate the
  abstention gate as a *safety property*, which no neuroforecasting work has done.
- **Method:**
  - Build the decision corpus: code consequential decisions of Trump term-1 (2016–2020), Xi,
    and a control figure into event vectors c (stakes, gain/loss frame, ambiguity, threat, time
    pressure, reference point) and action classes (escalate/de-escalate, risk-on/off, etc.).
  - **Two blind coders + Cohen's kappa** before any modeling (Gate 8). Public record only.
  - **Preregister** the temporal split (fit 2017–2018, predict 2019–2020) and the kill criteria
    in `checkpoints/` *before* fitting (Gate 11).
  - Fit the integrative channel on the early window with the affective channel's population prior
    from Step 3; forecast held-out late window; report conformal sets + proper score vs base rate.
  - The evaluation of interest: does the MDES/conformal gate **fire** where the confounds make
    theta unidentifiable? Report abstention rate, and the calibration of predictions on the
    non-abstained subset.
- **Data:** public-record decision corpora (curated here); no neural data for the individuals.
- **Deliverable:** `results/e5_abstention.json`; Figure 6; the preregistration doc.
- **Gate(s):** Gate 8 (inter-rater kappa), Gate 10 (ethics framing), Gate 11 (prereg), Gate 6
  (language = "revealed policy", never "neurology").
- **Kill criterion (C5):** if the gate does *not* abstain on the confounded subset (i.e. it
  emits confident predictions where identifiability is broken), the safety property is false —
  report that honestly as a negative result; do not tune the gate post hoc to look good.
- **Checkpoint C5:** abstention behavior characterized, prereg frozen, committed.

## Step 5 — Conclusions and extension

- **Goal:** state what actually transfers and what does not; extend the manifold to states and roles.
- **Method:** synthesize operating characteristics across C1–C5 (where does the method work, at
  what n, with what coverage). Extend: model an emotional **state** (sad/angry) as a transient
  theta shift with directions the affect literature specifies, testable on DEAP-style induced
  affect; model **roles** (trader/cop/doctor) as different priors + context weighting.
- **Deliverable:** the paper's Discussion; an operating-characteristic table; Figure 7.
- **Gate(s):** Gate 6 (every conclusion sized to its tier), Gate 5 (necessity via channel lesion).
- **Kill criterion:** any conclusion that cannot be sized to a tier gets cut, not softened.
- **Checkpoint C5-CONCLUDE.**

## Step 6 — Future directions

- Real **all-modality + market** dataset (the field's missing substrate: EEG + face + fMRI +
  a real market outcome on the same stimuli); active few-shot enrollment; connection to
  `cortex-of-anyone` (personal digital brains) and `affectprint` (idiographic affect readout).
- Preregistered replication on a held-out figure and a held-out market.
- **Deliverable:** the paper's Future Work; a concrete data-acquisition or collaboration plan.

---

## Dependency graph

```
Step1 (ground) --> Step2 (C1,C2 synthetic) --> Step3 (C3,C4 real neural) --> Step4 (C5 confounded) --> Step5 (conclude/extend) --> Step6 (future)
                         |                            |
                         +--- honesty layer feeds ----+--- gates every downstream number
```

If Step 3 kills C3, the paper is still whole as a methods + honesty-machinery contribution
(C1, C2, C4, C5). That is the fallback, pre-committed here so it is not a post-hoc rescue.
