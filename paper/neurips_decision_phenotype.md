# Neural-Grounded Computational Decision Phenotypes: Honest-by-Construction Prediction of Choice, with Provable Abstention

*NeurIPS-form working draft. Venue chosen after results land. Numbers marked `[TBD-Ex]` are
produced by the experiment named, never written before the run. Figures are vector PDF and tables
are `booktabs` LaTeX (see `figures/`, `paper/tables/`), so this converts to a NeurIPS LaTeX
template directly (e.g. via pandoc for the prose).*

---

## Abstract

Predicting how an agent will decide is usually posed as a black-box classification problem,
which yields uncalibrated point predictions and no principled way to say "I do not know."
We take a different route grounded in the neuroscience of choice. Following the
Affect-Integration-Motivation (AIM) account, we model a decision-maker as a low-dimensional
point on a decision manifold whose coordinates are neuroeconomic parameters, and we split the
choice process into an **affective** component that generalizes across individuals and an
**integrative** component that individuates. We estimate this manifold with a drift-diffusion
likelihood, calibrate the affective component on populations with real neural recordings
(fMRI and EEG), and wrap every prediction in distribution-free conformal coverage, a
strictly-proper score, and a minimum-detectable-effect abstention gate. We show (C1) the
estimator is identifiable on synthetic ground truth, (C2) the honesty layer delivers nominal
coverage and calibrated forecasts with provable withholding, (C3) the affective component
forecasts held-out and less-representative choice above and beyond behavior inside a single
estimator, (C4) hierarchical pooling across datasets lowers the minimum detectable effect,
and (C5) on genuinely confounded real-world targets the framework predicts where identifiable
and **provably abstains** where the information and attribution confounds break identifiability,
turning a failure mode into a tested safety property. As an extension (C6) we run the model in
reverse, inferring the latent affective posture most consistent with a documented decision, and we
show that this inverse inference is scale-dependent: identifiable at aggregate attribution scopes
and degrading to abstention at the individual scope, a dose-response that turns the attribution
confound into a measured axis. `[TBD-Ex1..6]`

## 1. Introduction

- Behavioral prediction is high-stakes and usually dishonest about its own uncertainty.
- The neuroforecasting literature (Genevsky & Knutson 2015; Genevsky, Yoon & Knutson 2017;
  Tong et al. 2020; Stallen et al. 2021; Genevsky, Tong & Knutson 2025) established that
  brain-derived *affective* signal can forecast aggregate, out-of-sample choice better than
  behavior, and generalizes even when the sample is unrepresentative. But those results are
  population->population, at small n, and have never been unified into a single identifiable
  estimator or equipped with distribution-free guarantees.
- Our contribution is a method that is (i) neurally grounded in AIM, (ii) honest by construction
  (conformal + proper scoring + abstention), and (iii) explicit about the one thing prior work
  leaves implicit: *when prediction is not licensed at all*.
- **We do not claim to measure any individual's brain.** The neural grounding is a population
  property; individual claims are about revealed decision policy, bounded by named confounds.

## 2. Related work

- **Neuroforecasting / AIM:** Knutson (2008) anticipatory affect; Samanez-Larkin & Knutson
  (2015) AIM; Knutson & Genevsky (2018) affective-vs-integrative transfer; Tong (2020),
  Stallen (2021), Smith/Montague (2014), Falk (2012/15). Fernandes (2022) gives AIM an EEG
  signature, which licenses our EEG arm.
- **Forecast evaluation:** Gneiting & Raftery (2007) strictly-proper scoring rules.
- **Distribution-free uncertainty:** Angelopoulos & Bates (2021) conformal prediction.
- **Choice process models:** drift-diffusion / sequential sampling; resource-rational analysis
  (Lieder & Griffiths 2020) as *interpretation* of the parameters, not a predictive claim.
- **Inverse inference:** Bayesian inverse planning and theory of mind as inverse RL (Baker, Saxe &
  Tenenbaum 2009; Jara-Ettinger); computational-psychiatry generative modeling and its identifiability
  discipline (Huys, Montague); neural latent-state inference (Schiereck et al. 2025; Blanco-Pozo et al.
  2024). Prospect theory in foreign-policy decision-making (McDermott; Levy), where leaders' latent
  risk postures are explicitly *inferred* from choices, motivates the attribution-scale ladder.
- **Gap we fill:** no prior work unifies AIM decomposition + a choice-process likelihood +
  distribution-free coverage + an identifiability-linked abstention gate in one estimator, nor
  evaluates abstention as a safety property on confounded real-world decision records.

## 3. Method

### 3.1 The decision manifold
theta = (rho_gain, lambda_loss, kappa_risk, delta_time, omega_threat, tau_consistency). An event
is a context vector c (stakes, gain/loss frame, ambiguity, threat, time pressure, reference point).

### 3.2 AIM-structured valuation and DDM likelihood
V(a | c, theta) = rho_gain E[gain] - lambda_loss E[loss] - kappa_risk sigma - omega_threat ambiguity
+ delta_time delay. The value difference drives a drift-diffusion process; boundary set by caution.
Affective channel (rho, lambda, omega) shares a hierarchical population prior; integrative channel
(kappa weighting, tau/threshold) is per-agent. This is the operationalization of Knutson & Genevsky's
affective-vs-integrative split.

### 3.3 Estimation
Hierarchical Bayes; partial pooling shrinks sparse individuals toward the population posterior so
underpowering shows up as posterior width, not overconfidence.

### 3.4 Honesty layer (C2)
Strictly-proper scoring (log/Brier/CRPS); split-conformal prediction sets at nominal coverage; an
MDES gate that abstains when the identifiable-event count puts the minimum detectable effect above
the claimed effect. Abstention is a first-class output.

### 3.5 Neural grounding
Affective-channel priors and the valuation form are calibrated on populations with real recordings:
NAcc/insula betas (fMRI, NARPS) and AIM-EEG correlates (Cue-P3, CNV; DEAP). No individual target is
scanned.

### 3.6 Inverse inference: from a decision to the latent state that best explains it
The forward model predicts a decision from theta and c. The **inverse** problem runs it backward:
given an observed decision D and its context c, infer the posterior p(z | D, c) over the latent
decision-state z, where z is the affective posture (the gain-approach vs threat-avoidance balance
and the risk posture) that best explains D under the forward AIM-DDM model. This is Bayesian inverse
planning (Baker, Saxe & Tenenbaum 2009; theory of mind as inverse reinforcement learning) applied to
a neurally-structured generative model, and it is a computational-phenotyping inference of the kind
that computational psychiatry performs with explicit attention to parameter identifiability (Huys,
Montague and others). Because the forward map is calibrated on scanned populations, z carries a
**neural interpretation** (a gain-approach posture is the computational analog of elevated NAcc-type
gain-anticipation relative to insula-type threat-anticipation). We report that interpretation as a
hypothesis attached to a latent, with its posterior width, and we never state it as measured neural
firing. The honest object is "decision D is consistent with an affective posture tilted toward
gain-approach, posterior width W", not "region R fired in person P".

### 3.7 The attribution-scale ladder: where does inverse inference survive?
Neuroforecasting is validated population-to-population; the aggregate affective signal is exactly
the component that generalizes. The attribution confound (was a decision the individual's or the
institution's) is therefore not only a nuisance, it is an axis. We evaluate inverse identifiability
at nested attribution scopes, from most aggregate to most individual:

- S1 executive branch including all agencies; S2 White House and the Pentagon; S3 the White House;
  S4 President and cabinet; S5 President and key advisors; S6 the President alone.

The prediction from AIM and the neuroforecasting results is a **dose-response**: inverse
identifiability (posterior sharpness, and out-of-sample skill of the recovered latent) is highest at
aggregate scopes, where the population-generalizable affective component dominates, and degrades as
scope narrows toward the individual, where the idiosyncratic integrative component and the attribution
confound take over. The abstention gate (Section 3.4) should engage increasingly as scope narrows.
The empirical question is the title of the ladder: how far toward S6 can inference survive before the
gate must abstain. A sharp individual-scope posterior would be a red flag (analyst degrees of freedom),
not a triumph. **Subject and decision selection:** US presidents with dense public policy records, and
within each, decisions with short response latency (rapid crisis reactions), because the anticipatory-
affect channel that neuroforecasting relies on operates on that fast timescale, so fast reactive
decisions are where the affective (generalizable) component is most applicable and slow deliberative
ones are dominated by the integrative (individuating) component.

## 4. Experiments

- **E1 (C1) Parameter recovery.** Simulate agents at known theta; fit; correlate recovered vs true;
  20 seeds; report r + CI per parameter. Pass: r >= 0.6 for affective parameters. `[TBD-Ex1]`
- **E2 (C2) Coverage + calibration.** Empirical conformal coverage vs nominal; reliability diagram;
  abstention behavior below the MDES floor. Pass: |coverage - nominal| <= 3 pts at n>=200. `[TBD-Ex2]`
- **E3 (C3) Affective-vs-behavioral transfer.** Within the unified estimator, forecast held-out /
  less-representative choice on NARPS (fMRI) and DEAP (EEG); stimulus-grouped CV; permutation null;
  ablation vs content-blind 2-factor PCA. Report affective ΔCRPS + bootstrap CI. `[TBD-Ex3]`
- **E4 (C4) Pooling buys power.** MDES and posterior width, joint vs single-dataset fits. `[TBD-Ex4]`
- **E5 (C5) Calibrated abstention on confounded targets.** Public-figure decision corpora
  (preregistered temporal split); conformal predictions where identifiable; abstention rate where
  confounds break identifiability; calibration on the non-abstained subset vs base rate. `[TBD-Ex5]`
- **E6 (C6, extension) Inverse inference + attribution-scale ladder.** Recover the latent posterior
  p(z | D, c) for documented decisions across attribution scopes S1..S6; measure posterior sharpness,
  out-of-sample skill of the recovered latent, and abstention rate as a function of scope. Test the
  predicted dose-response (identifiability falls, abstention rises, S1->S6). Report the scope at which
  the gate must abstain. Preregistered; language is latent affective posture with a neural
  interpretation, never measured firing. `[TBD-Ex6]`

## 5. Results

The results below are validations of the estimator and the honesty layer on controlled data:
parameter recovery against known ground truth (E1), the coverage/calibration/abstention guarantees
of the honesty layer (E2), pooling's effect on low-data recovery (E4), and positive/negative
controls on the transfer pipeline's operating characteristics (E3). They establish that the
instrument measures what it claims and does not manufacture effects; the empirical neuroforecasting
claim itself is evaluated on real fMRI/EEG (NARPS, DEAP), the deployment step (Section 7).

**E1 (C1) Identifiability, and its data requirement (Fig 3, Table 2).** Recovery of the affective
phenotype ratios rises with decisions per agent (20 seeds, 60 agents). Raw logistic coefficients
recover at r = 0.53 to 0.88 across 200 to 2000 decisions/agent, confirming the model is well
specified. The scale-free affective ratios reach the r >= 0.6 criterion by roughly 1000 to 2000
decisions/agent (threat r = 0.60 at 1000 and 0.73 at 2000; loss aversion r = 0.69 at 2000) and are
underdetermined at 200. This data requirement is the quantitative motivation for pooling (C4) and for
abstention on sparse records (C5). The decision temperature tau is not identifiable from choice alone
and is recovered from reaction times with the full DDM (future work, Section 7).

**E2 (C2) Honest-by-construction (positive control, `results/e2_honesty_selfcheck.json`).**
Split-conformal intervals achieve nominal coverage (empirical 0.90 at nominal 0.90 on a
heteroscedastic problem, within 0.03); a well-specified probability model has expected calibration
error 0.018; the MDES gate abstains on an underpowered effect (r = 0.15 at n = 50) and reports a
powered one (r = 0.30 at n = 500). The layer delivers coverage, calibration, and abstention as
designed.

**E4 (C4) Pooling buys power (Fig 6).** Empirical-Bayes partial pooling lifts low-data recovery: at
150 decisions/agent, recovery r improves for every ratio with a bootstrap CI on the improvement
excluding zero (loss aversion 0.10 to 0.30, threat 0.15 to 0.27, risk 0.09 to 0.44, discount 0.15 to
0.44; 20 seeds). Borrowing strength across the population reduces per-agent estimate variance, the
mechanism behind a lower effective MDES.

**E3 (C3) Transfer pipeline operating characteristics (Fig 5).** On the positive control, where the
aggregate market outcome tracks the affective latent by construction, the aggregate affective (brain)
arm forecasts held-out stimuli better than the aggregate behavioral arm (out-of-sample R2 0.845 vs
0.731; brain-minus-behavior Delta R2 = +0.114, 95% CI [+0.092, +0.137]; stimulus-grouped permutation
p = 0.003). On the negative control, where the market is independent noise, no arm exceeds chance and
the permutation null is not rejected (brain R2 = -0.05, permutation p = 0.90, brain-minus-behavior CI
includes 0). The pipeline detects the dissociation only when it exists. This is a synthetic
positive/negative control on operating characteristics, not the empirical claim.

**E6 (C5, C6) Inverse inference is scale-dependent (Fig 7).** Running the estimator in reverse to
recover the latent affective posture, evaluated across nested attribution scopes S1 (executive plus
agencies, 150 contributors) to S6 (the individual, one contributor), yields the predicted
dose-response (50 entities, 8 seeds). Identifiability of the posture falls monotonically from r = 0.35
at the aggregate scope to r = 0.11 at the individual scope, while the abstention rate rises from 0.16
to 0.86. On the subset the gate does not abstain, recovery stays higher than on the full set at every
scope (for example 0.57 vs 0.35 at S1), so abstention keeps what is reported more reliable. The
individual scope is where inference is mostly not licensed: the framework abstains on 86 percent of
entities there, exactly the behavior the attribution confound predicts and the safety property C5
asserts. The latent is a computational affective posture with a neural interpretation, never a claim
of measured firing; this simulation establishes the mechanism, and the real US-president decision
corpus (two blind coders and Cohen's kappa) is the deployment.

## 6. Discussion (Step 5)
Operating characteristics: where the method works, at what n, with what coverage. Extension to state
(sad/angry as transient theta shift) and role (trader/cop/doctor as prior + weighting) phenotypes.

## 7. Limitations
- Population->individual transfer is the central open risk (Gate 7); we test it, we do not assume it.
- Neural grounding is population-level; no individual is scanned.
- Public-figure targets carry information and attribution confounds; C5 is designed around, not
  through, them.
- Resource-rational interpretation is not a load-bearing predictive claim.

## 8. Broader impacts
Dual-use profiling risk. We restrict to public-record, office-level revealed policy; we forbid
psychodiagnostic language and any claim of measured neural firing in a named individual; the inverse
inference reports a latent computational posture with a neural *interpretation*, never a measurement.
Abstention-on-confounded-targets is a core deliverable rather than something suppressed, and the
attribution-scale ladder is designed to expose, not hide, the scope at which individual-level
inference stops being licensed.

## 9. Reproducibility
Seeds, configs, split hashes, code commit, dataset versions released; leakage guards
(subject-/stimulus-grouped CV) inherited from the `behavioral_decoding` harness.

## Appendix A — Preregistration
Temporal split, arms, kill criteria for C3 and C5, committed to `checkpoints/` before fitting.

## Figures and tables
See `figures/FIGURES.md`. All figures are vector PDF; Tables 1-2 are `booktabs` LaTeX in
`paper/tables/`. Fig 1 architecture; Fig 2 attribution-scale ladder + predicted dose-response;
Fig 3 recovery; Fig 4 coverage/calibration; Fig 5 transfer arms; Fig 6 pooling/power;
Fig 7 abstention operating curve. Table 1 central claims; Table 2 experiments and pass criteria.
