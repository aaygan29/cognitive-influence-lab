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
confound into a measured axis. The estimator, honesty layer, and pooling (C1, C2, C4) are then validated on real human
decisions from the NARPS mixed-gambles fMRI dataset (108 subjects, 27,454 choices), and the phenotype
is evaluated through three lenses on that data (C7): it has low-dimensional structure, predicts
held-out choice at AUC 0.96, and individuates (cross-run fingerprinting at 14x chance, p = 5e-4). The
real neural-grounding arm is directionally consistent but underpowered, and the framework abstains on
it accordingly.

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
- **Real-data paradigm:** the mixed-gambles reward task (Tom, Fox, Trepel & Poldrack 2007) and its
  large public replication dataset NARPS (Botvinik-Nezer et al. 2019 Sci Data; 2020 Nature) supply real
  human loss-aversion choices and reward-anticipation fMRI.
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
Hierarchical (empirical-Bayes) partial pooling shrinks sparse individuals toward the population mean
so underpowering shows up as estimate variance, not overconfidence. From choice data the identifiable
coordinates are the scale-free affective ratios (loss aversion, threat, risk, discount relative to
gain sensitivity); the decision temperature tau requires reaction times (the full DDM), which is a
deployment-step addition.

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

Experiments E1-E4, E6, E7 are run on controlled data with known ground truth (done). E5 and the
real-data instances of E3/E6 are the deployment step (Section 7, pending data).

- **E1 (C1) Parameter recovery.** Simulate agents at known theta; fit; correlate recovered vs true
  affective *ratios* across agents; 20 seeds; report the recovery curve over decisions-per-agent with
  CIs. Pass: affective ratios reach r >= 0.6 within an achievable per-agent decision count.
- **E2 (C2) Coverage, calibration, abstention.** Empirical split-conformal coverage vs nominal on a
  heteroscedastic problem; expected calibration error; the MDES gate on under- vs adequately-powered
  effects. Pass: coverage within 0.03 of nominal, low ECE, gate abstains when underpowered.
- **E3 (C3) Affective-vs-behavioral transfer, controls.** Within the estimator, forecast a per-stimulus
  market outcome from an aggregate affective (brain) arm vs an aggregate behavioral arm vs a
  content-only baseline; stimulus-grouped CV; out-of-sample R2 + CRPS + permutation null. Positive
  control (market tracks affect) and negative control (market is noise). Pass: brain > behavior on the
  positive control, null on the negative. Real NARPS (fMRI) + DEAP (EEG), plus the content-blind
  2-factor specificity ablation, are the deployment.
- **E4 (C4) Pooling buys power.** Unpooled vs empirical-Bayes pooled recovery r at low
  decisions-per-agent. Pass: the bootstrap CI on the improvement excludes zero.
- **E5 (C5) Calibrated abstention on a real decision corpus (deployment, pending).** Public-record
  decision corpus (two blind coders + Cohen's kappa; preregistered temporal split); conformal
  predictions where identifiable; abstention where confounds break identifiability; calibration on the
  non-abstained subset vs base rate.
- **E6 (C5, C6) Inverse inference + attribution-scale ladder.** Recover the latent affective posture
  across attribution scopes S1..S6; measure identifiability (recovery r) and abstention rate per scope.
  Pass: the predicted dose-response (identifiability falls, abstention rises, S1->S6). The latent is an
  affective posture with a neural interpretation, never measured firing.
- **E7 (extension) States as recoverable manifold displacements.** An entity decides in a baseline and
  a threat-elevated state (threat coordinate shifted by a known amount); recover the per-condition
  posture and test whether the recovered shift tracks the true shift, with a zero-shift negative
  control. Pass: shift detected on the positive control, null on the negative.

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

**E7 (extension) States are recoverable displacements.** A threat-elevated state is recovered as a
detected, direction-correct shift in the threat coordinate (mean recovered shift +0.73, 95% CI
excluding zero, tracking the true shift at r = 0.35 across 60 entities, 10 seeds), while a zero-shift
negative control recovers no shift (mean +0.002, CI includes zero). The same estimator that fits a
stable phenotype also detects its transient displacement under a state.

### 5.1 Real data: the phenotype on NARPS ds001734 (Fig 8)
We ran the estimator on real human choices from the NARPS mixed-gambles task (Botvinik-Nezer et al.
2019; the Tom, Fox, Trepel & Poldrack 2007 paradigm): 108 subjects, 27,454 accept/reject decisions
over 50/50 gain-vs-loss gambles, with reaction times (OpenNeuro ds001734, CC0, events only).

- **RN1 (C1 on real data).** Per-subject loss aversion lambda = |b_loss|/|b_gain| recovered from real
  choices, split by the study's range manipulation: the equalIndifference group (asymmetric gain/loss
  range, as in Tom 2007) has median lambda = 1.45, while the equalRange group (symmetric range) has
  median lambda = 1.00, the known range-adaptation of loss aversion. Within-subject held-out choice
  prediction: mean out-of-sample AUC = 0.958 (95% CI [0.951, 0.964]), Brier 0.061 vs a base-rate Brier
  of 0.233.
- **RN2 (C2 on real data).** Split-conformal coverage on real held-out choices is 1.00 (>= the 0.90
  nominal, conservative because the label set is small and predictions are confident), expected
  calibration error 0.034; the identifiability gate abstains on 1% of subjects.
- **RN3 (C4 on real data).** Empirical-Bayes pooling reduces the loss-aversion SD only slightly
  (0.80 -> 0.78), because every NARPS subject has ~256 trials, the high-n regime where the synthetic
  E4 predicted pooling helps least. The controlled and real results agree on when pooling matters.
- **RN4 (DDM signature on real data).** Reaction time correlates negatively with net decision-value
  magnitude (median r = -0.30 across 107 subjects): larger value magnitude, faster choice, the
  evidence-accumulation signature the DDM predicts, motivating the reaction-time DDM for tau.

So C1, C2, and C4 hold on real human decisions, not only in simulation.

### 5.2 Real fMRI: affective grounding, with the framework gating its own neural arm (Fig 9)
We fit per-subject first-level GLMs with gain and loss parametric regressors on the fmriprep-
preprocessed NARPS BOLD (MNI152, one run, 40 subjects) and extracted NAcc and anterior-insula betas
(6mm spheres). The loss/threat side of the affective channel is now grounded on real data: **NAcc
decreases to loss (t = -2.12, p = 0.040) and anterior insula increases to loss (t = +2.32,
p = 0.025)**, both AIM-consistent and significant across subjects. NAcc gain-tracking is in the
predicted positive direction but not yet significant (t = +1.47, p = 0.15), and insula does not track
gain (p = 0.63, as expected). All three AIM direction predictions hold in sign.

The cross-subject correlation between neural and behavioral loss aversion strengthened sharply from
n = 12 to n = 40 (r = -0.01 -> -0.40), consistent with a real effect emerging with power, but at
n = 40 the minimum detectable correlation is 0.44, so |r| = 0.40 sits just under the identifiability
bar and the honesty gate abstains. This is the gate behaving exactly as designed at the boundary: an
r of 0.40 with MDES 0.44 is suggestive, not established, and is reported as withheld rather than
claimed. (The neural loss-aversion ratio is also sign-unstable at small per-subject betas, a further
reason to abstain.) The loss-channel grounding is a real, significant neural result; the individual
neural-to-behavioral link needs a larger sample or multi-run per-subject estimates to cross the bar.

### 5.3 The phenotype through three lenses: structure, function, individuation (Fig 10)
A decision phenotype earns the name only if it has geometric structure, predicts behavior (function),
and identifies the individual. We evaluate all three on real NARPS data and fold them into the claims.

- **Structure (the manifold has real geometry).** Across 108 subjects the affective coordinates are
  strongly coupled: gain sensitivity and loss sensitivity correlate at r = -0.92, and the six-feature
  phenotype has an effective dimensionality of 4.4 (participation ratio), below its feature count.
  The population phenotype occupies a low-dimensional structure, as the manifold model (C1) assumes,
  rather than filling the feature space.
- **Function (it predicts choice, and grounds in reward circuitry).** The phenotype forecasts real
  held-out choices at AUC 0.96 (Section 5.1), and its affective coordinates align in direction with
  NAcc gain-tracking and insula loss-tracking (Section 5.2, underpowered). This is the affective->choice
  function the estimator is built on (C1, C3).
- **Individuation (it is an individual trait).** Cross-run fingerprinting identifies a subject in
  held-out runs from their run-01 phenotype at 13% accuracy versus 0.9% chance (14x, permutation
  p = 0.0005, differential identifiability I_diff = 1.10). The phenotype is stable and person-specific,
  which is exactly the "integrative component individuates" half of the AIM split the framework rests
  on, and the property the C5/C6 individual-scope analysis depends on.

Read together, the three lenses say the phenotype is a real, low-dimensional, behavior-predictive,
individuating object on real human data, not a curve-fit. Each lens strengthens a specific claim:
structure and function under-write C1/C3, individuation under-writes the AIM split and the C5/C6
individual-scope reasoning.

### 5.4 Cross-dataset generalization: the phenotype transfers across labs (Fig 11)
We pooled a second, independent mixed-gambles dataset, Tom, Fox, Trepel & Poldrack (2007) (ds000005,
16 subjects, the original loss-aversion study), with NARPS. Two results. First, replication: the
Tom-2007 loss aversion median is lambda = 1.94, reproducing the classic value, while NARPS is 1.12
(diluted by its symmetric-range equalRange group); the distributions legitimately differ (KS
p = 0.004) because of the range manipulation, not a failure of the estimator. Second, and more
important, cross-dataset transfer: a valuation model fit on NARPS predicts Tom-2007 choices
out-of-dataset at AUC 0.86, and a model fit on Tom-2007 predicts NARPS choices at AUC 0.89. The
affective valuation the estimator recovers is not a per-dataset artifact; trained in one lab it
forecasts choices collected in another, years apart. This is the strongest real form of the pooling
and external-validity claims (C4, Gate 7): the phenotype generalizes across datasets.

### 5.5 External confirmation of the gain channel in independent reward maps (Fig 12)
Our n = 40 NARPS GLM left the NAcc gain channel non-significant (p = 0.15), the one AIM direction we
could not establish internally from single-run data. We sampled NAcc and anterior insula in
independent group-level reward maps on NeuroVault. In an expected-value monetary-incentive-delay map
(104 subjects) NAcc shows positive expected-value activation (sphere mean +1.0), in a gain > no-gain
map (46 subjects) NAcc is +1.6, and in a social-reward-anticipation map NAcc is +0.7. The gain
channel that our own sample was underpowered to confirm is positive across three independent,
larger reward-anticipation samples. This is group-level external confirmation of the affective
grounding, not per-subject power; it strengthens the channel claim while the individual
neural-to-behavioral correlation still awaits a dataset with per-subject fMRI and choices together.

### 5.6 Grounding by triangulation, and its statistical validation (Fig 13)
No single public dataset carries per-subject neural signal and per-subject behavior in a form that
resolves the individual neural-to-behavioral link. Rather than force one dataset to carry a direct
causal claim, we ground the affective channel by triangulation: each link in the chain neural ->
affective construct -> behavior is established in whichever dataset or modality has that pair, and the
honesty layer marks each link established / abstained / pending (Fig 13). Neural -> construct is
established (NAcc gain-anticipation positive across three independent reward maps; NARPS loss channel
significant), and construct -> behavior is established (held-out and cross-dataset choice prediction),
so the construct is grounded even though the direct within-subject link is not.

We put this on a formal footing three ways. (i) The correlation-transitivity bound: a 3x3 correlation
matrix is positive semidefinite, so given r(neural,construct) and r(construct,behavior) the direct
r(neural,behavior) is confined to r_ab r_bc +/- sqrt((1-r_ab^2)(1-r_bc^2)). The established links
imply a feasible interval of [-0.51, +0.29] for the direct correlation, and the observed value
(-0.40) falls inside it; the interval is wide because the within-sample neural-to-construct
correlation is itself weak. (ii) A Sobel mediation test of the indirect path (delta-method SE) is not
significant (z = -0.73), consistent with that weak within-sample link. (iii) Fisher combination of the
established links' p-values gives a combined p = 2.4x10^-7: the convergent evidence for the affective
grounding is strong even though the single direct link is abstained. Within the one joint sample
(NARPS, 39 subjects with both modalities), the stable neural loss measures correlate near zero with
behavioral loss aversion (all gated to abstention), and the only nominally significant correlation
(the neural-lambda ratio, r = -0.40, p = 0.011) has the wrong sign and rests on an unstable
small-denominator ratio, so the gate abstains. The honest conclusion: the affective construct is
triangulated by convergent, statistically-combined evidence, while the direct individual neural-to-
behavior mapping remains unestablished and is reported as such. This is convergent, model-based
evidence, not a proof of a direct neural-to-behavior cause.

## 6. Discussion

**Operating characteristics, in one place.** The experiments jointly map where the instrument works.
Identifiability of the affective phenotype needs on the order of 1000 to 2000 clean decisions per
agent to reach r = 0.6 from choice alone (E1); hierarchical pooling buys back a substantial part of
that at low n (E4); the aggregate affective signal forecasts a market outcome beyond behavior when
the generative structure links them, and not otherwise (E3); and inverse inference is licensed at
aggregate attribution scopes but must abstain at the individual scope, where 86 percent of entities
fall below identifiability (E6). Read together, these say something specific: a single individual with
a sparse, confounded decision record is usually in the abstention regime, and the honest output there
is a withheld estimate, not a confident profile. That is not a limitation the method hides; it is the
result the method is built to report.

**States and roles as manifold structure.** A transient emotional state is a within-entity
displacement along the manifold: E7 recovers a threat-elevated state as a detected, direction-correct
shift in the threat coordinate, with a zero-shift control recovering nothing. Occupational roles
(trader, clinician, officer) are the between-entity analog: different priors on the same coordinates
plus different context weighting. The same estimator handles both, which is what it means for the
phenotype to be a point (or a short trajectory) on one shared manifold rather than a bespoke model per
case.

**Why the honesty layer is the contribution, not an add-on.** Every result above is reported as a
proper score with a coverage guarantee or an explicit abstention. The abstention behavior is what
turns the hardest case (an unscannable, institutionally-confounded individual) from an overclaim into
a calibrated non-answer. A behavioral-prediction method without this layer would report the E6
individual-scope numbers as if they were findings; ours reports that it cannot, and is right to.

## 6b. Conclusion
We presented a neurally-grounded decision-phenotype estimator that is identifiable, honest by
construction, and explicit about the boundary of its own licence. On controlled data it recovers the
phenotype, delivers distribution-free coverage and calibrated abstention, reproduces the affective
over behavioral forecasting dissociation when it is present and stays null when it is not, and maps
the attribution scope at which individual-level inference stops being warranted. The remaining work is
deployment on real neural datasets and a coded decision corpus (Section 7), not new machinery.

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
(subject-/stimulus-grouped CV) inherited from the `behavioral_decoding` harness. Every result and
figure regenerates with one command (`python reproduce.py`), which runs all experiments, rebuilds the
vector figures and LaTeX tables from the fresh `results/*.json`, runs the test suite, and prints a
per-claim gate summary.

## 9b. Extension: a temporal-difference front-end for the affective channel
Our affective channel rests on *anticipatory* affect: NAcc gain-anticipation is a signal about a
future reward, not the reward itself. Recent work shows how such reward-predictive representations
form: over learning, hippocampal reward representations shift backward in time from the reward to the
cues that predict it, a dynamic recapitulated by a temporal-difference (TD) model (Brandon, Williams &
Pehlevan et al., 2026, predictive coding of reward in the hippocampus). This suggests a principled
extension: treat the affective value that feeds the drift-diffusion valuation as a *learned* TD
predictive value rather than a fixed quantity, with our temporal-discount coordinate delta_time
identified with the TD discount gamma. Individual differences in TD learning (rate, discount) would
then ground both individual differences in the phenotype and its formation over experience, and would
give the individuation result (C7) a temporal axis: does a subject's phenotype stabilize as their
reward representation becomes predictive? This is a modeling extension grounded in a mechanistic
result, not a claim we test here.

## 10. Future work (deployment)
1. **Real neural data.** Wire the estimator to the NARPS (fMRI) and DEAP (EEG) loaders in the
   `behavioral_decoding` harness and run E3 on real recordings, with the content-blind two-factor
   specificity ablation and the brain-beyond-behavior and brain-beyond-content baselines. This is the
   empirical neuroforecasting test that the synthetic controls here only validate the instrument for.
2. **Reaction-time DDM.** Add the first-passage-time likelihood so the decision temperature tau and
   response style are identified, beyond the choice-only affective ratios recovered now.
3. **Real decision corpus.** Curate the US-president decision corpus with two blind coders and a
   reported Cohen's kappa, preregister the temporal split and kill criteria, and run E5/E6 on it; the
   expectation, from E1 and E6, is that most individual-scope cases land in the abstention regime.
4. **Per-person calibration.** Connect to enrollable personal-brain and idiographic-affect pipelines
   for few-shot per-person priors, and run a preregistered replication on a held-out figure and a
   held-out market.
5. **Power the gain-channel grounding.** Acquire a large Monetary Incentive Delay reward-anticipation
   fMRI dataset with preprocessed derivatives (many public instances) to bring the NAcc gain-tracking
   effect (currently p = 0.15) across significance and to push the neural-to-behavioral loss-aversion
   correlation past its identifiability bound.

## Appendix A — Preregistration
Temporal split, arms, kill criteria for C3 and C5, committed to `checkpoints/` before fitting.

## Figures and tables
See `figures/FIGURES.md`. All figures are vector PDF; Tables 1-2 are `booktabs` LaTeX in
`paper/tables/`. Fig 1 architecture (`fig1_architecture`); Fig 2 attribution-scale ladder +
predicted dose-response, schematic hypothesis (`fig2_scale_ladder`); Fig 3 parameter-recovery curve
(`fig3_recovery`); Fig 5 transfer arms, positive vs negative control (`fig5_transfer`); Fig 6
unpooled vs pooled recovery (`fig6_pooling`); Fig 7 attribution-scale ladder result, identifiability
and abstention vs scope (`fig7_scale_ladder_result`); Fig 8 real NARPS loss-aversion by group +
held-out prediction and honesty (`fig8_real_narps`). Table 1 central claims (`claims.tex`); Table 2
experiments and pass criteria (`experiments.tex`). A calibration figure for E2 renders from the
honesty self-check when its coverage sweep is exported.
