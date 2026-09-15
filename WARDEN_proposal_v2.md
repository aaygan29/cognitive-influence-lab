# WARDEN v2: a calibrated cognitive-security reasoning instrument for neural + behavioral influence analysis

**Revised design proposal, hardened against a six-persona red team and grounded in current literature. No em dashes.**

*Weighted Assessment of Reweighting, Detection, and Entrainment of Neural-behavioral signals.*

> **What changed from v1.** This revision implements every disposition in triage_warden.md. The three
> load-bearing corrections: (1) WARDEN is now explicitly a **reasoning instrument over stated assumptions**,
> not a field measurement device; H1's index is assumption-relative by construction and says so. (2) The
> defensive floor is now an **adaptive minimax saddle**, not an inversion against a fixed attacker, and is
> coupled to detector fragility. (3) The ethical scope is **enforced at the API boundary** (no per-individual
> output in asymmetric-power settings, EPPA screen), not merely stated. Every "bound" and "provable" claim
> over the human setting has been struck or confined to the in-silico connectome regime.

---

## 0. One paragraph

WARDEN is a defensive cognitive-security reasoning instrument. Given a stated set of assumptions about a
person, an influence channel, and an adversary, it computes three calibrated, abstaining, assumption-relative
outputs: (H1) a **manipulability sensitivity estimate** for how far a modeled influence input shifts a
modeled behavioral output, (H2) a **live-engagement detector** that reports whether the observable
neural-behavioral signal is consistent with an active influence attempt, with mandatory abstention and only
a simulator-referenced confidence, and (H3) a **simulated countermeasure ordering** that names which channel
must be hardened for a defense to hold against an adaptive adversary. It is the empirical, operational
companion to Bagley's mathematical framework for the security of cognition (arXiv:2403.07945): assumptions
in, a defensible countermeasure ordering out. It does not measure a person's manipulability as a fact about
them; it computes what a stated threat model implies, and its value is in making that implication explicit,
falsifiable, and honestly bounded. It inherits the deception project's discipline verbatim (calibration,
abstention, base-rate realism, explicit threat model, enforced scope) and is built by assembling instruments
this portfolio already owns.

---

## 1. The gap, and what WARDEN is and is not

Bagley (arXiv:2403.07945) gave cognitive security its equations: adversaries who violate the privacy and
autonomy of cognition, defenders who obstruct them, formal access channels and thresholds. Denning, Matsuoka
and Kohno (doi:10.3171/2009.4.FOCUS0985) named neurosecurity around neural-device privacy. Both stop at the
framework; the neurorights literature (doi:10.1007/s12152-021-09468-6) is normative. What is missing is an
instrument that takes a stated amount of access and a stated adversary and returns a calibrated, honest,
falsifiable defensive recommendation.

**WARDEN is that instrument, with one hard boundary the red team forced into the definition:** it is a
*reasoning tool over assumptions*, not a measurement of a mind. This is not modesty; it is a consequence of
the fact (critique C1) that the quantity H1 estimates has no field referent. There is no human study that
measures a manipulability gain operator, so any number WARDEN emits is a statement about its assumption set
Theta, not about the person. WARDEN's job is to make Theta explicit, propagate it honestly to a defensive
recommendation, and show how the recommendation moves as Theta moves. That is a legitimate and useful thing
for a cognitive-security analyst to have. It is not a scanner that reads how manipulable someone is.

**What WARDEN is not:** a lie detector, a manipulation scanner, a screening tool, an admissible instrument,
an employment or clearance or custody or immigration instrument, or an offensive targeting product. Sections
6 and 9 make each of these an enforced non-goal, not a disclaimed one.

---

## 2. Three heads, and why the split survives the red team

Cerberus guards a gate with three heads; WARDEN guards cognitive autonomy with three, because the three
questions have different failure modes and must not be collapsed (the PRISM naive-average error).

| Head | Question (v2, corrected) | Inherits | Isolated failure mode | Key red-team fix |
|---|---|---|---|---|
| **H1 Assay** | Under assumption set Theta, how sensitive is the modeled output to the modeled influence input? | cultist, neurobridge, wiring-not-weights | Reifying an assumption-relative estimate as a trait measurement | Relabeled sensitivity-not-bound (C2); assumption-relative (C1); defensive verdict not exploitability score (C13); worst-case-state and session-scoped (C11) |
| **H2 Detector** | Is the observable signal consistent with an active influence attempt right now? | deceptkit SPRT + abstention | False accusation at low base rate; oracle leakage | Ceiling recalibrated off CIT (C3); simulator-PPV only (C4); entrainment demoted to attention covariate (C10); gate hardened vs oracle titration (C6) |
| **H3 Floor** | What is the minimal countermeasure ordering that holds against an adaptive adversary? | CHORUS attack-surface math, neuroprivacy floor | Prescribing a defense that a reallocating adversary walks around | Minimax saddle not fixed-attacker inversion (C5); coupled to H2 fragility (C14); "provable" confined to in-silico (C7) |

Collapsing them remains the cardinal error: a modeled sensitivity (H1), an engagement event (H2), and an
adaptive defensive recommendation (H3) have different referents and different base rates, and the defense
(H3) is a conjunction over open channels, not a sum.

---

## 3. Inputs

Modality-agnostic on the read side, ranked by effect size as deceptkit ranks CIT channels. Per exposure
epoch, whatever subset is available:

**Neural (any subset):**
- ERP belief-processing markers: N400 and late-positivity to belief-incongruent claims, and prediction-error
  signals. These index *processing of incongruence*, explicitly not *yielding* (C10).
- Frontal-midline theta (cognitive-control load).
- fNIRS/fMRI, research mode only: prefrontal/vmPFC/dACC engagement during belief updating
  (doi:10.1523/JNEUROSCI.0063-10.2010, doi:10.1038/s42003-026-09794-6).
- Direct goal-layer read only in an instrumented BCI regime (the CHORUS setting).
- **Removed in v2:** inter-trial phase coherence / entrainment "to the influence carrier" as an evidence
  channel. A persuasion attempt has no periodic carrier; entrainment indexes attention, not influence
  (C10). Where retained, it is a labeled attentional-engagement covariate only.

**Behavioral (always required, the honest floor):**
- Choice shifts, response-time and micro-hesitation dynamics, stance/frame-uptake in language.
- Autonomic bridge channels: EDA, pupil, HR, respiration (the deceptkit stack, grounded in real ERP CORE
  statistics).

**Context (required for honesty):**
- The influence channel content and timing (the stimulus S), so H2 is conditioned, not free-floating.
- The **adversary model** (new in v2): what channels the operator controls, whether the operator can observe
  WARDEN's output, and the operator-class assumption for H1 (C6, C12).
- The prevalence prior for H2.

The commitment carried from deceptkit: **behavior-only is the baseline every neural claim must beat**, and
the brain-beats-behavior delta (target modest, ~20 to 25% of variance per Falk 2010, with a CI that may
include zero) is reported for every output. Human conscious discrimination is poor (only 22% beat chance at
rumor discrimination, doi:10.1038/s41598-024-82696-x), so subject self-report is never used as a training
label (C-crosscut X.1).

---

## 4. Methods, head by head (v2)

### H1. Manipulability sensitivity assay (assumption-relative)

Define a modeled behavioral readout g(W_i, S) and a modeled influence input u drawn from an **explicitly
stated influence distribution** P(u | operator-class). The v1 definition as a supremum operator norm is
withdrawn (C2): a human yields one non-repeatable trajectory through a non-stationary g, so a worst-case
sup over an unsampleable admissible set is not estimable. WARDEN instead estimates the **expected realized
sensitivity**

```
S_i(Theta) = E_{u ~ P(u|op-class)} [ || g(W_i, S+u) - g(W_i, S) || ]   (a conditional expectation, not a bound)
```

and reports it with the admissible-u set named. The estimate uses the cultist/neurobridge primitive: a
calibrated model predicting realized behavioral shift from neural belief-processing + autonomic + baseline
features, calibrated by isotonic regression on held-out folds grouped by subject (StratifiedGroupKFold),
exactly as deceptkit calibrates. Three honesty mechanisms, all new in v2:

1. **Assumption-relative labeling (C1).** The output is named "simulator sensitivity under Theta." A
   parameter-to-output sensitivity analysis is published; if S tracks the injected Falk increment
   one-to-one, that circularity is reported as the finding.
2. **Norm-choice sensitivity (C2).** S is recomputed under a range of reasonable norm/aggregation choices;
   if it swings more than threefold, the instability is a reported result, not a hidden degree of freedom.
3. **Operator-class scoping (C12).** S is labeled "sensitivity to the simulated benign-persuasion operator
   class." It is never presented as a general bound; a hostile-tradecraft operator is out-of-distribution
   and a low S is explicitly not clearance.

**Output of H1 is a defensive-sufficiency verdict, not an exploitability score (C13):** "under Theta, the
current defense holds / does not hold; channel X must be hardened for it to hold." No rank-ordered
per-channel gain vector, no cross-session storage, no ranking of people (C11). H1 is computed against the
**worst-case state**, not the assessed state, so a session-time estimate does not under-protect other states.

### H2. Live-engagement detector (simulator-referenced, abstaining, oracle-hardened)

Per exposure epoch, accumulate evidence that the observable signal is consistent with an active influence
attempt, using deceptkit's SPRT machinery, with per-channel log-likelihood ratios fused by PRISM's learned,
leakage-audited weights (never summed). Four corrections from v1:

1. **Ceiling recalibrated off the CIT (C3).** H2's achievable power is anchored on the naturalistic
   continuous-EEG decoding literature, not the averaged P300 CIT d*=1.59, because influence-landing in an
   aperiodic self-paced stream has no probe onset and no clean baseline. Two dilutions
   (averaged-to-single-trial AND oddball-to-naturalistic) are both booked. The pre-registered prediction is
   that **coverage** (fraction of non-abstain epochs) is low, single digits to low tens of percent, and
   coverage is a headline number. If coverage is near zero, H2 is reported as a null instrument.
2. **No field number (C4).** H2 reports only a simulator-referenced PPV under the simulator's
   label-generating process. Field sensitivity/specificity are declared **unmeasurable** (attempt-present
   AND effective are jointly unobservable and subjects cannot self-label), not merely unmeasured.
3. **Entrainment removed (C10).** Evidence channels are belief-incongruence ERP and prediction-error
   signals plus behavioral/autonomic drift, not carrier entrainment.
4. **Oracle hardening (C6).** An explicit adversary-observes-output model is added. If the operator can see
   WARDEN's output they can titrate steering to sit inside the ABSTAIN band; v2 adds a persistent
   boundary-hugging meta-detector, hidden gate state for the channel-controlling party, and randomized gate
   thresholds. Residual leakage when the attacker observes the output is irreducible and is stated.

The base-rate disqualifier is front and center (deceptkit: ~15 false accusations per true detection at 1%
prevalence): H2 reports the posterior and simulator-PPV given the caller's prevalence prior, and ABSTAIN
never clears a target and never confirms an operation.

### H3. Simulated countermeasure ordering (adaptive, coupled, in-silico)

H3 is the CHORUS attack-surface curve run in reverse, corrected for the adaptive-adversary error that killed
a decade of adversarial-ML defenses (C5). The v1 inversion against a fixed attacker is withdrawn. The floor
is the **minimax saddle**

```
d* = argmin_d  max_{u} S(d, u),   with u free to reallocate across all open channels after observing d
```

reported only as that saddle value. Three corrections:

1. **Adaptive by construction (C5).** The adaptive-reallocation attack is a pre-registered H3 falsification
   test; until d* survives a reallocating u it is labeled a "non-adaptive floor."
2. **Coupled to detector fragility (C14).** M(d) is estimated with the same read that H2 admits collapses
   under countermeasures. v2 propagates that collapse into d* as widened uncertainty; when the input signal
   is actively suppressed, d* is forced upward (conservative) or H3 abstains from prescribing a floor rather
   than solving on suppressed values.
3. **"Provable" confined to in-silico (C7).** Model-extraction threshold theorems assume query determinism,
   a fixed target, i.i.d. access, and stationary outputs; a brain violates all four. v2 strikes "provable"
   from the human setting, enumerates the four violated preconditions, and confines threshold language to
   the in-silico connectome regime (flyvis/RNN) where the preconditions hold. The human-facing output is a
   "simulated countermeasure ordering," not a floor guarantee.

The floor remains a **conjunction** across channels (PRISM discipline): a defense must hold across all open
channels because one unblocked high-gain channel defeats the average.

---

## 5. Outputs

A single per-assessment report (self-directed or IRB-supervised research only, see section 6):

1. **H1 defensive-sufficiency verdict** under stated Theta, with the norm-choice and parameter sensitivity
   attached. No exploitability ranking.
2. **H2 live-engagement posterior** for the epoch, with prevalence prior, simulator-PPV, coverage, and an
   INFORMED / NAIVE / ABSTAIN decision. No field PPV.
3. **H3 simulated countermeasure ordering**: which channel to harden, as an adaptive minimax
   recommendation, with the residual risk after it (some channels irreducible), and an abstention when the
   signal is suppressed.
4. **Brain-beats-behavior delta**: what neural data bought over behavior alone, with a CI that may include
   zero. Reported regardless.
5. **Assumption ledger**: the full Theta (operator class, admissible-u set, adversary observation model,
   prevalence prior) that produced the outputs, so a reader can see what the numbers are relative to.
6. **Abstention/quality log**: every gate that fired.

---

## 6. Enforced scope, threat model, and hard limits

**Defensive posture, enforced.** WARDEN protects a mind; it does not actuate. The write side stays in-silico
on a fly connectome (CHORUS). H1/H3 emit a defensive-sufficiency verdict, never an exploitability score
(C13).

**Enforced consent boundary (C8, the hardest limit).** H1 and H2 are most valuable on subjects not told
what is measured, and refusal in an asymmetric-power setting is itself scored as damping. Because the
physics defeats consent in adverse contexts, v2 **enforces at the API boundary** that WARDEN emits no
per-individual H1 or H2 output in any asymmetric-power relationship: employment, pre-employment, custody,
immigration, security clearance, adversarial negotiation, or interrogation. Per-individual output is
restricted to self-directed use or IRB-supervised research with an audited, contractually enforceable
no-adverse-consequence guarantee. "Abstain is not exoneration" is implemented as a **refusal to run** in
adverse contexts, not a printed caveat.

**Enforced EPPA and admissibility screen (C9).** WARDEN has no established field error rate and therefore no
valid field PPV; any PPV it prints is labeled a simulator PPV. An EPPA-and-equivalent screen refuses to run
in any employment context at the API boundary. Admissibility is an owned non-goal: WARDEN is
Daubert-inadmissible by construction and is never offered as evidence of a mental state.

**No physiological ground truth (C1, C4, irreducible).** No dataset pairs a real influence operation with
synchronized neural+behavioral capture and a labeled outcome, and the target label (attempt-present AND
effective) is jointly unobservable. Therefore real data grounds the simulator's parameters, not a deployed
classifier, and field validity is declared **unmeasurable**, not merely unestablished. WARDEN ships as a
calibrated in-silico reasoning instrument plus a thin adapter for the narrow real datasets that exist
(belief-updating EEG, persuasion neuroimaging), with that limit stamped on every output.

**Irreducible limits, owned and stated:**
- The construct is assumption-relative: H1 reports what Theta implies, and if the delta echoes the injected
  effect size, that circularity is the finding (C1).
- Field validity is unmeasurable, not unmeasured (C4).
- Consent is defeated by the instrument's physics in adverse contexts, hence the enforced refusal (C8).
- A skilled defender's own countermeasures degrade H2 toward chance (the deceptkit arms-race result,
  doi:10.1002/hbm.25814, doi:10.1016/j.ijpsycho.2022.01.009).
- An unseen channel and an unseen (hostile-tradecraft) operator class are unbounded risks; a low H1 is not
  clearance (C12).
- If the adversary can observe WARDEN's output, H2 leaks through the gate below even the damping-only result
  (C6 residual).

---

## 7. Falsifiable evaluation plan (v2, literature-grounded targets)

Every prediction is pre-registered, and each is designed so the honest literature effect size is the target,
not an inflated one. A miss is reported as a miss (the deceptkit AUC 0.635 precedent: the collapse was
reported, not hidden).

**H1 predictions.**
- P1.1 (brain-beats-behavior). Neural belief-processing features add a **modest** increment over
  behavior-only prediction of realized shift, target ~20 to 25% of variance (Falk 2010 r=0.49,
  doi:10.1523/JNEUROSCI.0063-10.2010), with a CI that may include zero for some channels. Prediction: the
  increment is modest and channel-dependent, not transformative.
- P1.2 (assumption sensitivity, C1/C2). S(Theta) tracks the injected Falk increment; the parameter-to-output
  sensitivity is near one. Reported as the circularity finding.
- P1.3 (operator-class fragility, C12). S estimated under the benign operator class fails to predict shift
  under a held-out adversarial operator class. Prediction: out-of-distribution operators break H1.

**H2 predictions.**
- P2.1 (low coverage headline, C3). On naturalistic continuous streams, non-abstain coverage is low (single
  digits to low tens of percent). Coverage is the headline; low coverage is a success of honesty, not a bug.
- P2.2 (base-rate disqualifier). At 1% prevalence the simulator-PPV is low and false-positive-dominated
  (deceptkit ~15:1); WARDEN abstains rather than accuses.
- P2.3 (countermeasure collapse). Under a modeled covert countermeasure, H2 detection collapses toward
  chance (deceptkit 96% miss; doi:10.1002/hbm.25814). Prediction: H2 is defeatable and says so.
- P2.4 (oracle leakage, C6). When the adversary observes WARDEN's output and titrates into the ABSTAIN band,
  the boundary-hugging meta-detector recovers only part of the lost signal; residual leakage is quantified
  and reported.

**H3 predictions.**
- P3.1 (adaptive gap, C5). A non-adaptive floor d* is defeated by a reallocating adversary: max_u S(d*, u)
  under reallocation exceeds the non-adaptive estimate. The minimax saddle closes the gap only partially;
  the residual is reported.
- P3.2 (conjunction beats single-channel). A single-channel defense is defeated by reallocation to an open
  channel while the minimax conjunction holds. Prediction: single-channel hardening fails, conjunction is
  necessary.
- P3.3 (fragility coupling, C14). When H2's input is suppressed, d* uncertainty widens and H3 abstains
  rather than prescribing a floor on suppressed values. Prediction: H3 refuses to over-promise under
  suppression.
- P3.4 (in-silico only, C7). The extraction-threshold theorem holds on the connectome regime and fails to
  transfer to the human read; the four violated preconditions are demonstrated. Prediction: no human floor
  guarantee.

**Global honesty check.** Calibration (isotonic, held-out) is reported for H1 and H2, so a stated p=0.8
means ~80% of such cases. If any head cannot be calibrated, it ships as ABSTAIN-only.

---

## 8. Build plan

Nothing is built from scratch; WARDEN assembles instruments this portfolio owns.

- **H2** reuses deceptkit's fusion.py (SPRT policy, isotonic calibration, StratifiedGroupKFold, abstention
  gates, base-rate posterior) almost verbatim, with the ceiling recalibrated to naturalistic decoding, ITPC
  removed, and the oracle-hardening meta-detector added.
- **H1** reuses the cultist/neurobridge calibrated forecaster and deceptkit's calibration stack, wrapped in
  the assumption-ledger and norm-sensitivity harness.
- **H3** reuses CHORUS's attack-surface curve A(f) and neuroprivacy floor, re-expressed as a minimax saddle
  with fragility coupling, run on the in-silico connectome.
- **Fusion** across channels uses PRISM's learned, leakage-audited weights and the conjunction law.

A v0.1 skeleton is one deceptkit-sized effort: fork fusion.py for H2, wire the cultist forecaster into an H1
sensitivity harness, and drive H3 from the CHORUS curve, on simulated neural+behavioral epochs with the
parameters fixed to the real effect sizes already cataloged (P300 averaged d~1.0 to 1.6; Falk r=0.49; Meijer
2014 per-channel d's).

---

## 9. Why this is worth building (v2)

The honest literature says machine influence is now real and measurable at the behavioral level: LLM
dialogues durably reduce conspiracy belief (Costello, Pennycook, Rand 2024, doi:10.1126/science.adq1814),
GPT-4 out-persuades humans especially when personalized (Salvi 2025, doi:10.1038/s41562-025-02194-6),
and effects extend to policy attitudes (doi:10.1038/s41467-025-61345-5) and anthropomorphic agents
(doi:10.1073/pnas.2415898122). The same literature says the neural side is **moderate and
countermeasure-fragile**, not a scanner: neural predicts behavior with a modest increment (Falk r=0.49),
concealed-knowledge detection is strong when averaged but collapses under covert countermeasures, and humans
themselves discriminate manipulation barely above chance.

An honest instrument for this moment is therefore not a manipulation detector that claims AUC 0.99. It is a
calibrated reasoning tool that takes a stated threat model and returns a defensible, abstaining, honestly
bounded countermeasure recommendation, and that refuses to run where it would be misused. That is exactly
the empirical companion Bagley's framework calls for, and it is what a cognitive-security analyst can
actually defend under cross-examination: not "this person was manipulated," but "under these stated
assumptions, this is the defense that holds, this is what it cannot cover, and here is where I abstain."

The red team did not break WARDEN. It forced WARDEN to say out loud what it is: an assumption-relative,
field-unmeasurable, consent-defeating-in-adverse-contexts, adversary-adaptive reasoning instrument. Saying
those four things out loud, and enforcing them in the architecture, is the contribution.

---

*Companion documents: literature_review.md (18 DOI-verified sources), validity_map.md (20 graded claims),
critiques_warden.md (14 red-team critiques), triage_warden.md (dispositions), revision_log_warden.md
(v1 to v2 changelog). Original design: WARDEN_proposal.md.*
