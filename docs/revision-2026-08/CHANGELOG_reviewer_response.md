# Response to Reviewers, NeuroXR 2026

**Paper: WARDEN, a calibrated cognitive-security reasoning instrument.**
**Original decisions: R1 Weak Accept, R2 Weak Accept, R3 Weak Reject.**

We thank all three reviewers. The reviews converge on two real problems and we have fixed both rather
than argued around them. First, the paper described an instrument without giving readers the formal
object, the algorithm, or a worked trace, so a reader could not tell what was implemented versus
proposed (R1, R2). Second, and more serious, the calibration case study rested on a null result at N=8
that we had not power-checked, and presented as evidence of safety a finding that a proper power
analysis shows is a Type II error waiting to happen (R3, central objection). We ran that power analysis.
R3 was right. We are not walking the finding back quietly; we are foregrounding the correction, because
it is the more interesting and more defensible result: an instrument that certifies "safe" from an
underpowered null is itself a cognitive-security failure mode, and WARDEN is redesigned so it cannot do
that.

Below, point by point. Section numbers refer to `REVISED_MANUSCRIPT_SECTIONS.md`, submitted alongside
this letter. `IMPLEMENTATION_STATUS.md` gives the full, audited implemented-vs-proposed table referenced
throughout.

---

## Reviewer #1 (Weak Accept, High expertise)

> "Work remains largely conceptual, lacks formalism/pseudocode/worked example; hard to tell implemented
> vs aspirational; calibration case study (motion re-ID vs neural individuation null) needs more
> methodological detail."

**On formalism and pseudocode.** We agree the original submission named "SPRT" and "minimax saddle"
without defining either. Section (a) of the revised sections gives both from first principles: the
classical Wald sequential probability ratio test (log-likelihood-ratio accumulation, two decision
boundaries, a third "continue/abstain" region) and how WARDEN's H2 instantiates it as a **single-epoch,
multi-channel** fusion of that same three-region rule, not a literal multi-epoch sequential test. That
distinction matters and the original paper blurred it; the revision states plainly that true epoch-to-
epoch sequential accumulation is a described capability, not yet implemented, and IMPLEMENTATION_STATUS.md
says so in the same row as the code that exists today. The minimax saddle is defined as
`d* = argmin_d max_u M(d,u)` over a **finite candidate set** (all single channels plus all channel pairs,
not the full power set of channels), which is what `warden/heads.py::h3_floor` actually enumerates. We
had claimed a more general search than the code performs; that gap is now stated, not hidden.

**On implemented vs. aspirational.** `IMPLEMENTATION_STATUS.md` is a new, dedicated document with one row
per component, each tagged CODE-VERIFIED (runs today, on simulated data, cite the exact file), PARTIALLY
IMPLEMENTED (a real function exists but omits a described safeguard, cite the gap), or DESIGN PROPOSAL
ONLY (named in the paper, no code). Three gaps we found while writing this, that the original paper did
not disclose: `h3_floor()` does not call `enforce_scope()`, so the C8 asymmetric-power refusal that H1 and
H2 enforce is **not yet enforced on H3's code path**; `h3_floor()` never emits ABSTAIN, so the proposal's
"H3 abstains rather than prescribes a floor on suppressed values" (C14) is a design commitment, not a
behavior the code has; and `h1_assay()` has no calibration-quality gate, so "any head that cannot be
calibrated ships ABSTAIN-only" is likewise not yet true of H1. We would rather a reviewer catch us saying
this than catch us not saying it.

**On the calibration case study.** See the response to R3 below, which subsumes this: R3 identified the
deeper problem (the null itself was underpowered), and fixing that also gives the methodological detail
R1 asked for, because a proper power analysis forces us to state sample size, effect size, chance level,
and correction method for every arm. Section (c) is the rewritten case study; Section (d) is the new
per-channel table.

---

## Reviewer #2 (Weak Accept, Medium expertise)

> "Same core limitation, described as an instrument but no formal spec/pseudocode/system diagram/worked
> example. Wants a compact worked XR scenario showing assumptions, each output, and abstention
> conditions."

Section (b) is new: a single, compact worked scenario (an XR social/wellness app with an adaptive
LLM-driven in-app companion), one stated `Theta` assumption ledger, and a full trace through H1, H2, and
H3, ending with the C8 refusal firing when the same query is re-run under an adverse-power context. Every
number in the trace is the literal output of `run_warden.py` against the committed `results/warden_results.json`,
not a hand-picked or hypothetical figure, and every number is labeled as simulator output pinned to
published effect sizes, not a claim about any real person. We list, head by head, the exact numeric or
categorical condition that produces ABSTAIN (H2's three gates: `quality < 0.5`, `max_foil_z >= 2.5` or
`foil_disp >= 1.6`, or `|evidence_bits| < 1.5`) versus the exact condition that produces the hard C8
refusal (`theta.context` in the adverse-context set), because these are different mechanisms and the
original paper's prose ran them together. A system diagram in the geometric sense (boxes and arrows) adds
little beyond the assumption-ledger-to-heads-to-outputs flow the pseudocode in Section (a) already makes
explicit; we judged the worked trace to be the more informative artifact for this venue and did not add a
separate figure for it, but can add one in camera-ready if the committee wants it.

---

## Reviewer #3 (Weak Reject, High expertise)

> "(1) calibration case study relies on withheld self-citation, unverifiable; (2) 'zero of 25 regions'
> null at N=8 is likely a Type II error / underpowered, not proof of safety, this is the central rigor
> objection; (3) the offense/defense asymmetry argument leans on LLM-persuasion literature poorly bridged
> to continuous multimodal XR physiological sensing; (4) 'sequential-testing machinery with mandatory
> abstention' and 'minimax saddle' used without formal definition. Wants: a per-channel risk comparison
> table (motion vs neural), a bar chart of the 25-ROI statistical findings, tighter LLM-to-XR bridging, and
> a concrete adversary-exploitation scenario."

### (2) The central objection: N=8, zero of 25 regions, is a Type II error, not proof of safety

**We concede this fully.** It is correct, and it was the single most important thing wrong with the
original submission. We ran the power analysis R3 is implicitly demanding and it confirms the objection
in both directions:

- The digital-brain fingerprinting result (N=4, chance = 25%) that the original case study cited as strong
  support does not survive correction: **0 of 25 ROIs survive Benjamini-Hochberg FDR at q=0.05, and 0 of
  25 survive Bonferroni**, against the committed per-ROI p-values (raw p in [0.026, 0.048], all just under
  the uncorrected 0.05 line, the textbook signature of small-N over-read). Source:
  `digitalbrain_geometry_results.json`, block `correction_of_committed_pvalues`. An independent
  re-derivation of the same ROIs from the raw fMRI betas, using a subject-respecting permutation null
  instead of the parametric one, gives 25 of 25 surviving BH-FDR, which sounds like the opposite result,
  but the same file documents why that is *also* not evidence of an effect: at N=4 the permutation null has
  only 4! = 24 achievable label arrangements, so its minimum reachable p-value floors near 1/24 = 0.042,
  and every ROI's p-value sits at that floor. BH-FDR does not protect against a null whose p-values are all
  piled at a hard floor; that re-derivation's own 0-of-25 Bonferroni result is the trustworthy number from
  that arm. Both analyses agree once corrected: **zero ROIs survive the correction that actually controls
  false discoveries at this N.**
- The companion functional-axis result (`wiring-not-weights/exp04`, N=8, chance = 12.5%, "0 of 25 ROIs
  significant") that the original paper read as "reconstruction is dear," meaning safely non-individuating,
  fares no better under a power check. The minimum detectable effect size (MDES) for 80% power at N=8,
  8-way identification, alpha=0.05, is an accuracy of **~0.59**; the observed accuracy was **0.175**. A
  study that can only detect true effects at or above 0.59 accuracy, run at an observed 0.175, has not
  ruled out a real effect anywhere in the plausible range between chance and 0.59. It has ruled out only
  the largest possible effects. (A separately computed MDES specifically on the digital-brain N=8 8-way
  design, in the same `digitalbrain_geometry_results.json` file, gives 0.503 under a related but distinct
  exact-binomial construction; we report both numbers rather than picking the more favorable one, and both
  say the same thing: this design cannot distinguish "no individuating signal" from "moderate individuating
  signal.")

**The reframe, which is the actual fix, not a hedge.** The original calibration case study said: motion
telemetry re-identifies strongly, neural individuation is null, therefore the neural channel is the safer
one and a defender should prioritize hardening the motion channel. That conclusion required the neural
null to mean "no signal." It does not; it means "underpowered." The corrected case study (Section (c),
below) instead says: motion/hand telemetry is a **confirmed, high-power, independently-replicated re-
identification risk** (two published studies, hundreds to tens of thousands of subjects, effect sizes near
ceiling); neural individuation at currently achievable XR-adjacent sample sizes is **statistically
undetermined**, not proven low-risk. An analyst using WARDEN who reads "channel risk: UNDETERMINED" is
told to keep the channel gated pending a properly powered study. An analyst who had read the original
paper's "0 of 25, reconstruction is dear" would have been told the opposite: that the channel is
comparatively safe to leave open. That is a materially worse operational recommendation, and it is the
exact failure mode a cognitive-security instrument exists to prevent: certifying a channel safe on
evidence that could not have detected the risk if it were there. WARDEN v2's H1/H3 outputs are a
defensive-sufficiency verdict, never a clearance, and Section 6 of the design proposal already enforced
this in principle (C1, C12); the rigor pass shows why that enforcement is load-bearing rather than
decorative. We now treat "the null is underpowered" as the headline finding of the calibration case study,
not a footnote to it, and we believe this **strengthens** rather than weakens the paper's cognitive-
security argument: the paper's contribution shifts from "here is a channel we measured as safe" (a claim
we cannot support) to "here is why measuring channel safety from an underpowered null is itself the
security-relevant lesson, and here is an instrument built to say UNDETERMINED instead of SAFE when that is
the honest answer" (a claim the corrected numbers directly support).

### (1) Withheld self-citation, unverifiable

The original calibration case study's supporting numbers came from an unpublished internal re-analysis
that the reviewer correctly could not check. That is fixed structurally, not just by adding a citation.
Every statistic in the revised Section (c) and the Section (d) table traces to one of two artifacts we are
including as supplementary material: `identification_rigor_results.json` and
`digitalbrain_geometry_results.json`, both regenerated fresh for this revision (`generated_at:
2026-08-16`), both containing the raw per-ROI and per-arm numbers behind every summary statistic quoted
in the paper, and both reproducible from the analysis scripts in the same directory
(`exp_digitalbrain_geometry.py`, `exp_identification_rigor.py`). The two supporting figures, "Fig. 1" and
"Fig. 2" in the revised manuscript, are `figures/fig1_digitalbrain_fdr.png` (the per-ROI p-value/FDR
picture) and `figures/identification_rigor.png` (the identification-equivalence and RMT-null picture); we
also generated `figures/fig2_identification_equivalence.png` and `figures/digitalbrain_geometry.png` as
supplementary detail. Nothing in the calibration case study now depends on an unshared source.

### (3) LLM-persuasion literature poorly bridged to continuous multimodal XR physiological sensing

Fair, and fixed. Section (e) is a rewritten bridging paragraph that stops at the level of "LLMs can
persuade" and instead names three specific, published XR sensing channels and states concretely how an
adaptive LLM dialogue policy could use each one as a persuasion-optimization signal: head/hand motion
telemetry (closing the loop on hesitation and approach/avoidance before the user speaks), eye-gaze and
fixation dynamics (closing the loop on attentional capture and claim-by-claim uptake), and headset-
integrated autonomic sensing, EDA and PPG-derived heart rate (closing the loop on arousal, the same signal
class deceptkit already uses as an evidence channel). The persuasion-efficacy literature already in the
paper (Costello, Pennycook and Rand 2024; Salvi et al. 2025) establishes that personalized, adaptive LLM
dialogue outperforms static messaging; the bridging paragraph's job, which it did not previously do, is to
say what "personalized" and "adaptive" concretely consume in an XR context where those three channels are
already instrumented and streaming.

### (4) "Sequential-testing machinery" and "minimax saddle" used without formal definition

Fixed in Section (a), which we would ask the reviewer to read against the actual code
(`deceptkit/fusion.py::BayesianFusion`/`MultivariateFusion`, `warden/heads.py::h3_floor`) rather than
against a diagram, because the formal definitions given are exactly what those functions compute, no more
and no less, including the two gaps (H3's missing scope-enforcement call, H3's missing ABSTAIN path)
disclosed above and in `IMPLEMENTATION_STATUS.md`.

### Requested artifacts

- **Per-channel risk comparison table (motion vs. neural):** Section (d).
- **Bar chart of the 25-ROI statistical findings:** we point to the already-generated
  `figures/fig1_digitalbrain_fdr.png`, which plots exactly this (per-ROI raw p-value against the BH-FDR and
  Bonferroni thresholds across all 25 ROIs), rather than regenerating a duplicate; Section (c) describes
  what it shows.
- **Tighter LLM-to-XR bridging:** Section (e).
- **Concrete adversary-exploitation scenario:** Section (b), the worked XR scenario, doubles as this; it
  is written from the adversary's (the platform operator's) incentive structure inward, then shows what
  WARDEN, run by the user or an auditor, would report and where it would refuse to report anything at all.

---

## Summary of changes

| Reviewer ask | Where addressed | What changed |
|---|---|---|
| Formal spec / pseudocode (R1, R2) | Section (a) | SPRT and minimax saddle defined from first principles, matched line-for-line to the actual code, gaps between design and code stated explicitly |
| Worked example / system trace (R1, R2) | Section (b) | Full Theta-to-outputs trace on one concrete XR scenario, real simulator numbers, explicit abstention triggers |
| Calibration case study detail (R1) | Section (c) | Rewritten with sourced, reproducible statistics and figure references |
| N=8 null is Type II, not safety (R3, central) | REBUTTAL above + Section (c) | Conceded directly; MDES computed and reported; case study reframed as UNDETERMINED not SAFE |
| Withheld self-citation (R3) | Section (c), (d) | Every number traced to a shared, reproducible artifact and script |
| LLM-to-XR bridging (R3) | Section (e) | Three named, cited XR sensor channels with concrete exploitation mechanics |
| Per-channel risk table (R3) | Section (d) | New table, motion/gaze telemetry vs. neural individuation, corrected statistics |
| 25-ROI bar chart (R3) | Section (c) | Points to existing `fig1_digitalbrain_fdr.png` |
| Undefined SPRT / minimax terms (R3) | Section (a) | Formal definitions, code-matched |

We believe this revision answers the formalism objection (R1, R2) completely and answers R3's central
rigor objection by conceding it and showing that the concession strengthens rather than undermines the
paper's thesis. We have not tried to rescue the original "reconstruction is dear" framing; we replaced it.
