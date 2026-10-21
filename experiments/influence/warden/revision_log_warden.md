# WARDEN revision log: v1 to v2

Record of how the design was tested and hardened. No em dashes.

## Process

1. **Literature review** (literature_review.md, sources.csv): 18 DOI-verified sources across cognitive
   security, neurosecurity, ERP/fMRI belief-processing, concealed-information detection, countermeasure
   fragility, and current LLM-persuasion RCTs.
2. **Validity map** (validity_map.md): 20 load-bearing claims extracted from v1 and graded STRONG / MODERATE
   / WEAK / ABSENT against the retrieved evidence. Four genuine weak points flagged.
3. **Agent-council red team** (critiques_warden.md): six adversarial personas (neuro-methodologist,
   ML-security, statistics-skeptic, neuroethics-lawyer, red-teamer, grant-reviewer) attacked the design.
   Consolidated to 14 critiques: 9 critical, 5 major.
4. **Triage** (triage_warden.md): each critique dispositioned FIX (6), REFRAME (5), or IRREDUCIBLE (3),
   echoing the deception project's own 12/4/2 discipline.
5. **Revision** (WARDEN_proposal_v2.md): every disposition implemented.

## What the literature changed

The single largest effect of grounding the design in current evidence was **downward recalibration of every
performance claim**. The honest literature says:

- Neural predicts behavior with a **moderate** increment, not a decisive one: Falk et al. 2010 MPFC r=0.49,
  neural adds ~23% variance beyond self-report (doi:10.1523/JNEUROSCI.0063-10.2010). This anchors H1's
  brain-beats-behavior delta at a modest target with a CI allowed to include zero, not a transformative gain.
- Concealed-information detection is strong when **averaged** (P300 CIT d*=1.59, k=54,
  doi:10.1016/j.ijpsycho.2025.113236) but collapses under covert countermeasures (doi:10.1002/hbm.25814,
  doi:10.1016/j.ijpsycho.2022.01.009). This forced H2's live-detection ceiling down and made countermeasure
  collapse a pre-registered prediction rather than a footnote.
- Humans discriminate manipulation barely above chance (22% beat chance at rumor discrimination,
  doi:10.1038/s41598-024-82696-x). This forbids using subject self-report as a training label.
- Machine influence is real, durable, and current, but bounded, not mind control (Costello 2024 Science
  doi:10.1126/science.adq1814; Salvi 2025 NHB doi:10.1038/s41562-025-02194-6; policy-persuasion 2025
  doi:10.1038/s41467-025-61345-5). This sized WARDEN to a resistible threat and rejected alarmism.

The through-line: any design claiming AUC near 0.99 is contradicted by the entire corpus. A calibrated,
abstaining, honestly-bounded posture is the only one the evidence supports. This is exactly the lesson the
deception project already learned.

## Disposition table (all 14 critiques)

| ID | Severity | Critique (short) | Disposition | v2 implementation |
|---|---|---|---|---|
| C1 | Critical | Manipulability M validated only against itself | IRREDUCIBLE | Renamed "simulator sensitivity under assumption set Theta"; no per-person field number; circularity reported as the finding |
| C2 | Critical | M is a point estimate of a conditional mean, not a bound | REFRAME | Dropped "bound"; renamed "sensitivity estimate under stated influence distribution"; norm-choice sensitivity published |
| C3 | Critical | Averaged-CIT d*=1.59 is the wrong-paradigm ceiling for live detection | REFRAME | H2 recalibrated off naturalistic continuous-EEG decoding; coverage is the headline |
| C4 | Critical | No field ground truth for "being manipulated" | IRREDUCIBLE | Report NO field AUC; field validity declared "unmeasurable" not unmeasured; simulator-PPV only |
| C5 | Critical | No adaptive adversary in the floor derivation | FIX | d* is the minimax saddle argmin_d max_u S(d,u); adaptive reallocation is a pre-registered falsification |
| C6 | Critical | Abstain gate is a covert oracle an adversary can titrate against | FIX | Adversary-observes-output model; boundary-hugging meta-detector; hidden/randomized gate thresholds; residual leakage disclosed |
| C7 | Critical | "Provably fails" imports preconditions a brain violates | REFRAME | Struck "provable" for the human setting; confined to in-silico connectome; renamed "simulated countermeasure ordering" |
| C8 | Critical | Consent is defeated by the instrument's own physics | IRREDUCIBLE | Enforced at API boundary: no per-individual H1/H2 output in asymmetric-power settings; abstain = refusal to run |
| C9 | Critical | No field error rate defeats Daubert; a crisp PPV invites misuse | REFRAME | Affirm no field error rate / PPV; label any PPV a simulator PPV; EPPA-equivalent screen at API boundary |
| C10 | Major | Entrainment (ITPC) indexes attention, not yielding | FIX | ITPC removed from H2 evidence; use N400 / late-positivity / prediction-error; ITPC retained only as labeled attention covariate |
| C11 | Major | State/trait confound re-imported by the floor | FIX | Compute d* against worst-case state; forbid stored per-person M across sessions; forbid ranking people |
| C12 | Major | M calibrated on a benign, not hostile, operator class | REFRAME | Relabeled "sensitivity to the simulated benign-persuasion operator class"; low M is explicitly not clearance |
| C13 | Major | The crown-jewel channel map is an offensive product | FIX | Emit only a defensive-sufficiency verdict, never a per-channel exploitability score |
| C14 | Major | The floor is estimated from a substrate the adversary can suppress | FIX | Couple H2 fragility into d* uncertainty; force d* upward or ABSTAIN when the signal is suppressed |

## Net effect

The red team did not invalidate WARDEN. It converted WARDEN from a tool that *claimed* to measure
manipulability into a tool that *reasons* about it under stated assumptions and says so. The three sentences
v2 is now required to say out loud, and enforce architecturally, are the contribution:

1. The manipulability index is assumption-relative and has no field referent (C1, C2, C12).
2. Field validity is unmeasurable, not merely unmeasured (C4, C9).
3. Consent is defeated in adverse contexts, so WARDEN refuses to run there (C8).

The discipline is unchanged from the deception project: honest calibration, mandatory abstention, base-rate
realism, an explicit and now adaptive threat model, no overclaiming, falsifiable pre-registered predictions,
in-silico-first with real data fixing simulator parameters only, and enforced dual-use handling.
