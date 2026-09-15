# Scope: Experiment 04, the Real-Data MVE (and the weight-proxy resolution)

Aayush Gandhi. 2026-06-19. No em dashes. Gated on Exp01-03b passing (they have).

Exp01-03b validated the apparatus on synthetic ground truth (linear and nonlinear/compensatory degeneracy).
The one load-bearing undefined item before real data is Gate 0c: on real human brains there are NO measured
synaptic weights, so what stands in for "wiring" versus "weights"? This doc resolves that and specs the MVE.

## 1. The wiring-vs-weights operationalization on real data

We cannot measure synaptic weights in living humans. We define a model-level proxy and state its validity
limits honestly.

- **Wiring (architecture) proxy** = (i) a SHARED backbone encoder (a brain foundation model, Brain-JEPA or
  BrainLM, or a shared linear/ridge encoder for the minimal version) PLUS (ii) the subject's own
  structural/functional CONNECTOME used as conditioning. This is the population-level, individual-invariant
  scaffold: the thing Nectome's EM traceability and TVB-style models capture.
- **Weights (individuating function) proxy** = the PER-SUBJECT parameters fit on top of the shared backbone
  to reproduce that subject's responses: a low-rank / adapter-style weight delta theta_i (LoRA-style), or a
  per-subject readout head. theta_i is the individuating weight-function. It is the residual individual
  signal BEYOND what the connectome and shared backbone already predict.

Read precisely: the connectome is treated as wiring; the individuating adapter is treated as weights. This
matches the program thesis and lets the same four-arm ablation run on real data.

## 2. The four arms (mirror of the synthetic apparatus)

- **FULL**: backbone + subject connectome conditioning + theta_i.
- **WITHIN (degeneracy)**: theta_i transformed by a function-preserving reparameterization (adapter
  permutation/scaling symmetry, exactly as Exp03b PERM_SCALE; or resample theta in directions with zero
  held-out gradient). Predictions on held-out stimuli unchanged. Predicted: identity preserved.
- **OUTSIDE**: theta_j from a different subject, or theta_i corrupted in functional directions. Predicted:
  identity lost.
- **WIRING-only**: backbone + connectome + GROUP adapter theta_pop (no individual weight-function).
  Predicted: identity at the connectome-fingerprint level only (see axis 1), not at reconstruction.

Capacity control: all arms share adapter dimensionality and parameter count; the only difference is whether
theta carries the individual's information. The graded curve (Exp03a) becomes: alpha = fraction of the
individual adapter dimensions retained.

## 3. Identity axes (pre-commit ONE primary; report the rest)

- **Axis A, connectome fingerprint (Finn 2015 replication).** Identify subjects from functional connectomes.
  This is the WIRING-level, sense-1 (identification) baseline. Expectation: wiring DOES identify people
  (~90%). This is important to report because it shows wiring is sufficient for identification yet (per the
  next axis) not for reconstruction. The two axes together are the whole thesis.
- **Axis B (PRIMARY, pre-committed), reconstruction.** Match a subject to their twin by held-out response
  prediction, SNR/noise-ceiling matched, capacity matched, >= 10 seeds, effect size + CI. This is where the
  arms separate: FULL ~ WITHIN >> OUTSIDE ~ WIRING.

The headline real-data result is the DISSOCIATION between Axis A (wiring identifies) and Axis B (wiring does
not reconstruct), plus the Exp03a-style sufficiency curve over adapter dimensions.

## 4. Controls (carried from the synthetic program, now mandatory on noisy real data)

- SNR / noise-ceiling: per-voxel reliability ceiling; never claim identity above what reliability supports.
- Capacity match: equal adapter DoF across arms (assert).
- Multiple comparisons: one pre-committed axis (B); others are secondary.
- Degeneracy: WITHIN uses a function-preserving reparameterization, never uniform randomization.
- Honest abstention: conformal coverage on every identity verdict (neurobridge layer).

## 5. Data

| Dataset | Role | Status |
|---|---|---|
| HCP (~1000, rest + task) | connectome manifold + Axis A fingerprint (wiring identification) | public, large download |
| NSD (8 subj, 7T, dense) | Axis B response substrate; per-subject encoders/adapters | public; raw deleted locally, re-download (see digital-brain/data/RAW_DATA_REMOVED.md) |
| NSD-imagery / NSD-synthetic (same 8) | twin generalization (does the adapter predict the SAME person on new stimulus types) | public |
| Courtois NeuroMod (Friends, n=6, deep) | alternative deep per-subject response substrate | public |

Data-matching caveat: the connectome (wiring) and the dense naturalistic responses (weights) often come
from different subjects/datasets. The clean fusion needs subjects with BOTH. Two honest paths: (a) HCP has
both resting (connectome) and task fMRI in the same subjects -> run the full pipeline within HCP using task
responses; (b) run Axis A on HCP and Axis B on NSD separately, and treat the fusion as a later step. The MVE
below takes the minimal, self-contained path.

## 6. The MVE (smallest self-contained real-data step)

No foundation model required. Minimal weight proxy = per-subject ridge readout.
1. NSD shared-image responses, per ROI. Fit a SHARED encoder (group) and a per-subject delta (the adapter =
   weight proxy). Capacity matched across arms.
2. Run Axis B (reconstruction identification) with the four arms (FULL / WITHIN / OUTSIDE / WIRING),
   SNR-matched via the NSD noise ceiling, >= 10 seeds, effect size + CI.
3. Produce the Exp03a sufficiency curve over adapter dimensions on real responses.
4. Separately, Axis A: replicate Finn connectome fingerprinting on HCP (the wiring-identifies baseline).
5. Report the dissociation (A: wiring identifies; B: wiring does not reconstruct) with conformal coverage.

Gate (real-data, pre-registered): WITHIN ~ FULL (degeneracy invariance) AND FULL >> OUTSIDE ~ WIRING on
Axis B, SNR- and capacity-controlled, replicated across seeds and >= 2 ROIs/datasets. Kill: if WIRING
matches FULL on Axis B, the connectome is reconstruction-sufficient and the thesis is wrong on real data
(publishable either way).

## 7. The honest caveat that must travel with every real-data claim (Gate 0c provenance)

The adapter weights are a MODEL-LEVEL proxy for the individuating parameters, NOT measured synaptic weights.
The real-data claim is about identity-relevant individuating parameters at the model level. The inference
from "model adapter weights" to "biological synaptic/molecular weights" is exactly the resolution-ladder
question (Sandberg & Bostrom), and is bracketed, not asserted. This keeps the result defensible: we show
that wiring (connectome) identifies but does not reconstruct, and that reconstruction needs an individuating
weight-function up to degeneracy; we do NOT claim to have measured synapses.

## 8. Blockers before Exp04 can run

1. Data download (HCP connectomes; NSD derived features/betas; re-download per RAW_DATA_REMOVED.md).
2. Choice of backbone for the non-minimal version (Brain-JEPA / BrainLM weight availability). The MVE avoids
   this with a ridge readout.
3. Compute for per-subject fitting at scale (modest for the ridge MVE; larger for foundation-model adapters).

## 9. Recommendation

Run the MVE (section 6) on NSD as soon as the NSD derived features are re-downloaded; it is self-contained,
needs no foundation model, and directly tests the thesis on real human responses with the apparatus already
validated in Exp01-03b. Axis A (HCP Finn replication) can proceed in parallel and is low-risk. The fused
within-subject connectome+response pipeline (HCP task fMRI) is the follow-up once the MVE pattern holds.
