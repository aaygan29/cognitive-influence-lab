# Pre-registration: Experiment 01, Apparatus Validation on Synthetic Ground Truth

Aayush Gandhi. 2026-06-19. No em dashes. Pre-committed before looking at any output.

## Purpose

Validate the wiring-vs-weights identity apparatus on synthetic data where the answer is known by
construction, BEFORE touching real (HCP/NSD) data. We are not testing a claim about real brains here. We
are testing whether the experimental machinery (the four-arm ablation, the reconstruction identity metric,
the capacity matching, the null control, the replication harness) behaves correctly when identity is, by
construction, carried by the individual weight-function up to degeneracy. If the apparatus cannot recover a
ground truth we planted, it cannot be trusted on real data. This is the program's standard
"validate-on-synthetic-ground-truth-first" gate.

## Generative ground truth (what is true by construction)

- Population of N individuals. Each has a stimulus-to-response linear map A_i in R^{p x d}.
- Stimulus space splits into a FUNCTIONAL subspace (first k dims, nonzero variance) and a NULL subspace
  (remaining d-k dims, zero variance, never excited). The null subspace is the degeneracy direction.
- A_i functional columns = A_pop[:, :k] + B_i, where B_i is the individual weight-function (the planted
  identity signal). Null columns carry no functional consequence.
- Observed responses Y_i = A_i X + observation noise (fixed SNR). Identity lives ONLY in B_i (the
  individual functional weights), NOT in the wiring/architecture (the shared support and A_pop) and NOT in
  the null-space weights.

## Twins and the four arms (capacity matched: every arm is a p x d map, identical parameter count)

Twins are FIT from training data by ridge regression (so the capacity confound is live and controlled, not
assumed away). Arms:

1. FULL: the individually fit twin Â_i.
2. WITHIN-manifold resample: Â_i with weights redrawn in the null subspace (degeneracy class). Predictions
   on test stimuli are unchanged by construction. Tests: does changing weights without changing function
   preserve identity? Predicted YES.
3. OUTSIDE-manifold resample: Â_i with the functional weights replaced by a mismatched individual signal
   (wrong person), magnitude matched. Tests: does changing the functional weights destroy identity?
   Predicted YES (identity lost).
4. WIRING-ONLY: the group-fit map Â_pop (shared architecture, no individual weight-function), identical
   across subjects. Tests: is the shared wiring/architecture alone identity-bearing? Predicted NO.

Capacity control: arms 1, 2, 3, 4 are all p x d and use the same number of free parameters. Any identity
advantage of FULL over WIRING is therefore NOT a parameter-count effect; it is individual weight
information. We assert equal parameter count in code.

## Identity metric (reconstruction axis, Finn-style)

For each subject i, take true held-out responses Y_i(test). For every twin Â_j, predict Ŷ_j = Â_j X(test).
Match subject i to the twin whose prediction best correlates with Y_i(test). Identification accuracy =
fraction of subjects matched to their own twin. Chance = 1/N.

## Controls

- NULL control (label shuffle): permute the subject-to-twin correspondence; identification accuracy must
  fall to chance for ALL arms, including FULL. Validates the metric is not inflated.
- SNR control: identical observation-noise process across all arms (arms differ only in the twin matrix).
- Capacity control: equal parameter count across arms (asserted).

## Statistical design and replication

- COHORTS: R independent random regenerations of the whole world (A_pop, B_i, stimuli). This is the
  "replication against other random datasets" requirement. Pre-set R = 20.
- SEEDS: within each cohort, S = 10 re-draws of noise and test stimuli.
- Per arm we report identification accuracy mean with 95% bootstrap CI across cohorts.
- PRIMARY contrast (single, pre-committed, to avoid multiple comparisons): FULL vs OUTSIDE identification
  accuracy. Report Cohen's d and 95% CI across cohorts.
- SECONDARY (pre-committed): WITHIN vs FULL equivalence (two one-sided tests / small standardized
  difference), and OUTSIDE vs WIRING both within chance CI.
- ORDERING check per cohort: PASS if (FULL and WITHIN clearly above chance) AND (OUTSIDE and WIRING within
  the chance band). Replication rate = fraction of cohorts passing.

## Pre-registered success and kill criteria

APPARATUS VALID if all hold:
1. NULL control identification accuracy CI includes chance and excludes high accuracy.
2. FULL identification accuracy >> chance (target > 0.80 at N=40, chance = 0.025), large effect.
3. WITHIN not meaningfully below FULL (standardized diff small; predictions identical by construction).
4. OUTSIDE and WIRING within the chance band.
5. Ordering replicates in >= 90% of the R cohorts.
6. Equal parameter count across arms (asserted true).

KILL (apparatus invalid, do not proceed to real data) if any of:
- FULL does not beat chance (metric or arm logic broken).
- OUTSIDE or WIRING achieves high identification accuracy (the planted ground truth is not what drives the
  metric; confound present).
- NULL control is above chance (metric inflated).
- Ordering replicates in < 90% of cohorts (apparatus is fragile to the random world).

## What this does NOT establish

This validates the instrument, not a claim about real brains. It shows the apparatus correctly detects
"identity is in the individual weight-function up to degeneracy" when that is true, and correctly reports
its absence under the null. The real MVE (HCP wiring substrate + NSD responses, with a stated weight proxy)
is the next experiment and is gated on this one passing.
