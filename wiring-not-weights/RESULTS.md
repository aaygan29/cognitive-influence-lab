# Results: Experiments 01-02 (apparatus validation). 2026-06-19. No em dashes.

Status: APPARATUS VALIDATED on synthetic ground truth. This validates the instrument, NOT a claim about
real brains. The real-data MVE (HCP wiring + NSD responses + a stated weight proxy) is the next gate and is
unlocked by these passing.

## Exp01 (`exp01_apparatus_validation.py`, results/exp01_apparatus_results.json)
Four capacity-matched arms (all 8192-param p x d maps), reconstruction identity metric, N=40 (chance
0.025), R=20 random cohorts x S=10 seeds.
- FULL 1.000, WITHIN 1.000, OUTSIDE 0.000, WIRING 0.025, NULL 0.022. Ordering replicates 20/20. All 7
  pre-registered checks PASS; APPARATUS_VALID=True.
- Two self-caught problems (the world was too easy): (1) OUTSIDE=0.000 was a cyclic-mismatch artifact
  (identity replaced by a SPECIFIC other), not true chance; (2) WITHIN=FULL exactly was a tautology
  (perturbing an exact zero-variance null space). Both fixed in Exp02.

## Exp02 (`exp02_stress_and_controls.py`, results/exp02_stress_results.json)
Independent verification with corrected controls: OUTSIDE = fresh-random non-existent person (-> true
chance); degeneracy made APPROXIMATE (anisotropic stimuli with decaying eigenvalues; WITHIN_APPROX perturbs
the lowest-variance functional directions). Sweep over SNR in {0,3,6} dB x individual-signal in {0.3,0.7},
R=10 x S=8 per regime.
- Predicted ordering FULL ~ WITHIN_APPROX >> OUTSIDE ~ WIRING ~ NULL ~ chance holds in ALL 6 regimes,
  100% cohort replication.
- OUTSIDE 0.023-0.029, WIRING 0.025, NULL 0.019-0.031: all at chance. Confirmed.
- WITHIN_APPROX 0.915-0.960 at weak signal (isc=0.3), 1.000 at strong (isc=0.7): graceful, signal-dependent
  near-invariance to within-manifold weight changes. This is the meaningful, non-tautological degeneracy
  result.
- Primary contrast FULL vs OUTSIDE: Cohen's d 125-242 (separation is total, by design when truth is
  planted).

## Honest limitations (carry forward; do NOT overclaim)
1. FULL sits at ceiling (1.000) even at 0 dB because the identity metric integrates over p=128 x n_test=200
   dims -> huge effective SNR. The world is "easy"; this validates plumbing and the conceptual ordering, not
   a graded identity-sufficiency CURVE. Next: a sub-ceiling regime (fewer response dims / test stimuli /
   higher noise) to get a real curve.
2. Linear-Gaussian world. Degeneracy is modeled as low-variance LINEAR directions, not Marder's actual
   NONLINEAR compensatory degeneracy (different conductances, same output via nonlinear interaction). Next:
   nonlinear twins.
3. Synthetic. The weight "proxy" for real human twins (Gate 0c) is still undefined and is the load-bearing
   open item before the real MVE.

## Exp03a (`exp03a_identity_sufficiency_curve.py`, results/exp03a_curve_results.json) -- THE CURVE
Sub-ceiling regime via bisection on a continuous difficulty knob (individual-signal scale) -> FULL lands at
0.78 (chance 0.020, N=50), R=15 cohorts x S=8 seeds. alpha = fraction of individuating weight-DIMENSIONS
retained (ranked by deviation energy; the meaningful "how much of the weight-function must you store" axis).
- Graded, monotonic, CONVEX identity-sufficiency curve:
  alpha   0.00 0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.80 1.00
  id-acc  0.02 0.02 0.03 0.03 0.05 0.08 0.14 0.22 0.33 0.57 0.78
  Minimal sufficient fraction alpha* (halfway to FULL) = 0.8. Reading: in this world identity is
  DISTRIBUTED and high-dimensional -- you must retain most individuating weight-dims to recover it; a few
  top features do not suffice. (This SHAPE is the empirical quantity; the value here is a property of the
  synthetic signal distribution, not a claim about real brains. The apparatus now measures it.)
- Controls: beta_null (exact null-space/degeneracy weights) PERFECTLY FLAT at 0.78 across all perturbation
  magnitudes -> exact degeneracy invariance. beta_lowvar (approx-degeneracy functional dims) collapses
  0.78 -> 0.07. functional_equivalence_confirmed = True (alpha range 0.762, null range 0.000).
- Two self-caught mis-tunings before landing the band: too-hard regime (FULL 0.089, no range) and
  noise-limited saturation (alpha* = 0.05, step not curve). Fixed by bisection calibration + switching
  alpha from signal-scaling to dimension-retention.

## Headline across Exp01-03a
The apparatus is validated, replicates across random cohorts and regimes, and now emits a graded,
interpretable identity-sufficiency curve with a perfectly flat exact-degeneracy control. The conceptual
claim "identity = the functional-equivalence class of the weights, not the wiring and not the exact weights"
is demonstrated on synthetic ground truth. What remains is realism, not plumbing.

## Exp03b (`exp03b_nonlinear_degeneracy.py`, results/exp03b_nonlinear_results.json) -- MARDER TEST PASSED
Nonlinear ReLU-MLP twins; compensatory degeneracy via EXACT hidden-unit permutation + positive rescaling
(function-invariant ReLU symmetry). N=50, chance 0.020, R=15 x S=8. NONLINEAR_DEGENERACY_CONFIRMED=True
(8/8 checks, ordering replicates 15/15).
  FULL 1.000 | PERM_SCALE 1.000 | FUNC_PERTURB 0.243 | FUNC_RAND 0.020 | WIRING 0.020 | NULL 0.026
- PERM_SCALE changes weights by ||dW||/||W|| = 1.57 (more than their own norm) yet identity is UNCHANGED
  (|FULL-PERM| = 0.0000). A FUNC_PERTURB of the SAME magnitude (1.57) but in functional directions drops
  identity to 0.243 (Cohen's d = 33). FUNC_RAND (function replaced) and WIRING (group) at chance.
- This is the honest gradient: invariant to compensatory weight change, degrades with functional change of
  equal size, vanishes when function is replaced/absent. Marder-style degeneracy handled.
- Self-caught: first run used a mis-specified criterion (expected FUNC_PERTURB at chance; it is a partial
  perturbation so it sits above chance). Fixed by adding FUNC_RAND (true-chance control) and correcting the
  predicted ordering to FULL~PERM_SCALE >> FUNC_PERTURB > FUNC_RAND~WIRING~chance.

## Exp04 SCOPED (`SCOPE_exp04_real_mve.md`) -- weight proxy resolved
Resolved Gate 0c: on real brains there are no measured synaptic weights, so wiring = shared backbone +
subject connectome (conditioning); weights = per-subject adapter theta_i (LoRA-style delta / ridge readout),
the individuating residual beyond the connectome. Two axes: A = connectome fingerprint (Finn, wiring
IDENTIFIES, sense 1); B (primary) = reconstruction (wiring does NOT reconstruct). The dissociation A vs B is
the real-data headline. MVE = per-subject ridge readout on NSD shared images, four arms, SNR + capacity
controlled, conformal coverage; Axis A = HCP Finn replication in parallel. Honest caveat travels with every
claim: adapter weights are a MODEL-LEVEL proxy, not measured synapses; the model->biology mapping is the
resolution-ladder question and is bracketed. Blockers: NSD/HCP re-download; backbone choice (MVE avoids it).

## Exp04 (`exp04_real_pilot_nsd.py`, results/exp04_real_pilot_results.json) -- REAL DATA, HONEST NEGATIVE
Ran on REAL NSD N=8 (260 shared images, 25 ROIs) using LOCAL data only (no download):
~/Downloads/_n8work/matrices/ + digital-brain/data/features/algonauts_shared268_*. Stimulus features
X=(260,1024); per-subject ridge encoders; voxel-invariant fingerprints (functional = predicted-RDM;
weight = encoder feature-tuning coef^T coef). Chance = 1/8 = 0.125.
- FUNCTIONAL axis (predicted-RDM): mean acc 0.175, mean margin +0.025, 0/25 ROIs p<.05. ~At chance.
- WEIGHT axis (encoder feature-tuning): mean acc 0.205, margin +0.011, 1/25 ROIs p<.05 (VWFA-2 p=.019, does
  NOT survive 25-test correction). ~At chance.
- Degeneracy machinery verified: null-space weight perturbation changed PREDICTIONS by 7e-15 (provably
  function-invariant); weight-fingerprint margin essentially unchanged (0.0114 -> 0.0113), but that is moot
  because there is no above-chance identity signal to confound at N=8.
- HONEST CONCLUSION: at N=8 neither functional nor weight fingerprint individuates above chance. This
  REPRODUCES the digital-brain N=8 null ("encoding is not identity"). The apparatus runs correctly on real
  data; N=8 is simply underpowered to resolve the wiring-vs-weights dissociation. Do NOT claim identity from
  this pilot. (Consistent with the program rule: verify before reporting; no fingerprinting overclaims.)

## The powered test needs more subjects (download required)
NSD has only 8 subjects total, so it caps here. Resolving the dissociation needs many subjects:
- HCP (~1000, connectomes + task fMRI, same subjects) is the right dataset. BLOCKER: HCP requires a
  ConnectomeDB account + acceptance of Open Access Data Use Terms -> cannot be done autonomously (account
  creation / terms acceptance). Needs the user.
- Credential-free large-N alternatives for Axis A (connectome fingerprint) exist (OpenNeuro / NKI-Rockland /
  ABIDE resting-state) and could be scripted, but lack matched naturalistic responses for Axis B.

## Exp05-ABIDE (`exp05_abide_wiring_vs_weights.py`, results/exp05_abide_results.json) -- POWERED REAL DATA
Pivoted off HCP (gated, Aspera-only, could not complete autonomously) to ABIDE Preprocessed on the PUBLIC
fcp-indi S3 bucket (anonymous, no login). CC200 atlas -> 200 common-space ROIs; downloaded 300 subjects,
248 usable. Within-subject split-half fingerprinting, three arms. chance=0.004.
  FULL_weighted 0.980 | WIRING_binarized 0.972 | NODE_STRENGTH 0.446  (all perm-p 0.0005)
  FULL - WIRING = +0.008.
- RESULT: binarized topology fingerprints people almost as well as the weighted connectome (97.2 vs 98.0%).
  Coarse topology (node strength) is insufficient (0.446); thresholded edge pattern is sufficient.
- HONEST INTERPRETATION (critical): this is an IDENTIFICATION test (sense 1), the axis the thesis ALREADY
  assigns to wiring (Finn). It does NOT test RECONSTRUCTION/constitution (sense 2/3), the thesis crux. So it
  is CONSISTENT WITH the framework but does NOT adjudicate the core claim. The code's auto-flag
  "kill_topology_sufficient=True" is keyed to a connectome-IDENTIFICATION framing and must NOT be read as
  killing the (reconstruction-level) thesis. What it cleanly shows: identification is a topology property,
  and even sign-thresholded topology (no edge magnitudes) suffices.
- CAVEATS: within-session split (inflates vs cross-session Finn); multi-site ABIDE (scanner/site signature
  is a partial confound for fingerprinting). 
- IMPLICATION: the thesis crux needs a POWERED RECONSTRUCTION dataset (many subjects with stimulus-evoked
  responses). HCP-identification would just reproduce this sense-1 result; NSD-reconstruction tests sense-2
  but is N=8 underpowered. The powered-reconstruction substrate is the real open gap.

## Status of the program
Synthetic apparatus: VALIDATED end-to-end (Exp01-03b), including nonlinear/compensatory degeneracy.
Real data: apparatus runs (Exp04); N=8 underpowered -> honest null, reproduces prior. Powered confirmation
is a data-access problem (HCP credentialed download), not a method problem.
