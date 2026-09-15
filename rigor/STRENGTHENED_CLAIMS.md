# Strengthened, defensible claim set (post math-framework + FIM/geometry/identification experiments)

2026-08-16. Hostile-reviewer pass over `SYNTHESIS_STRESSTESTED_2026-08.md`,
`MATH_FRAMEWORK.md`, and the three new artifacts (`fim_spectrum_results.json`,
`identification_rigor_results.json`, `digitalbrain_geometry_results.json` + figures). For each
central claim: the verdict after the new math and experiments, the exact formalism now attached,
the single remaining threat, and the exact next test. Ends with the corrections this pass enforces
and the statements that must be dropped or softened. No em dashes.

Verdict scale: SURVIVES (unchanged, still holds), STRENGTHENED (new evidence tightens it),
FAILS (does not hold as stated).

---

## CLAIM 1 - Identification is structural and cheap. Verdict: STRENGTHENED.

**Defensible statement.** On real connectomes (ABIDE CC200), individual identification is at ceiling
and is carried by coarse structure: a binarized connectome identifies as well as the full weighted
one (committed N=248: 97.98% vs 97.17%, gap 0.8 pts; fresh independent N=75 re-fetch: both 100%),
and the top subject-Gram eigenvalue (22.25) sits far above a random-matrix null (mean 1.12, p95 1.13).
Identification survives regressing out each subject's mean node strength (a motion/global-SNR proxy).

**Formalism now attached.**
- Information geometry (Sec 1): the "which subject" label is a coarse projection onto the top
  *stiff* FIM eigendirections, which carry the smallest Cramer-Rao variance and are estimable at
  small N. Identification is a *practically identifiable* coarse coordinate (Sec 4).
- RMT / Marchenko-Pastur (Sec 6): the correct null for "subject structure beyond finite-sampling
  noise" is MP, not the label-shuffle floor. The real top eigenvalue exceeds the empirical
  random-Gaussian null (implemented because N=75 << D=19900 makes the asymptotic MP edge the wrong
  comparison).
- Equivalence testing (Sec 7): binarized ~ weighted is an *equivalence* claim, made with TOST
  against a 5% margin (substantive evidence: committed N=248, 95% CI on the gap [-1.9%, +3.5%],
  includes 0), not with a difference test.
- Manifold capacity (Sec 5): identification rides *center separation* of the per-subject manifolds,
  which is coarse and structural.

**Single remaining threat.** The within-session confound. Every ABIDE arm (committed and fresh) is a
split-half of a *single* resting-state scan, so a stable within-scan motion/scanner/SNR signature is
shared by both halves and is indistinguishable from trait connectivity. Neither the RMT null (which
only rejects iid noise; within-session nuisance is also non-random subject structure) nor the
node-strength regression (which removes only the gross scalar version, and is ceiling-limited here so
its "0.0 pt drop" is uninformative) separates trait from nuisance. ABIDE cannot fix this: there is no
independent second acquisition.

**Exact next test.** HCP test-retest between-session fingerprinting: REST1 vs REST2 (different days),
regressing each session's own mean framewise displacement and global signal out of that session's
edges *before* computing FC, then (framework P6a) projecting each fingerprint onto the supra-MP
signal subspace vs the MP bulk. Pass = identification survives session-to-session on the signal
subspace. `wiring-not-weights/exp05_hcp_wiring_vs_weights.py` already targets this; blocked on gated
HCP Aspera access, not on method.

---

## CLAIM 2 - Human reconstruction-individuation is UNDETERMINED. Verdict: STRENGTHENED (the correction is now doubly supported).

**Defensible statement.** Human stimulus-evoked individuation is a powered open question, not a
result in either direction. The two arms are underpowered and neither survives correction:
`digital-brain` (N=4, chance 25%) gives 0/25 ROIs surviving BH-FDR or Bonferroni on the committed
p-values; `wiring-not-weights/exp04` (N=8, chance 12.5%) gives 0/25 with a per-ROI MDES of ~0.59 vs
observed 0.175. The new geometry reanalysis adds that even the *permutation-based* positive is an
artifact: the subject-respecting permutation reports 25/25 ROIs surviving BH-FDR, but with N=4 there
are only 4! = 24 label arrangements, so the achievable p floor is ~1/24 = 0.042, every ROI piles at
that floor, and BH-FDR does not protect against a shared floor. Bonferroni gives 0/25. The MDES is
0.83 at N=4 (the design can only detect near-perfect individuation) and 0.50 at N=8.

**Formalism now attached.**
- Structural vs practical identifiability (Sec 4): the individuating combinations are *structurally*
  identifiable given the ReLU quotient (Claim 3 proves the effect exists noise-free) but
  *practically* non-identifiable at N=4 or N=8 (flat profile likelihood).
- Cramer-Rao floor (Sec 1, P1b): individuating-parameter variance scales as
  sigma^2 / (N * lambda_a) for a *sloppy* (tiny) eigenvalue lambda_a, so the observed nulls are the
  direct consequence of small individuating eigenvalues, not of no effect.
- MDES ~ 1 / sqrt(N * lambda_stiffest-individuating) (Sec 7): the null is the statistical shadow of
  the FIM sloppy floor.

**Single remaining threat.** It is a null of unknown meaning, and we do not yet even have a target N:
the FIM-derived reconstruction sample threshold N* (framework P4) has not been computed, so the
"properly powered design" has no principled N attached to it yet.

**Exact next test.** First compute the between-individual FIM on the human reconstruction twin to get
the smallest individuating eigenvalue and hence N* (framework P4). Then run the preregistered powered
test at N > N*: pooled public 7T/3T, between-session split, BH-FDR, subject-respecting permutation
with N large enough that the permutation p floor (1/N!) sits below 0.05, reported MDES, tied to one
behavioral endpoint already in the program (neurobridge choice or spikeprint choice-distortion).

---

## CLAIM 3 - In simulation, reconstruction-identity IS the functional-equivalence class of the weights. Verdict: STRENGTHENED (existence proof), with its STRONG unification form still UNTESTED.

**Defensible statement (existence proof, holds).** In the noiseless apparatus the dissociation is
exact and now has a direct information-geometric signature: the linear null space is numerically
exactly flat (max null FIM eigenvalue = 0.0), and the ReLU continuous positive-rescaling generator is
numerically flat (v^T F v / (trace(F)/n_theta) ~ 1e-16 to 1e-17, pooled over 24 seeds and all hidden
units). Reconstruction rides the functional weight-function (seed-robust alpha* = 0.642, sd 0.008)
and is exactly flat in the null-space and permutation-scaling directions (within-class degeneracy
1.000, sd 0.000).

**Formalism now attached.**
- Lie-group quotient (Sec 3): identity is the orbit [theta] in M = Theta/G, G = S_H semidirect
  (R>0)^H; the metric that matters is the orbit distance d_M, never ||dtheta||. This is what fixes
  the d=33 artifact: PERM_SCALE has d_M = 0 exactly, so any Euclidean-in-Theta effect size is
  category-mistaken. Report "perfect separation in simulation."
- FIM kernel = symmetry (Sec 1, 3, P1a): the measured exact-zero / flat directions ARE the symmetry
  orbit, confirmed with zero free parameters.

**Single remaining threat (why the STRONG form is not yet a claim).** The framework's flagship
unification, that alpha* = f* = the stiff-eigenvalue cumulative crossing = the reconstruction MDES,
one constant ~0.642 from four formalisms (Sec 9, P8a), is UNTESTED, and the single attempt in
`fim_spectrum_results.json` is null for two independent spec reasons, both verified in code:
1. **Wrong stimulus design.** The FIM script (`exp_fim_spectrum.py:78`) draws isotropic stimuli
   `X[:, :k] = randn`, whereas the actual apparatus `exp03a` (lines 34, 52) draws anisotropic
   `X[:, :k] = randn * sqrt(linspace(1.0, 0.02, 40))`. The reported linear stiff range
   [0.729, 1.316] is exactly the Marchenko-Pastur bulk for gamma = 40/2000 = 0.02 (edges
   [0.737, 1.303]), i.e. pure sampling noise, not the apparatus's designed 1.7-decade spectrum. So
   the FIM script measured a different, well-conditioned object than the one alpha* was defined on.
2. **Wrong statistic.** The script reports the 50%-eigenvalue-*mass* crossing (linear 0.275, ReLU
   0.136), but the framework's prediction is the accuracy-*halfway* crossing (Sec 2: "the fraction
   m*/k at which cumulative stiff information drives identification across half"). These are
   different quantities: even on the correct exp03a linspace spectrum the 50%-mass crossing is 0.30,
   still not 0.642, because the mass crossing was never the predicted construct.
Therefore the 0.367 (linear) and 0.506 (ReLU) "mismatches" are artifacts of config plus statistic
mis-spec and are null information about the unification, neither confirming nor refuting it.

**Exact next test.** Re-run the FIM on the *identical* exp03a config (d=64, k=40, p=8, N=50,
anisotropic stimuli `randn * sqrt(linspace(1,0.02,40))`), and compute alpha* as the framework
defines it: form the cumulative retained Fisher-SNR curve C(m), push it through the actual ridge
read-out (the exp03a/exp06 decoder), and read the accuracy-halfway crossing. Only a crossing at
~0.642 on that identical config tests P8a. Then run the decisive Sec-9 experiment on one real
connectome-constrained, scanner-free ensemble (flyvis or the Beiran RNN), predicting the exact-zero
count, alpha*, f*, and N* from the single spectrum before running the tasks.

---

## CLAIM 4 - In artificial networks, brain-like structure is emergent, not causal. Verdict: SURVIVES (unchanged; still the cleanest result, but the new math adds no rigor here).

**Defensible statement.** Imposing brain-like attention topology on a transformer hurts performance:
`neuro-ai-pipeline`, 120 runs, 20 seeds, capacity-matched, Cohen d = -1.20, one-tailed p = 0.9997
for "helps"; brain-like heads are ~8.6x redundant. This is the program's most seed-stable negative
and generalizes the biological lesson: copy computation and objective, not anatomy.

**Formalism now attached.** Only framing, not machinery. The FIM / sloppy-model apparatus formalizes
the reconstruction *twin*, not the transformer, so the math framework does not add a rigorous
derivation to Claim 4. State it as an empirical negative that is *consistent with* the spine's
"structure is a byproduct" reading, not as a consequence of the FIM formalism.

**Single remaining threat.** Generality. One architecture, one task family, one operationalization of
"brain-like topology." A reviewer will say the penalty may be specific to this particular graph or
task, not to brain-mimicry per se.

**Exact next test.** Replicate the capacity-matched topology penalty on a second architecture and a
second task (or a scale sweep), plus an ablation isolating which topological property (small-worldness
vs modularity vs degree distribution) drives the penalty, so the result is "brain-likeness hurts,"
not "one bad graph hurts."

---

## Corrections this pass enforces (carry into every downstream doc)

1. **Do NOT present the unification (Sec 9 / P8a, "one number, four derivations", 0.642) as
   demonstrated.** It is a proposed test. The one attempt is null due to config + statistic mis-spec
   (Claim 3 threat). Downgrade `MATH_FRAMEWORK.md` Section 0 and Section 9 from "the organizing
   result" to "the central conjecture and its decisive test."
2. **Reattribute FIM caveat #2.** "The linear apparatus stiff block is well-conditioned, not sloppy
   (0.26 decades)" is a property of the *mis-configured reimplementation* (isotropic stimuli), not of
   the apparatus. On exp03a's anisotropic `linspace(1,0.02,40)` design the excited block spans ~1.7
   decades. The [0.729, 1.316] range is the MP noise bulk, and should be labeled as such.
3. **Do NOT cite the digital-brain "25/25 survive BH-FDR (permutation)" number as reconstruction
   support.** It is a permutation-floor artifact (N=4 -> 24 arrangements -> p floor 0.042); Bonferroni
   is 0/25. It reinforces "undetermined," and it is a 1NN fingerprinting (identification) probe, not a
   reconstruction probe.
4. **Do NOT headline the fresh N=75 identification TOST (p=0.0000) or the node-strength "0.0 pt
   drop."** Both arms are at ceiling (1.0), so the TOST se (1e-6) and the zero drop are ceiling
   artifacts. The substantive equivalence evidence is the committed N=248 CI [-1.9%, +3.5%].
5. **Label the RMT null as necessary, not sufficient.** Passing it rejects only the iid-noise null; it
   does not separate trait connectivity from within-session motion/scanner structure.
6. **ReLU "20 decades" is span-sloppy, not clean log-uniform.** The KS-vs-log-uniform test rejects
   (p ~ 0); the span is dominated by near-zero symmetry and near-permutation directions. Call it
   "hierarchically degenerate," not textbook Transtrum sloppiness.
7. **Keep the standing numeric corrections.** Never cite d = 33 (sd -> 0 artifact; use orbit distance
   / perfect separation) or alpha* = 0.8 (grid artifact; use 0.642 from exp06). Replace the ABIDE
   perm_p = 5e-4 between-arm claim with the MP null + TOST.

## What is genuinely new and defensible after this pass

- The exact-zero kernel = symmetry orbit confirmation (P1a): a clean, zero-parameter positive that
  tightens Claim 3's language from slogan to measured geometry.
- Claim 1's independent fresh-sample replication + RMT null + node-strength survival: real
  strengthening, gated only by the (unresolved) within-session confound.
- The doubly-supported "undetermined" verdict for Claim 2, now robust to the permutation-floor
  loophole as well as to FDR.
- The methodological spine remains the program's most publishable asset: it caught the BOLD5000
  overclaim, the N=8 overclaim, the N=4 FDR failure, and now (this pass) the alpha*-unification
  mis-spec, before any of them reached a paper.
