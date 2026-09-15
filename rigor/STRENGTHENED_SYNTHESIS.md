# Structure Identifies, Function Individuates: the strengthened synthesis

2026-08-16. Senior-author merge of `SYNTHESIS_STRESSTESTED_2026-08.md`,
`PORTFOLIO_CLUSTERS.md`, `MATH_FRAMEWORK.md`, `STRENGTHENED_CLAIMS.md`, and the three
new artifacts (`identification_rigor_results.json`, `fim_spectrum_results.json`,
`digitalbrain_geometry_results.json` + `figures/`). This is the reference the three
papers cite for claims, statistics, and named formalism. It supersedes the evidence
tables in the earlier synthesis where they conflict. No em dashes.

---

## 0. The thesis after the rigor pass

> What makes a neural system identifiable is coarse structure, and it is cheap. What
> individuates it (its function, its behavior) is a different and harder quantity, and
> current human data cannot yet resolve how hard. The same lesson holds in artificial
> networks: brain-like structure is not the performance lever, function is. Copy
> computation, not anatomy.

Three deliverables, not one: the dissociation (structure vs function), the honest power
accounting that keeps neither side over-claimed, and a control-first discipline that
caught the program's own false positives. The math framework is now the spine's
backbone, but exactly one of its results (the four-way unification) is a conjecture, not
a demonstration, and this document quarantines it as such.

---

## 1. The updated 3-paper arc, with the mathematics named per claim

The portfolio remains three theses and three papers. What changed: each claim now carries
a named formalism, a corrected statistic, and an explicit "existence proof vs powered
result vs conjecture" label. Recommended order unchanged: B (reviews in hand), A (results
+ figures exist), C (natural third).

### Paper A - "Structure Identifies, Function Individuates" (thesis T2 + AI leg T1)
Flagship. Merges the neural-identity spine with the AI echo as a cross-substrate section.
The math framework is Paper A's Methods/Theory section.

- **CLAIM 1 (identification cheap, structural).** Named rigor: **information-geometry
  stiff-FIM practical identifiability** (Sec 1, 4), **Marchenko-Pastur / RMT signal-vs-bulk
  null** (Sec 6), **manifold-capacity center separation** (Sec 5), **TOST equivalence**
  (Sec 7). Verdict: **STRENGTHENED**. Backed by the fresh independent N=75 re-fetch and the
  RMT null (`identification_rigor_results.json`, `figures/identification_rigor.png`,
  `fig2_identification_equivalence.png`).
- **CLAIM 2 (human reconstruction undetermined).** Named rigor: **structural-vs-practical
  identifiability / flat profile likelihood** (Sec 4), **Cramer-Rao sloppy-eigenvalue floor**
  (Sec 1, P1b), **MDES ~ 1/sqrt(N lambda)** (Sec 7). Verdict: **STRENGTHENED as a correction,
  now doubly supported** (survives both FDR and the permutation-floor loophole). Backed by
  `digitalbrain_geometry_results.json`, `figures/digitalbrain_geometry.png`,
  `fig1_digitalbrain_fdr.png`.
- **CLAIM 3 (functional-equivalence class in simulation).** Named rigor: **Lie-group
  quotient manifold** M = Theta/G, G = S_H semidirect (R>0)^H (Sec 3), **FIM kernel =
  symmetry orbit** (Sec 1, P1a), **rate-distortion on the quotient source** (Sec 8).
  Verdict: **STRENGTHENED existence proof**; its STRONG unification form (Sec 9 / P8a) is
  **UNTESTED conjecture**. Backed by `fim_spectrum_results.json`, `figures/fim_spectrum.png`.
- **CLAIM 4 (brain-like structure not causal in AI).** Named rigor: framing only; the FIM
  apparatus formalizes the reconstruction twin, not the transformer. Verdict: **SURVIVES,
  unchanged**. The cleanest negative (d = -1.20, 120 runs). State as consistent-with the
  spine, not derived from the FIM.

The one figure for Paper A: identification accuracy vs reconstruction-individuation accuracy
on one axis, each with its correct null (RMT for ID, MDES/permutation-floor for
reconstruction), honest that ABIDE (N=248/N=75) and NSD (N=4/N=8) are different datasets.
Headline: "at ceiling and binarization-equivalent for identification, vs a null we are not
yet powered to interpret for reconstruction." Target: eLife / Nature Communications / PNAS.

### Paper B - "An Instrument, Not a Scanner" (thesis T3 defense: cognitive neurosecurity)
WARDEN, reviewed at NeuroXR 2026 (2x Weak Accept, 1x Weak Reject). The reviewers' fix is
exactly this pass: reframe the N=8 "safe channel" as **undetermined, MDES-stated**, and use
Paper A's dissociation as the calibration case study (motion re-ID strong vs neural
individuation undetermined). Adds the RMT null as the per-channel "is this signal or
finite-T noise" gate. Named rigor borrowed: **conformal coverage + specificity gate**
(Sec 7), **MP null as channel risk test** (Sec 6). Target: NeuroXR camera-ready to journal.

### Paper C - "The Honest Boundary of Behavioral Neuroforecasting" (thesis T3 capability)
Population forecasting works with conformal coverage; individual influence is overclaimed.
Feeds `neurobridge`, `spikeprint` (choice distortion, kill criteria), `cultist` (belief
imparting). Named rigor borrowed: **rate-distortion S(f) bit-budget** (Sec 8) as the
preservation-claim converter, **conformal set size as the dual of S(f)** (P7b). Target:
cog-sci / AI-safety venue.

### Shared Methods (spine of all three): `neuro_ai_core`
Split-conformal intervals, specificity gate, provenance fingerprint, plus the
control-first, power-accounted, self-correcting discipline. Write once, cite from A/B/C.
Its track record is the most publishable asset: it caught the TRIBE length confound, the
BOLD5000 fingerprinting overclaim, the N=8 overclaim, the N=4 FDR failure, and now the
alpha*-unification mis-specification, before any reached a paper.

---

## 2. Per-claim evidence table (corrected statistics + backing artifact)

| Claim | Corrected statistic (use this) | Named formalism | NEW result / figure that backs it | Verdict |
|---|---|---|---|---|
| C1 identification cheap | Committed N=248: FULL 97.98% vs WIRING 97.17%, gap 0.8 pt; TOST 95% CI on gap [-1.9%, +3.5%] includes 0. Fresh independent N=75: both arms 100%, McNemar discordant=0 (p=1.0). RMT: top subject-Gram eigenvalue 22.25 vs random-matrix null mean 1.12 / p95 1.13 (empirical p=0.005). Node-strength regression: 1.0 -> 1.0 (drop 0.0). | Stiff-FIM practical identifiability; Marchenko-Pastur signal-vs-bulk; manifold center separation; TOST | `identification_rigor_results.json`; `figures/identification_rigor.png`, `fig2_identification_equivalence.png` | STRENGTHENED |
| C2 human reconstruction undetermined | digital-brain N=4 (chance 25%): 0/25 ROIs survive BH-FDR, 0/25 Bonferroni; committed p-values cluster [0.026, 0.048]. Independent fMRI-derived subject-respecting permutation: 25/25 "survive" BH-FDR is a **permutation-floor artifact** (N=4 -> 4!=24 arrangements -> p-floor ~1/24=0.042), 0/25 Bonferroni. MDES accuracy 0.832 at N=4, 0.503 at N=8 (design detects only near-perfect individuation). exp04 N=8: 0/25, per-ROI MDES ~0.59 vs observed 0.175. | Structural-vs-practical identifiability (flat profile likelihood); Cramer-Rao floor sigma^2/(N lambda_a); MDES ~ 1/sqrt(N lambda) | `digitalbrain_geometry_results.json`; `figures/digitalbrain_geometry.png`, `fig1_digitalbrain_fdr.png` | STRENGTHENED (doubly supported correction) |
| C3 functional-equivalence class (sim) | Linear null space exactly flat: max null FIM eigenvalue = 0.0. ReLU positive-rescaling generator flat: v^T F v / (trace/n) = -1.2e-17 (max 7.9e-16), pooled 24 seeds x all units. alpha* = 0.642 (sd 0.008, exp06); within-class degeneracy 1.000 (sd 0.000). **Unification UNTESTED:** the fim_spectrum attempt is null for two code-verified spec reasons (isotropic stimuli -> stiff range [0.729, 1.316] = MP bulk for gamma=40/2000; 50%-mass crossing computed, not the accuracy-halfway crossing). | Lie-group quotient M=Theta/G; FIM kernel = symmetry orbit; rate-distortion on quotient | `fim_spectrum_results.json`; `figures/fim_spectrum.png` | STRENGTHENED existence proof; STRONG form UNTESTED |
| C4 brain-like structure not causal (AI) | Cohen d = -1.20, 120 runs, 20 seeds, capacity-matched, one-tailed p = 0.9997 for "helps"; brain-like heads ~8.6x redundant. | Framing only (FIM formalizes the twin, not the transformer) | No new artifact; unchanged from `neuro-ai-pipeline` | SURVIVES |

---

## 3. Honest limitations and the experiments still needed

**Standing numeric corrections (carry into every paper).**
1. Never cite d = 33 (sd -> 0 artifact); report the G-invariant orbit distance and "perfect
   separation in simulation."
2. Never cite alpha* = 0.8 (coarse-grid artifact); use 0.642 (exp06).
3. Never cite the ABIDE perm_p = 5e-4 as a between-arm test; it is the each-arm-vs-chance
   floor. Use the MP null + TOST.
4. Do not headline the fresh N=75 TOST p=0.0000 or the node-strength "0.0 pt drop": both
   arms are at ceiling (1.0), so the TOST se (1e-6) and the zero drop are ceiling artifacts.
   The substantive equivalence evidence is the committed N=248 CI [-1.9%, +3.5%].
5. Do not present the Sec 9 / P8a unification ("one number, four derivations", 0.642) as
   demonstrated. It is a proposed test; the one attempt is null due to config + statistic
   mis-spec. Downgrade MATH_FRAMEWORK Sec 0 and Sec 9 from "the organizing result" to "the
   central conjecture and its decisive test."
6. Do not cite the digital-brain "25/25 survive BH-FDR (permutation)" as reconstruction
   support: permutation-floor artifact (N=4 -> p-floor 0.042); Bonferroni 0/25; and it is a
   1NN identification probe, not a reconstruction probe. It reinforces "undetermined."
7. Label the RMT null necessary, not sufficient: passing it rejects only the iid-noise null,
   not within-session motion/scanner structure. ReLU "20 decades" is span-sloppy
   (KS-vs-log-uniform p ~ 0, dominated by near-symmetry directions), call it
   "hierarchically degenerate," not textbook Transtrum sloppiness.

**The one live confound (blocks Paper A).** CLAIM 1's within-session confound is untouched by
all the new rigor. Every ABIDE arm (committed and fresh) is a split-half of a single
resting-state scan, so a stable within-scan motion/scanner/SNR signature is shared by both
halves and is indistinguishable from trait connectivity. The RMT null rejects only iid
noise; within-session nuisance is also non-random subject structure. The node-strength
regression removes only the gross scalar version and is ceiling-limited here. ABIDE cannot
fix this: there is no independent second acquisition.

**Powered / between-session experiments still needed.**
- **P6a (CLAIM 1 fix):** HCP REST1-vs-REST2 between-session fingerprinting, regress each
  session's own mean framewise displacement + global signal out of that session's edges
  before FC, then project onto the supra-MP signal subspace vs the MP bulk. Pass =
  identification survives session-to-session on the signal subspace.
  `wiring-not-weights/exp05_hcp_wiring_vs_weights.py` targets this; blocked on gated HCP
  Aspera access, not method.
- **P4 + powered run (CLAIM 2 fix):** compute the human-twin between-individual FIM to get N*
  (the smallest individuating eigenvalue), then the preregistered powered run at N > N*:
  pooled public 7T/3T, between-session split, BH-FDR, subject-respecting permutation with N
  large enough that the permutation p-floor (1/N!) sits below 0.05, reported MDES, tied to
  one behavioral endpoint already in the program (neurobridge choice or spikeprint
  choice-distortion).
- **P8a re-run (CLAIM 3 STRONG form):** re-run the FIM on the identical exp03a config
  (d=64, k=40, p=8, N=50, anisotropic stimuli randn*sqrt(linspace(1,0.02,40))) and compute
  alpha* as the accuracy-halfway crossing through the actual exp03a/exp06 ridge readout, not
  the eigenvalue-mass crossing. Only a crossing at ~0.642 on that config tests P8a.
- **CLAIM 4 generality:** replicate the capacity-matched topology penalty on a second
  architecture and second task, plus an ablation isolating which topological property
  (small-worldness vs modularity vs degree distribution) drives the penalty.

---

## 4. The single highest-value next experiment

**Compute the between-individual FIM on one real, scanner-free, connectome-constrained
ensemble (flyvis or the Beiran RNN), and from that single eigenspectrum predict, ahead of
running any task: (i) the exact-zero kernel count, (ii) alpha* via the accuracy-halfway
cumulative-stiffness crossing, (iii) f* and the gauge-adjusted bit-budget, (iv) the
reconstruction N* / MDES. Then run identification and behavioral reconstruction to test all
four at once.**

Why this one over the two data-gated fixes (HCP P6a, powered human P4): it is feasible now
(no gated data), and it is the only experiment that converts the math framework from
apparatus-fitted machinery into a real-network prediction. A four-way agreement makes the
spine a theorem with a measured constant. alpha*/f* landing below 0.64 on a genuinely
log-uniform spectrum gives the program its first real confirmation that sloppiness
concentrates identity (prediction P2a). A divergence localizes exactly which link (ranking,
decoder, or distortion measure) is wrong. It also unblocks Paper B's "Instrument, Not a
Scanner" framing and Paper A's Claim 3 STRONG form in one run, and it retires the standing
alpha*-unification conjecture either way. The two between-session/powered fixes remain the
publication blockers for Paper A, but they wait on data access; this one does not.
