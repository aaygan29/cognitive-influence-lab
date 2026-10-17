# STATUS — living log

**Read this first every session.** Newest entry on top. A result that is not logged here in
the same commit that produced it did not happen. Every result line carries: seed count, CI,
null, proper score, provenance stamp (or "abstained").

---

## 2026-08-20 — Project stood up (Step 1, checkpoint C1-DONE)

- **Scope set** to five central claims C1–C5 (see README). The "individual neurology as a measured
  quantity" direction is out of scope (not identifiable: no neural ground truth, researcher-coded
  labels) and is reformulated as C5, plus the attribution-scale extension C6. Scoping rationale +
  risk register (internal engineering note, not a paper artifact): `checkpoints/design_rationale_2026-08-20.md`.
- **Honesty layer BUILT** (`src/honesty.py`): proper scores (log/Brier/CRPS), split-conformal
  intervals + sets, MDES power gate, and the `GatedNumber` abstention wrapper. Positive control
  green (`results/e2_honesty_selfcheck.json`): conformal coverage on target, ECE 0.018, gate abstains
  on the underpowered effect and reports the powered one. This is C2's machinery.
- **Literature + math grounding** complete: neuroforecasting lineage (Knutson 2008; Carter 2009;
  Samanez-Larkin & Knutson 2015; Genevsky 2015/2017/2025; Tong 2020; Stallen 2021; Falk 2012/15;
  Smith 2014; Fernandes 2022 AIM-EEG) + forecasting math (Gneiting-Raftery scoring; Angelopoulos-
  Bates conformal; DDM; resource-rational). Citations in `data/DATA_COLLECTION.md` §Literature.
- **Harness audited** (`~/Desktop/Research/behavioral_decoding`): real, CI-green 3.9/3.11, leakage
  guards verified (subject- and stimulus-grouped CV, nested OOF weighting, honest negative R2,
  bootstrap_ci + permutation_test present). Gap confirmed: **no MDES/power module** — that is the
  first thing Step 2 builds.
- **Repo docs** written: README, ROADMAP (6 steps + kill criteria), GUIDELINES (gate ladder +
  honesty layer), paper skeleton, FIGURES plan.
- **Figures**: architecture (Fig 1) and claim->gate->tier map (Fig 2) generated as placeholders
  from `figures/make_figures.py` (schematic; result figures come as experiments land).
- **Git**: initialized; remote to be wired to `ideal-doodle`.

### E1 parameter recovery (Step 2) DONE, honest curve
`src/phenotype.py` built (AIM-DDM choice-marginal estimator). E1 reports recovery as a **curve over
decisions-per-agent** (20 seeds), not a single arbitrary n, because that is the honest object.
Result (`results/e1_recovery.json`, Fig 3): raw-coefficient recovery is high and rises with data
(0.53 -> 0.88 over 200 -> 2000 trials), so the model is identifiable and well-specified. The
affective phenotype RATIOS (loss aversion, threat) are the ratio-of-noisy-estimates case: they clear
the r>=0.6 kill threshold at ~1000-2000 decisions/agent (threat 0.60 at 1000, 0.73 at 2000; loss
aversion 0.69 at 2000) and fail at 200. **Affective kill criterion PASS at n=2000.** Honest caveat
now driving the roadmap: the per-agent data requirement is exactly why hierarchical pooling (C4) and
abstention on sparse records (C5) matter, and it is the reason a single president's decision record
(tens to low hundreds of clean decisions) will often land in the abstention regime, not the
confident one. Identifiability of the decision temperature tau needs reaction times (DDM), added in
Step 3.

### Step 3 DONE (C3, C4 operating characteristics)
- **E4 pooling (C4)** `results/e4_pooling.json`, Fig 6: empirical-Bayes partial pooling lifts low-n
  recovery, CI on improvement excludes 0 for all ratios at 150 decisions/agent (loss aversion
  0.10->0.30, threat 0.15->0.27, risk 0.09->0.44, discount 0.15->0.44). Mechanism for lower MDES.
- **E3 transfer (C3)** `results/e3_transfer.json`, Fig 5: positive/negative control on the pipeline.
  Positive control (market tracks affect): brain arm oosR2 0.845 > behavior 0.731, Delta R2 +0.114
  CI[+0.092,+0.137], permutation p=0.003. Negative control (market = noise): all arms at chance,
  permutation p=0.90, no arm beats another. **pipeline_valid = True.** Honest framing: this validates
  operating characteristics on synthetic data; the empirical claim runs on real NARPS/DEAP.
- Tests: 10 green (6 honesty + 4 pipeline). Estimator, pooling, transfer all covered.

### Step 4 DONE (C5, C6 mechanism)
- **E6 attribution-scale ladder** `results/e6_scale_ladder.json`, Fig 7 (`src/inverse.py`): inverse
  identifiability of the affective posture falls monotonically S1->S6 (r 0.35 -> 0.11) while abstention
  rises (0.16 -> 0.86). Gate keeps the non-abstained subset more reliable (e.g. 0.57 vs 0.35 at S1).
  At the individual scope the framework abstains on 86% of entities: the C5 safety property, and the
  attribution-confound prediction, both borne out. dose_response_holds = True.
- Latent = affective posture with a neural interpretation, never measured firing (GUIDELINES inverse
  clause). Simulation of the mechanism; real president corpus (2 blind coders + kappa) = deployment.
- Tests: 11 green (6 honesty + 5 pipeline incl. scale-ladder dose-response).

### Step 5 DONE (extend + coherence)
- **E7 state-shift extension** `results/e7_state_shift.json` (`src/states.py`): a threat-elevated
  state is recovered as a detected, direction-correct posture shift (+0.73, CI>0, tracks true at
  r=0.35); zero-shift control detects nothing. States = recoverable within-entity manifold
  displacements; roles = between-entity prior shifts (discussed).
- **Paper coherence pass:** Discussion (operating-characteristics synthesis) + Conclusion written;
  abstract/experiments/results reconciled; stale [TBD] markers removed; E1/E3 descriptions aligned to
  what was actually run; E5 marked as real-corpus deployment (pending), E6 as the simulation; figure
  list fixed to match generated assets; Tables 1-2 include C6/E6/E7. 12 tests green.
- Paper now maps cleanly: C1->E1, C2->E2, C3->E3, C4->E4, C5->E5(deploy)+E6, C6->E6, ext->E7.

### Step 6 DONE (reproduce + future work)
- **`reproduce.py`**: one command runs all experiments, regenerates figures + LaTeX tables from fresh
  results, runs the suite, prints a per-claim gate summary. Full end-to-end run: **all 6 gates PASS**
  (~100s). This is the reproducibility capstone.
- Paper §10 Future work written (real neural data + RT/DDM for tau + real president corpus + per-person
  calibration). README + ROADMAP status banners added. Steps 1-5 done on controlled data; step 6 =
  real-data deployment scoped as future work.

### REAL DATA run (NARPS ds001734) — behavioral arm DONE
`src/real_narps.py`, `results/real_narps.json`, Fig 8. Downloaded all 432 events.tsv (108 subjects,
27,454 real accept/reject choices + RTs) from OpenNeuro (CC0).
- **RN1 (C1 real):** loss aversion recovered from real choices; by the study's range manipulation:
  equalIndifference median lambda=1.45 (matches Tom 2007), equalRange median=1.00 (range adaptation).
  Held-out choice prediction mean AUC=0.958 CI[0.951,0.964], Brier 0.061 vs base 0.233.
- **RN2 (C2 real):** conformal coverage 1.00 (>=0.90, conservative), ECE 0.034, gate abstains 1%.
- **RN3 (C4 real):** pooling barely moves lambda SD (0.80->0.78) because every subject has ~256
  trials (high-n regime where synthetic E4 predicted little pooling gain — controlled + real agree).
- **RN4 (DDM signature real):** median corr(|net value|, RT) = -0.30 (n=107); bigger value -> faster
  choice, the evidence-accumulation signature; motivates full-DDM tau.
C1/C2/C4 now validated on REAL human decisions.

**fMRI affective-grounding arm DONE (n=12, honest underpowered result)** `src/real_narps_fmri.py`,
`results/real_narps_fmri.json`, Fig 9. Fixed the NAcc-reads-0.0 bug (pass fmriprep brainmask to both
GLM and sphere masker). 12 subjects, run-01, gain/loss parametric GLM, NAcc/aIns 6mm spheres.
- AIM directions all consistent: NAcc +gain (+0.007), NAcc -loss (-0.007), aIns +loss (+0.007); none
  significant (p~0.2, n=12).
- Neural-vs-behavioral loss aversion r=-0.01 -> **honesty gate ABSTAINS** (MDES at n=12 is 0.73;
  observed |r| far below -> unidentifiable). Establishing grounding needs n>=40 (MDES<0.45).
- This is the framework applying its own abstention discipline to its own neural arm. Scaling the
  subject count is the power fix (roi json `results/narps_fmri_roi.json` is resumable).

### Three-lens evaluation on real data (structure / function / individuation) DONE — C7
`src/real_narps_individuation.py`, `results/real_narps_individuation.json`, Fig 10. Uses the 4 runs
already on disk (no new downloads).
- **STRUCTURE:** across 108 subjects the affective coordinates are strongly coupled (gain-loss
  r=-0.92) and the 6-feature phenotype has effective dimensionality 4.4 (participation ratio) — real
  low-dim manifold geometry, as C1 assumes.
- **FUNCTION:** behavioral prediction AUC 0.96 (real_narps) + neural NAcc+gain/insula+loss directions
  (real_narps_fmri, underpowered).
- **INDIVIDUATION:** cross-run fingerprinting (db=run-01, target=mean runs 02-04, DB-standardized,
  leakage-safe) identifies subjects at 13% vs 0.9% chance (14x, permutation p=0.0005, I_diff=1.10).
  The phenotype is a stable individual trait — the AIM "integrative individuates" half, on real data.
Added as claim C7; abstract + paper 5.3 + README claims table + claims.tex updated. 13 tests green.

### fMRI n=40 scale-up DONE — loss-channel grounding now SIGNIFICANT
`results/real_narps_fmri.json` (n=40), Fig 9. Two AIM effects significant on real data:
**NAcc decreases to loss (t=-2.12, p=0.040), anterior insula increases to loss (t=+2.32, p=0.025)**.
NAcc gain-tracking positive but NS (t=+1.47, p=0.15); insula-gain null (correct). All 3 AIM directions
consistent in sign. Neural-vs-behavioral loss aversion r jumped -0.01 (n12) -> -0.40 (n40), but MDES
at n=40 is 0.44, so |r|=0.40 sits just under the bar -> **gate still ABSTAINS** (boundary behavior,
+ ratio sign-instability). Loss-channel neural grounding = real significant result; individual
neural->behavioral link needs larger n or multi-run per-subject estimates. Honest win.

### Cross-dataset generalization DONE (real C4 / external validity)
`src/real_crossdataset.py`, `results/real_crossdataset.json`, Fig 11. Pooled Tom 2007 ds000005 (16
subj, PDDL, events only — no fmriprep derivatives so behavioral only).
- Tom-2007 median lambda=1.94 (replicates the original loss-aversion value); NARPS 1.12 (equalRange
  dilution); distributions differ by design (KS p=0.004), honestly.
- TRANSFER: NARPS-trained valuation predicts Tom-2007 choices out-of-dataset AUC=0.86; Tom->NARPS 0.89.
  The phenotype generalizes across labs -- strongest real form of C4 / Gate-7 external validity.

### TD/predictive-coding extension (paper §9b)
Folded the Nature 2026 hippocampal predictive-coding-of-reward paper (Brandon/Williams/Pehlevan) in as
a modeling extension: affective value = a LEARNED TD predictive signal, delta_time <-> TD gamma; gives
individuation a temporal axis. Honest framing: proposed extension, not tested here.

### Next dataset (to power underpowered neural gain-channel)
Search term given to user: "monetary incentive delay fMRI" (with fmriprep/MNI derivatives +
individual-subject). MID isolates NAcc gain-anticipation = the affective coordinate still NS (p=0.15).

### fNIRS link L8 DONE (literature) + bibliography + source verification
- **L8 fNIRS**: reviewed the 16 openfnirs.org datasets — NONE is a reward/value/decision task (all
  motor/sensory/auditory/resting/imagery), and no open reward-fNIRS dataset is downloadable to
  re-analyze. So L8 is a **literature-supported** link (new status "literature", distinct + weaker
  than measured): fNIRS prefrontal reward/value tracking, cited Balconi et al. 2018 (10.3233/JPD-171290)
  + Wang/Xu/Ball 2026 (10.1016/j.neuroimage.2026.121942), both fNIRS+IGT verified via PubMed. Chain
  now has **0 pending** — complete. Verdict still TRIANGULATED (5 measured, 1 external, 1 literature,
  2 abstained). Fig 13 updated (measured/literature/abstained legend).
- **Bibliography** `paper/references.bib` (BibTeX): all refs; datasets from their dataset_description.json,
  newest citations verified via PubMed. Key: Cavanagh 2015 (the ds003458 EEG paper, PubMed-verified)
  is itself about the feedback-locked Reward Positivity we reproduce -> perfect provenance for L7.
- **Source verification**: NARPS ds001734 = Botvinik-Nezer 2019 (10.1038/s41597-019-0113-7),
  Tom 2007 ds000005 = Science 315:515, ds003458 = Cavanagh 2015 NeuroImage — all confirmed from
  dataset metadata / PubMed. Paper §References + provenance note added.

### EEG link L7 DONE (real, strong) + full coherence pass
- **EEG Reward Positivity** `src/real_eeg.py`, `results/real_eeg.json`, Fig 14: OpenNeuro ds003458
  (23 subj, open; DEAP is licence-gated + not usable non-interactively). Win-feedback frontocentral
  ERP more positive than loss by +3.0uV, paired t=6.58, **p<1e-6, d=1.37**. Establishes L7 -> the
  grounding chain now spans TWO neural modalities (fMRI + EEG). Chain verdict still TRIANGULATED
  (5 established, 2 abstained, 1 pending L8 fNIRS).
- **Coherence pass:** abstract + §3.5 + §5.6/5.6b + figure list updated for EEG; stale DEAP refs
  fixed to ds003458 (open); README claims table de-jargoned (removed T3/T4/T5 tiers -> plain
  real/simulation); README status section condensed; fig13 simplified; byline comma fixed.

### Grounding by triangulation + statistical validation DONE
- **External NAcc gain confirmation** `src/neurovault_grounding.py`, `results/neurovault_grounding.json`,
  Fig 12: sampled NAcc/insula in independent NeuroVault reward group maps. NAcc gain-anticipation
  POSITIVE across 3 maps (EV-MID 104 subj +1.0, gain>no-gain 46 subj +1.6, social-reward +0.7) ->
  externally confirms the gain channel that was NS (p=0.15) in our n=40 NARPS GLM. Group-level.
- **Grounding chain** `src/grounding_chain.py`, `results/grounding_chain.json`, Fig 13: assembles all
  links from the result JSONs, marks each established/abstained/pending. VERDICT = TRIANGULATED
  (neural->construct established via L1 external + L2 significant; construct->behavior via L4 held-out
  AUC + L5 cross-dataset; direct L6 abstained; EEG/fNIRS L7/L8 pending). Convergent evidence, not causal.
- **Statistical validation** `src/triangulation_stats.py`, `results/triangulation_stats.json`:
  (1) direct joint-sample correlations (NARPS n=39): stable neural loss measures ~0 with behavioral
  lambda (abstained); neural-lambda ratio r=-0.40 p=0.011 but wrong sign + gate abstains (ratio
  artifact). (2) transitivity bound (PSD): direct r confined to [-0.51,+0.29], observed -0.40 inside
  (but wide, bc within-sample neural-construct link weak). (3) Sobel mediation NS (z=-0.73). (4) Fisher
  combine of established links p=2.4e-7 -> convergent grounding strong; direct link honestly unestablished.
- Math tools (transitivity_bound / sobel / fisher_combine) unit-tested. 14 tests green.
- Link triage of user-supplied datasets: NeuroVault = group maps (confirmatory only); dataverse
  9BAJTD = timing files only; figshare = a table. NONE give per-subject neural+behavior together.
  Search term for the one that would: "monetary incentive delay fMRI" w/ derivatives + individual data.

### Remaining deployment work (real data)
1. Wire the estimator to the harness NARPS (fMRI) + DEAP (EEG) loaders and run E3 on real neural data
   (the empirical neuroforecasting test, vs the synthetic operating-characteristic control done here).
2. Curate the US-president decision corpus (2 blind coders + Cohen's kappa) and run E6 on it.
3. Add the RT/DDM path to identify the decision temperature tau (E1 currently identifies ratios only).
Next in-repo step: a Discussion/operating-characteristics synthesis pass + coherence check so the
paper reads end to end for a reviewer.

### Open risks being tracked
- Gate 3 specificity: affective/integrative split must beat content-blind 2-factor PCA.
- Gate 7 external validity: population->individual transfer is asserted by the field, tested by us.
- Gate 8 measurement: leader decision coding needs 2 blind coders + kappa before any C5 number.
