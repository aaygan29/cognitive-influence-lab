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
