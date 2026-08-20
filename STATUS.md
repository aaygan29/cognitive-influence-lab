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

### Next action (Step 3)
Wire the estimator to the real NARPS/DEAP loaders in the `behavioral_decoding` harness; add the
hierarchical-pooling variant and show it lifts low-n recovery (previews C4); run E3 affective-vs-
behavioral transfer with the harness permutation null. Every number through the honesty layer.

### Open risks being tracked
- Gate 3 specificity: affective/integrative split must beat content-blind 2-factor PCA.
- Gate 7 external validity: population->individual transfer is asserted by the field, tested by us.
- Gate 8 measurement: leader decision coding needs 2 blind coders + kappa before any C5 number.
