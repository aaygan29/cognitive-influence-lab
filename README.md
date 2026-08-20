# Neural-Grounded Computational Decision Phenotypes

**A honest-by-construction estimator that predicts decisions from a neurally-grounded
parameter profile, and provably abstains when it cannot.**

Anonymous,(harness lineage: `behavioral_decoding`).
Status document. Living. Updated every time a result lands. Last update: 2026-08-20.

---

## The idea in one paragraph

A decision-maker is not a black box; they are a low-dimensional point on a decision
manifold whose coordinates are neuroeconomic parameters (reward sensitivity, loss
aversion, risk curvature, temporal discount, threat sensitivity, decision temperature).
Knutson's Affect-Integration-Motivation (AIM) framework says a choice is built from an
**affective** component that generalizes across people and an **integrative** component
that individuates. We estimate that manifold with a drift-diffusion likelihood under
AIM-structured valuation, calibrate the affective component on populations we can
actually scan (fMRI: NARPS; EEG: DEAP), fit the integrative component from a target's own
documented decisions, and wrap every prediction in distribution-free conformal coverage
plus a strictly-proper score. The system's defining property is that it **knows when it
cannot predict**: below the identifiability floor it abstains rather than guesses.

## What this is NOT

It is **not** a measured neuroprofile of any individual. No living person named here has
been scanned. The neural grounding lives in the *population* manifold and the model's
functional form, never in a fabricated scan of a target. Claims about individuals are
claims about their **revealed decision policy**, bounded by explicit confounds, never
about their private neurology or psychology. See `GUIDELINES.md` §Ethics.

---

## The five central claims (the paper's contributions)

| # | Claim | Evidence tier target | Where it is tested |
|---|---|---|---|
| **C1** | AIM-structured DDM phenotype is *identifiable*: recovers known parameters | T4/T5 | synthetic ground truth (parameter recovery) |
| **C2** | Conformal + proper-scoring + MDES abstention = distribution-free coverage, calibration, provable withholding | T4/T5 | math + simulation |
| **C3** | Affective component forecasts held-out / non-representative choice above and beyond behavior, inside one estimator | T3→T4 | NARPS + DEAP (real neural data) |
| **C4** | Hierarchical cross-dataset pooling lowers MDES and per-individual posterior width | T4 | NARPS + DEAP joint fit |
| **C5** | On genuinely confounded targets the framework predicts where identifiable and *provably abstains* where not | T4 | public-figure decision corpora (preregistered) |
| **C6** *(ext.)* | Inverse inference (decision -> latent affective posture) is *scale-dependent*: identifiable at aggregate attribution scope, abstaining at the individual scope | T4 | US-president decision corpus across scopes S1..S6 |

These five are the project's central claims (scoping rationale in
`checkpoints/design_rationale_2026-08-20.md`, an internal engineering note, not a paper artifact).
The direction "predict an individual's actions from their neurology" is out of scope as a measured
claim: there is no neural ground truth for any individual and the target decisions are
researcher-coded, so it is not identifiable. It is reformulated as C5, and as the attribution-scale
question C6 (extension).

---

## Six steps forward (the arc; full detail in `ROADMAP.md`)

1. **Ground** — literature + AIM math + audit the `behavioral_decoding` harness + synthetic positive control. *(mostly done)*
2. **Advance the application** — build the unified AIM-DDM estimator + honesty layer onto the harness; prove C1 (recovery) and C2 (coverage) on synthetic ground truth.
3. **Replicate in new data / new formats** — port to real fMRI (NARPS) and EEG (DEAP); replicate affective>behavioral transfer (C3) and cross-dataset pooling (C4).
4. **A genuinely new method** — calibrated abstention on confounded real-world targets (C5): the public-figure decision corpus, preregistered temporal split, the abstention gate evaluated as a *safety property*, not a storytelling device.
5. **Conclude and extend** — characterize operating characteristics; extend to state (sad/angry) and role (trader/cop/doctor) phenotypes as manifold shifts.
6. **Future directions** — real all-modality + market data; active enrollment; links to `cortex-of-anyone` and `affectprint`.

## Documents

- `ROADMAP.md` — the six-step plan, each step with goal / method / dataset / deliverable / the gate it must pass / **kill criterion** / checkpoint.
- `GUIDELINES.md` — the rules every result must obey (gate ladder, evidence tiers, seed mandate, leakage guards, honesty layer, preregistration, ethics, checkpoint discipline).
- `paper/neurips_decision_phenotype.md` — the paper in NeurIPS form, built only on C1–C5.
- `STATUS.md` — the living log. Dated entries. Updated on every result. **This is the file to read first each session.**
- `data/DATA_COLLECTION.md` — datasets, provenance, licenses, access status, coding protocol.
- `figures/` — figure plan (`FIGURES.md`) + generated figures + the script that makes them.
- `checkpoints/` — dated snapshots of decisions, reviews, and reproducible result bundles.

## Reproduce everything

```bash
python reproduce.py
```

Runs all experiments (honesty positive control, E1 recovery, E4 pooling, E3 transfer controls, E6
attribution-scale ladder, E7 state shift), regenerates the vector figures and LaTeX tables from the
fresh `results/*.json`, runs the test suite, and prints a per-claim gate summary. Full run is a few
minutes (E3 does permutation nulls). All six headline gates currently pass.

## Status (2026-08-20)

Steps 1-5 complete on controlled data; step 6 (real-data deployment) is scoped as future work.

| Claim | Experiment | Result | Status |
|---|---|---|---|
| C1 | E1 recovery curve | affective ratios reach r>=0.6 by ~1-2k decisions/agent | PASS (controlled) |
| C2 | E2 honesty self-check | coverage on target, ECE 0.018, gate abstains when underpowered | PASS |
| C3 | E3 transfer controls | brain>behavior on +control (ΔR2=+0.11, p=0.003), null on -control | PASS (controlled) |
| C4 | E4 pooling | pooled > unpooled recovery at low n, CI excludes 0 | PASS |
| C5/C6 | E6 scale ladder | identifiability falls (0.35->0.11), abstention rises (0.16->0.86) S1->S6 | PASS |
| ext. | E7 state shift | threat-elevated state recovered (+0.73, CI>0), null control | PASS |

**Real data (NARPS ds001734, 108 subjects, 27,454 real choices):** C1/C2/C4 validated on real human
decisions. Loss aversion λ=1.45 (equalIndifference) / 1.00 (equalRange, range-adapted); held-out
choice prediction AUC=0.96; conformal coverage holds; RT shows the DDM evidence-accumulation signature
(r=−0.30). See `src/real_narps.py`, `results/real_narps.json`, Fig 8. The real fMRI affective-grounding
arm (`src/real_narps_fmri.py`, NAcc/insula betas) streams the fmriprep volumes and is being brought
online. A coded decision corpus (C5/C6 real) remains future work (paper Section 10).

## Checkpoint protocol (so nothing is lost)

- **Git commit at every checkpoint.** No Claude co-author trailer (per user convention). Never push to `main` without explicit ask.
- **STATUS.md updated in the same commit as any result.** A result that is not logged did not happen.
- **Every figure is regenerated by `figures/make_figures.py` from data in `results/`**, never hand-edited, so a figure always traces to the numbers behind it.
- **Every reported number is born gated**: multi-seed, with a CI, a permutation null, and a proper score. See `GUIDELINES.md`.
