# Neural-Grounded Computational Decision Phenotypes

**A honest-by-construction estimator that predicts decisions from a neurally-grounded
parameter profile, and provably abstains when it cannot.**

Anonymous (harness lineage: `behavioral_decoding`).
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
actually scan (fMRI: NARPS; EEG: open dataset ds003458), fit the integrative component from a target's own
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

## The central claims (C1–C7)

Each claim states one plain idea and lists the strongest evidence we have for it so far. "Real"
means measured on real human data; "simulation" means shown on controlled data with known ground
truth (a positive/negative control on the method itself).

| # | In one sentence | Strongest evidence so far |
|---|---|---|
| **C1** | The model recovers a person's decision parameters (e.g. their loss aversion) from their choices. | **Real**: loss aversion recovered on NARPS (108 subjects), matching the classic value; plus simulation recovery. |
| **C2** | Every prediction is calibrated with a coverage guarantee, or the model says "I don't know" (abstains). | **Real + proof**: conformal coverage holds on NARPS; the abstention gate fires when a sample is underpowered. |
| **C3** | The affective (reward-anticipation) signal forecasts choice beyond behavior alone. | **Simulation** (positive/negative control); the loss channel is grounded on real fMRI; the individual link is still underpowered. |
| **C4** | Pooling across people and datasets buys statistical power. | **Real**: the phenotype transfers across labs (NARPS↔Tom 2007, out-of-dataset AUC 0.86/0.89). |
| **C5** | On confounded targets it predicts where it can and abstains where it can't. | **Simulation** (the attribution-scale ladder). |
| **C6** *(ext.)* | Running the model in reverse works at the group scale and abstains at the individual scale. | **Simulation** (identifiability falls, abstention rises, group→individual). |
| **C7** *(real)* | The phenotype has geometric **structure**, predicts choice (**function**), and identifies the **individual**. | **Real** NARPS: gain–loss coupling r=−0.92; choice AUC 0.96; cross-run fingerprint 14× chance (p=5e-4). |

The direction "predict an individual's actions from their neurology" is deliberately out of scope as
a *measured* claim: no individual is ever scanned, so it is not identifiable. It is reformulated as C5
(predict-or-abstain) and the attribution-scale question C6.

---

## Six steps forward (the arc; full detail in `ROADMAP.md`)

1. **Ground** — literature + AIM math + audit the `behavioral_decoding` harness + synthetic positive control. *(mostly done)*
2. **Advance the application** — build the unified AIM-DDM estimator + honesty layer onto the harness; prove C1 (recovery) and C2 (coverage) on synthetic ground truth.
3. **Replicate in new data / new formats** — port to real fMRI (NARPS) and EEG (open ds003458; DEAP is licence-gated); replicate affective>behavioral transfer (C3) and cross-dataset pooling (C4).
4. **A genuinely new method** — calibrated abstention on confounded real-world targets (C5): the public-figure decision corpus, preregistered temporal split, the abstention gate evaluated as a *safety property*, not a storytelling device.
5. **Conclude and extend** — characterize operating characteristics; extend to state (sad/angry) and role (trader/cop/doctor) phenotypes as manifold shifts.
6. **Future directions** — real all-modality + market data; active enrollment; links to `cortex-of-anyone` and `affectprint`.

## Documents

- `ROADMAP.md` — the six-step plan, each step with goal / method / dataset / deliverable / the gate it must pass / **kill criterion** / checkpoint.
- `GUIDELINES.md` — the rules every result must obey (gate ladder, evidence standards, seed mandate, leakage guards, honesty layer, preregistration, ethics, checkpoint discipline).
- `paper/neurips_decision_phenotype.md` — the paper in NeurIPS form, built on the claims C1–C7.
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

The method (C1–C7) is established on controlled data, and validated on real public data where the data
exist. What is on real data vs simulation is in the claims table above. In short:

**Real-data results (all from committed scripts + result files):**
- **Behavior** — NARPS (108 subjects, 27,454 choices): loss aversion recovered (λ=1.45 vs 1.00 by the
  range manipulation, matching the literature), held-out choice AUC=0.96, conformal coverage holds.
  A second dataset (Tom 2007) replicates λ=1.94, and a model trained on one lab predicts the other's
  choices out-of-dataset (AUC 0.86/0.89) — the phenotype transfers across labs.
- **Neural grounding, two modalities** — fMRI (NARPS, n=40): NAcc decreases to loss (p=0.040), insula
  increases to loss (p=0.025); NAcc gain-anticipation confirmed in independent NeuroVault maps. EEG
  (ds003458, n=23): the Reward Positivity is large and significant (p<10⁻⁶).
- **Grounded by triangulation** — the affective channel is grounded through a chain (neural→construct→
  behavior) whose links live in whichever dataset has each pair. Verdict: **triangulated**. The one
  direct within-subject neural↔behavior link is underpowered, so the model **abstains** on it — and
  the transitivity-bound math shows the chain does not require it.

**Honestly not yet done:** the direct individual neural↔behavior correlation (needs a dataset with
per-subject fMRI *and* choices together), a coded real decision corpus for C5/C6, and reaction-time
identification of the decision temperature τ. See paper Section 10.

## Checkpoint protocol (so nothing is lost)

- **STATUS.md updated in the same commit as any result.** A result that is not logged did not happen.
- **Every figure is regenerated by `figures/make_figures.py` from data in `results/`**, never hand-edited, so a figure always traces to the numbers behind it.
- **Every reported number is born gated**: multi-seed, with a CI, a permutation null, and a proper score. See `GUIDELINES.md`.
