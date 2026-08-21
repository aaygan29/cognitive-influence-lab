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
3. **Replicate in new data / new formats** — port to real fMRI (NARPS) and EEG (DEAP); replicate affective>behavioral transfer (C3) and cross-dataset pooling (C4).
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
(r=−0.30). See `src/real_narps.py`, `results/real_narps.json`, Fig 8. **Real fMRI grounding (n=40, `src/real_narps_fmri.py`, Fig 9):** the loss/threat channel is grounded
on real neural data — NAcc decreases to loss (p=0.040) and anterior insula increases to loss (p=0.025),
both AIM-consistent. The neural↔behavioral loss-aversion correlation strengthened to r=−0.40 but sits
just under MDES (0.44) at n=40, so the honesty gate abstains at the boundary. A coded decision corpus
(C5/C6 real) remains future work (paper Section 10).

**Cross-dataset generalization (`src/real_crossdataset.py`, Fig 11):** pooled a second real dataset,
Tom et al. 2007 (ds000005, 16 subjects). Tom-2007 median λ=1.94 (replicates the original); a valuation
model trained on NARPS predicts Tom-2007 choices out-of-dataset at AUC=0.86, and Tom→NARPS at 0.89 —
the phenotype transfers across labs (C4 / external validity on real data). A TD/predictive-coding
front-end (Brandon/Williams/Pehlevan 2026, hippocampal reward prediction) is proposed as a modeling
extension (paper §9b). Next dataset to power the neural gain-channel: a large **MID
reward-anticipation** fMRI set with derivatives.

## Checkpoint protocol (so nothing is lost)

- **STATUS.md updated in the same commit as any result.** A result that is not logged did not happen.
- **Every figure is regenerated by `figures/make_figures.py` from data in `results/`**, never hand-edited, so a figure always traces to the numbers behind it.
- **Every reported number is born gated**: multi-seed, with a CI, a permutation null, and a proper score. See `GUIDELINES.md`.
