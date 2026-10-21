# Provenance: where every number and source comes from

Nothing in this repository is estimated from human data. Every result is produced by a simulation of the model in `theory/model.md`. This file lets you check each number against the script that produced it.

**How to reproduce.** Both experiments rerun from their seeds and reproduce the committed JSON exactly (checked when this file was written).

```
python3 experiments/e1_e2_dissociation.py   # seed 7,  about 10 s
python3 experiments/e4_dose_budget.py       # seed 11, about 40 s
```
Requires Python 3, `numpy`, `matplotlib`. Outputs go to `experiments/*.json` and `figures/`.

**Labels used below**
- **chosen**: an input parameter picked by the author, not derived from data.
- **derived**: a closed-form consequence of the model or of another number here.
- **fitted**: a constant fitted inside the simulation.
- **measured (sim)**: an output of the simulation.
- **read from figure**: quoted from a plot in the results write-up, not stored in the JSON.

## The model (shared by E1, E2, E4)

`U(x; a) = x^4/4 - x^2/2 - a x`, dynamics `dx = -U'(x;a) dt + sqrt(2D) dW`. This is a textbook bistable potential. It is **chosen for tractability**; nothing about it is fitted to belief data.

| Quantity | Value | Label | Source |
|---|---|---|---|
| Fold (critical attacker strength) `a*` | 0.3849 = `2/(3*sqrt(3))` | derived | `a_star` in both JSON files |

## E1 + E2: `experiments/e1_e2_dissociation.py` → `e1_e2_results.json`

| Quantity | Value | Label | Notes |
|---|---|---|---|
| Noise `D` | 0.025 | chosen | Called "matched to behavioural data" in the plan, but not calibrated to any dataset. |
| Step `dt`, horizon `T` | 0.01, 80 | chosen | |
| Sub-critical drive `a_sub` | 0.24 | chosen | Barrier intact. Used for conditions A and S. |
| Detector window, debounce | 400, 30 | chosen | Trailing residual variance about the moving well bottom. |
| Trials per condition, seed | 200, 7 | chosen | |
| Warning threshold | 0.0926 | derived | 95th percentile of the variance peak on 54 non-crossing surrogate runs. |
| B (forced crossing): warned before crossing | 75 of 200 = 0.375 | measured (sim) | `B_detection_rate` |
| B median lead time | 5.8 time units (mean 7.16) | measured (sim) | `B_lead_time_median` |
| A (noise-activated): warned before crossing | 7 of 200 = 0.035 | measured (sim) | At the false-alarm floor by construction: a 95th-percentile threshold trips on about 5% of surrogates. |
| A lead time | median 4.06, mean 17.4 | measured (sim) | Not meaningful; these are false alarms. |
| "10x separation" | 0.375 / 0.035 = 10.7 | derived | |
| "Variance rises about 10 time units before the crossing" | about 10 | read from figure | Panel (c) of `figures/e1_e2_dissociation.png`. Not in the JSON. |

## E4: `experiments/e4_dose_budget.py` → `e4_results.json`

| Quantity | Value | Label | Notes |
|---|---|---|---|
| Noise `D` | 0.03 | chosen | Differs from E2 (0.025). Fitted `k` below applies to this `D` and this well shape only. |
| Trials per point, seed | 400, 11 | chosen | |
| Ramp time, short dwell, long dwell | 20, 2, 120 | chosen | |
| Peak-strength sweep `a_peak` | 0.10 to 0.375, 16 points | chosen | |
| Minimum barrier along the sweep | 0.158 down to 0.001 | measured (sim) | The write-up says "0.16 to 0". |
| Cumulative dose along the sweep | 4.5e-5 to 1.1e-3 | measured (sim) | The write-up says "about 0.001". |
| Hold-attacker crossing probability | 0.13 up to 1.00 | measured (sim) | `pc_hold` |
| Kramers constant `k` | 0.190 | fitted | `kramers_k`; fit error 7.3e-4 (`kramers_mse`). |
| Target risk `eps`, session length `T` | 0.05, 120 | chosen | |
| Barrier floor `ΔU_floor` | 0.1829 | derived | `D * ln(k*T / -ln(1-eps))`. Recomputed from the fitted `k`; matches the JSON. |
| Crossing probability with the floor enforced | ramp 0.06, front-load 0.105, 1.5x dwell 0.115 | measured (sim) | 0.105 is the worst of the two nominal strategies; 0.115 is the stress case with a longer dwell. The write-up says both "about 0.10" and "worst 0.115". |
| Exploit attacker (parks below the fold) | `a = 0.370`, crossing probability 1.00 | measured (sim) | Dose spent is 0.001. |

## Outside sources cited in `theory/`

These are cited in the theory notes. They were **not re-verified** when this file was written. Two
of the five have since been checked against Europe PMC (2026-10-20) and now live in
[`corpus/references.json`](../../../corpus/references.json); the other three are still pointers to
check, not established support.

| Used for | Source as cited in this repo | State |
|---|---|---|
| Norepinephrine as a barrier and noise knob | Su et al., Allen Institute, bioRxiv 717727 | **Verified, and superseded.** Now published: Su Z et al., "Topographic structure and function of locus coeruleus noradrenaline neurons", *Nature* 2026, doi 10.1038/s41586-026-11026-0, PMID 42749800. The abstract matches this repo's description closely (dorsal LC neurons projecting to isocortex activate on choice switches and carry reward-prediction-error signals; ventral LC activity is higher when animals ignore reward-predicting stimuli). **It is a mouse result**, so it licenses LC-NE as a candidate mechanism for the gain parameter, not a human magnitude. Cite `su2026-lc-topography`, not the preprint. |
| Active inference and the dose-to-barrier bridge | Novelli, Stoliker, Razi et al., PsiConnect, Scientific Data 2026 | **Verified.** Novelli L, Stoliker D, Barta T, Greaves MD, Chopra S, Jackson J, Kwee J, Pang JC, Williams ML, Razi A, "PsiConnect: Multimodal Neuroimaging of Context-Dependent Brain and Behaviour Dynamics under Psilocybin", *Scientific Data* 2026, doi 10.1038/s41597-026-07312-1, PMID 42168232. N=62, single 19 mg dose, multi-echo fMRI plus EEG, rest and three naturalistic conditions, with half the cohort meditation-trained so **set and setting are crossed with drug**. Cite `novelli2026-psiconnect`. A follow-up preprint on the same cohort bears directly on the precision bridge; see [`evidence/psilocybin-precision`](../../../evidence/psilocybin-precision). |
| Collective-scale model of agency erosion | Moon and Boudreaux, RAND, 2026 | Still unverified. |
| Neural readout of a decision crossing | Li et al., iScience 2026; bioRxiv 730072 | Still unverified. |
| Example of a transfer claim dissolved by a proper null | Salzberg et al., Science 2001 | **Verified.** Salzberg SL, White O, Peterson J, Eisen JA, "Microbial genes in the human genome: lateral transfer or gene loss?", *Science* 2001, doi 10.1126/science.1061036, PMID 11358996. Cite `salzberg2001-lateral-transfer`. |

**One caveat this node should absorb.** `theory/related.md` proposes an NE/arousal proxy (pupil,
tonic latency drift) as the specificity control that separates a genuine approach to the separatrix
from an arousal shift. That control is weaker than it looks: the paper usually cited to license
pupil-as-LC concludes the relationship is **not specific to LC** (`joshi2016-pupil-lc`), pupil
predicts only a small fraction of LC spiking moment to moment (`megemont2022-pupil-not-lc`), and in
humans the pharmacological and pupillometric handles dissociate (`nuiten2026-arousal-dissociation`).
A pupil covariate controls for something, but not specifically for LC gain. See
[`evidence/lc-ne`](../../../evidence/lc-ne) for the full record.

The dynamical-systems background (Kramers escape, critical slowing down before a fold) is standard, but the theory notes do not cite specific papers for it. Adding those citations is an open task in [COLLABORATING.md](COLLABORATING.md).
