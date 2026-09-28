# learning-rule-fusion

**Extracting and manipulating learning rules by multimodal biosignal fusion: a calibrated environment.**

Anonymous.

## Thesis
Behaviour is the window into how a person learns, and a learner's decision rule can be
inferred from choices. This repo asks what *fusion* of multiple biological signals adds:
whether fusing biosignals lets an actor **extract** a learner's rule more accurately and,
through that accuracy, **manipulate** it more precisely.

Collecting one real dataset with every biosignal, the same task, and an intervention is
infeasible, and static records cannot identify influence (no intervention, no
counterfactual). So we **propose and calibrate the fusion environment** from real data
and study the question where it is identifiable: in a controllable simulation.

## What it shows (results/fusion_environment.json)
Calibrated to the real NARPS loss-aversion distribution (n=108, median 0.88):
- **Extraction.** Inverse-variance fusion estimates the learning rule (loss aversion)
  with RMSE **0.082**, below the best single biosignal (0.088 neural; 0.183 choice; 0.245
  peripheral). Fused estimator SD 0.072 <= best single.
- **Manipulation.** A closed-loop controller dosing the learning rule on the *extracted*
  estimate reaches the target policy with control error **0.045**, below the best single
  biosignal (0.052). The manipulation advantage is inherited from the extraction
  advantage.

The enabling math is the classical inverse-variance fusion inequality
`sigma^2_fused = 1/sum(1/sigma_m^2) <= min sigma_m^2`, with the detection corollary that
SNRs add in quadrature. Both are **proved in Lean 4 + Mathlib** (`proofs/`; see REPRODUCE).

## Honesty scope
This measures a *proposed, calibrated environment*, not real manipulation. Influence is
identifiable only because the learner is controllable; real data enters as calibration
(loss-aversion distribution, value-signal structure), and non-aligned corpora are bridged
with pair-matched controls + synthetic gap-filling, not merged.

## Layout
```
scripts/run_fusion_environment.py   # extraction + manipulation demonstration (main)
scripts/run_learning_rule_risk.py   # NARPS decision-policy individuation/forecasting probe
scripts/run_elicitation_control.py  # exploratory controllable-learner POC (parked; see docs)
src/learning_rule.py                # dynamic-GLM decision-policy / learning-rule extractor
proofs/FusionMath.lean              # inverse-variance inequality (reused)
proofs/FusionDetection.lean         # SNR-quadrature detection corollary (new)
results/                            # JSON results, figures, narps_lambda.npy calibration
docs/learning_rule_influence_PLAN.md
```

## Run
```bash
python3 scripts/run_fusion_environment.py     # reads results/narps_lambda.npy
```

## Related paper
Feeds the MM4SciReal @ ACCV 2026 submission "Extracting and Manipulating Learning Rules
by Multimodal Biosignal Fusion: A Calibrated Environment."

## Citations
Roy et al. 2021 (PsyTrack); Ashwood et al. 2022 (GLM-HMM); Liu, Geadah & Pillow 2025
(learning-rule inference); Krakauer et al. 2017 (behaviour reveals learning); Bagley et
al. 2023 (RL control of an evolving system); Sole et al. 2026 (cognitive-contagion);
Tom et al. 2007 (loss aversion); Finn et al. 2015 (fingerprinting); Genevsky, Yoon &
Knutson 2017 (neuroforecasting); Botvinik-Nezer et al. 2020 (NARPS).
