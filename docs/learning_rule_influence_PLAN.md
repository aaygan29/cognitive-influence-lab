# Learning-rule influence: build plan (DEFERRED, proper build)

Status 2026-09-28. Not for ACCV/Oct 1. Aim at a better-fit venue after a valid
build. Scaffolding is local only (not pushed). This doc is the spec so the proper
build does not repeat the failure modes already found.

## Thesis
A measured, *interventional* framework for influence in value-based decision-making:
represent a learner by its **learning rule** (Pillow/Liu), define target
decision-related **brain states** from neuroeconomic theory (Knutson/Genevsky),
and test whether altering the learning rule can synthetically **elicit** a target
state (e.g., a risk-seeking / "positive gambling" approach state), quantifying the
control budget and identifiability. Influence is identifiable here because the
learner is controllable, unlike static choice data.

## Verified foundations (all real, links checked 2026-09-28)
- Roy, Bak, Akrami, Brody, Pillow (2021) *Neuron* — PsyTrack, dynamic GLM weights.
  https://www.cell.com/neuron/fulltext/S0896-6273(20)30963-6
- Liu (Yuhan Helena), Geadah, Pillow (2025) *NeurIPS* — flexible learning-rule
  inference: GLM policy + DNN learning rule. https://arxiv.org/abs/2509.04661
- Ashwood, ..., Pillow (2022) *Nat. Neurosci.* 25:201-212 — GLM-HMM latent states.
  https://www.nature.com/articles/s41593-021-01007-z
- Botvinik-Nezer et al. (2020) *Nature* 582:84-88 — NARPS ds001734 mixed gambles.
  https://www.nature.com/articles/s41597-019-0113-7
- ADD + verify before use: Knutson & Genevsky neuroforecasting (NAcc reward
  anticipation -> approach; anterior insula -> avoidance); Genevsky, Yoon,
  Knutson (2017) on aggregate/market forecasting; Tom, Fox, Trepel, Poldrack
  (2007) *Science* loss aversion. Do NOT cite until link-verified.

## Failure modes already found (do not repeat)
1. **Static choice data cannot identify influence.** No intervention, no
   counterfactual, and gamble accept/reject has no ground truth. Any "manipulation"
   number on NARPS choices is a what-if on a fitted model, not evidence. Influence
   must be studied in a controllable learner.
2. **Preference-fit is not accuracy.** AUC on choices measures fitting a
   preference; do not call it predictive accuracy.
3. **Scale-invariance degeneracy.** With a linear value code V = g^rho - lambda*l^rho,
   loss-aversion only *scales* the loss pattern, and RSA/correlation is
   scale-invariant, so free (no-intervention) learners already match the target
   (success rate 1.0 with and without control). The internal state must change
   **shape**, not scale, with the learning rule, and the elicitation metric must
   not be scale-invariant.
4. **Ceiling from easy trials.** On NARPS, gain/loss magnitudes nearly determine
   choice (pop AUC 0.96); individual signal lives only near the indifference point.
   Any evaluation must isolate decision-relevant (ambiguous) regimes.

## What a valid build requires
1. **Reproduce Liu-Geadah-Pillow.** GLM decision policy + DNN learning rule; train
   on de-novo learning data; verify recovered rules match published reward-history
   dependence. The learner's **hidden state** (RNN/DNN), not a 2-term formula, is
   the internal geometry -- non-degenerate by construction.
2. **Target states from theory, not the test set.** Use Knutson/Genevsky to name
   the target regions/representations (NAcc approach for "positive gambling";
   insula avoidance). Use real distributions (NARPS loss-aversion range: median
   0.88, 10-90 pctile 0.57-1.35) only as the plausible target prior.
3. **Valid elicitation metric.** A non-scale-invariant distance in the learner's
   hidden-state geometry (e.g., Procrustes/CKA on hidden reps, or decodable
   approach-bias), plus a **null control**: a target the learning rule should NOT
   be able to elicit (falsifiability), and a no-intervention baseline that must
   fail.
4. **Control-theoretic analysis.** Bounded per-update dose; measure control budget
   to reach target, dose-response, and bounded-influence signatures (hysteresis /
   bifurcation on release). Report identifiability limits.

## Datasets in hand
- NARPS ds001734 (108 subj, mixed gambles, behavior) -> target loss-aversion prior.
- ds002989 delay-discounting (40 subj) -> second value-decision prior.
- BOLD5000 (reproduced) -> individuation calibration reference from digital-brain.

## Local scaffolding (not pushed)
- `src/behavioral_decoding/features/learning_rule.py` -- dynamic-GLM learning-rule
  vector (PsyTrack-style approximation).
- `scripts/run_learning_rule_risk.py` -- NARPS individuation/forecasting/what-if
  (kept as a documented negative-result / validity probe, not a result).
- `scripts/run_elicitation_control.py` -- interventional POC; currently degenerate
  (scale-invariance), retained to encode failure mode 3.
