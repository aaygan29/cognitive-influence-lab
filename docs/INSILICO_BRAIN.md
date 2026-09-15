# The in-silico belief model (the "fake brain")

Aayush Gandhi. A model that scores content for how likely an audience is to believe it, generalizes
across domains, and can be optimized by altering content factors. No em dashes. Numbers from
results/insilico_brain.json; reproduce with `python -m src.insilico_brain`.

## What it is

A brain-structured behavioral encoder:

    content factors  ->  brain-state layer (approach - avoidance)  ->  belief score P(believe)

Six alterable content factors (evidence, framing, affect, social proof, identity threat, concession) map
to a brain-state layer structured as the validated neural composite: approach (evidence, framing, affect,
social proof, concession) minus avoidance (identity threat), mirroring the NAcc minus anterior-insula
signal that generalized out-of-sample on the real Knutson fMRI. A logistic read-out is fit to real belief
outcomes (ChangeMyView deltas, Persuasion-for-Good donations). Grounding for the approach comes from the
encoding-model literature: foundation models predicting neural responses to new stimuli (Wang et al.
2025), cognitive encoding models predicting activation from stimulus features (Walters et al. 2022), and
Kriegeskorte et al. 2018 on stating the generalization level explicitly.

Honest scope: this is not a scanned brain. The intermediate representation is constrained to the validated
approach-avoidance structure, and the output is trained on real behavior. A learned fMRI encoder (TRIBE,
or the author's Mary/Qualia) would replace the text-to-factor step and is the stated next step.

## Results

1. Cross-domain generalization (train one corpus, score the other):
   - CMV to P4G: brain-state AUC 0.546, full-factor AUC 0.537.
   - P4G to CMV: brain-state AUC 0.536, full-factor AUC 0.519.
   The brain-state read-out generalizes slightly better than the unconstrained model in both directions.
   Modest (AUC ~ 0.54) but above chance and cross-domain, and the brain constraint helps.

2. Multi-factor variance (bootstrap each factor's belief effect, cross-corpus robustness):
   - evidence: +0.060 (CMV), +0.196 (P4G), both significant, matches the approach prior. ROBUST.
   - identity threat: -0.046 (CMV), -0.184 (P4G), both significant, matches the avoidance prior. ROBUST.
   - framing, affect, social proof, concession: not robust across both domains (inconsistent sign or ns).
   From various angles, only two levers robustly move belief: more evidence, less identity threat. These
   are exactly the approach and avoidance poles.

3. Optimization (counterfactually alter the robust factors on low-belief content):
   - CMV: mean belief 0.600 -> 0.625 (lift +0.025).
   - P4G: mean belief 0.450 -> 0.541 (lift +0.091).
   Moving the two robust, brain-consistent factors raises the predicted belief score.

## What this says

- The in-silico model does what was asked: it scores content for belief-likelihood, generalizes to a new
  domain, and can be optimized by altering content, with the alterations grounded in a brain-validated
  structure rather than arbitrary.
- The robust belief drivers are parsimonious and interpretable: evidence up, identity threat down. This
  agrees with the ChangeMyView within-pair result (engagement/evidence wins) and with the approach-
  avoidance neural composite, so three independent analyses converge on the same two levers.
- The effects are modest (cross-domain AUC ~ 0.54, optimization lift 0.03 to 0.09). This is a real,
  generalizing, optimizable signal, not a strong predictor. Belief is heterogeneous (the persuasion
  meta-analysis I-squared = 76%), so a small cross-domain effect is expected and is honestly reported.

## The optimization loop, stated honestly

Measure: the belief score is computed from content. Optimize: move the robust, brain-consistent factors
to raise it, with a measured lift. This is a counterfactual sensitivity analysis on interpretable factors,
not a deployed text generator. Closing the loop to a full content generator that maximizes a validated
fMRI-encoder belief score is the remaining step and needs the learned encoder. What is demonstrated here
is that the measure-and-optimize principle works end to end on real belief data with a brain-structured,
cross-domain-generalizing model.

## References

Wang, E. Y., et al. (2025). Foundation model of neural activity predicts response to new stimulus types.
Nature.
Walters, J., et al. (2022). Predicting brain activation maps for arbitrary tasks with cognitive encoding
models. NeuroImage.
Kriegeskorte, N., Douglas, P. K. (2018). Interpreting encoding and decoding models. Current Opinion in
Neurobiology.
