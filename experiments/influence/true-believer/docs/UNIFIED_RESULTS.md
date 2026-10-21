# Unified results: three real-data arms of one signature

Aayush Gandhi. What the E + V - R signature does across text and neural data. No em dashes. All numbers
from committed results JSON; reproduce with the named commands.

The signature has one structure, B = z(E) + z(V) - z(R): engagement (medial prefrontal), value (nucleus
accumbens), resistance (default-mode / insula). We now test it on three independent real datasets.

## Arm 1: text, ChangeMyView (Tan 2016). `python -m src.run_real`

Within matched argument pairs (topic fixed), the engagement channel is higher for the argument that
actually changed a view: win rate 0.579, Wilcoxon p = 9e-25, permutation null 0.506. Value is null, the
resistance text-proxy runs the wrong way, and the pooled increment over content is null. Real,
confound-controlled, but the engagement leg carries it.

## Arm 2: text, Persuasion-for-Good (Wang 2019). `python -m src.multicorpus`

A second, different domain (charity-donation dialogues, outcome = donated money). The engagement
coefficient points the same positive way as in ChangeMyView (CMV +0.076 significant; P4G +0.059 same sign,
not individually significant; pooled fixed-effect +0.075, 95% CI [0.036, 0.114]). Directional
generalization, significant when pooled, second corpus underpowered.

## Arm 3: real fMRI, Knutson neuroforecasting (Kuhnen and Knutson 2005). `python -m src.neuralforecast`

The real-neural test. Group-averaged BOLD in NAcc, MPFC, and anterior insula predicts the aggregate
fraction of people who bought each stock, across two experiments. The signature computed from real fMRI,
B = z(MPFC) + z(NAcc) - z(insula), predicts aggregate choice out-of-sample (cross-experiment r = 0.182,
AUC 0.571, permutation p = 0.003). The composite is the only predictor sign-stable across both experiments;
single ROIs flip sign and do not generalize. Honest caveat: the effect is near-null in experiment 1
(B r = 0.05) and strong in experiment 2 (B r = 0.32), so it is weak and experiment-dependent.

## What the three arms say together

1. The same E + V - R structure carries a real, out-of-sample, permutation-validated relationship to
   behavior in both text persuasion (engagement channel) and real neural choice (composite). Two very
   different data types, one signature.
2. The composite is more robust than its parts. On the neural data the single ROIs do not generalize
   across experiments but the composite does. This matches the neuroforecasting finding that combined
   affective components generalize better than single regions (Genevsky et al. 2025) and is the strongest
   methodological point here: use the composite, not any one region or channel.
3. The effects are real but modest and, on the neural data, fragile. None of this is a large effect. The
   honest headline is that a neuroscience-grounded composite signature carries a small, replicated,
   out-of-sample signal about behavior across text and brain data, not that it is a strong predictor.

## The boundaries, stated plainly

- Text arms use a persuasion-neuroscience-grounded text proxy, not a brain encoder. The neural arm uses
  real fMRI but a different behavior (financial choice) than belief change, and it is experiment-dependent.
- No single dataset closes the loop from AI content, to a neural state, to a belief outcome. Each arm
  tests one link. The program's value is that the same signature structure shows a consistent, if small,
  signal across all three, which is the evidence that the construct is not vacuous.

## What would make it strong rather than promising

A dataset that has all three at once: AI-generated persuasive content, a measured or well-encoded neural
response, and a measured belief outcome in the same people. That does not yet exist publicly at scale.
Until it does, the honest claim is a consistent small signal across three partial views, plus a formal
model (paper/MODEL.md) that says how the pieces should fit.
