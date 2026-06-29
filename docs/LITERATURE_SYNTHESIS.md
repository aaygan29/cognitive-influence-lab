# Literature synthesis: LLM-driven persuasion and belief dynamics (10-year scan)

Aayush Gandhi. The field scan behind MODEL.md and the statistical-validity choices. No em dashes.

## 1. Effect size and the heterogeneity problem (the statistical-validity core)

- Meta-analysis of LLM persuasion (Holbling et al. 2025, Scientific Reports): 7 studies, 17,422
  participants, 12 effects. Mean LLM-vs-human difference Hedges g = 0.02 (not significant). Between-study
  heterogeneity I-squared = 76%. Moderators (model, conversation design, domain) jointly explain
  R-squared = 82% of between-study variance.
- LLM persuasion is real in absolute terms (Bai et al. 2025, Nature Communications, N = 4829 across three
  pre-registered experiments; Hackenburg et al. 2025, Science, N = 76,977, 19 models; Argyle et al. 2025,
  PNAS, four AI strategies all significant).
- Microtargeting and cognitive elaboration add little over a strong generic message (Argyle et al. 2025;
  Hackenburg et al. 2024, 2025). Persuasiveness comes mostly from post-training and prompting, and the
  same methods that raise persuasion lower factual accuracy (Hackenburg et al. 2025).

Implication for cultist: the effect is real but heavily context-moderated. A single-corpus estimate of
any persuasion quantity, including the neural loading w_n, will not transfer. The estimation must be a
hierarchical model with partial pooling across corpora and explicit moderators, and generalization must be
judged on the random-effects prediction interval, not a single-corpus confidence interval (MODEL.md
Section 6).

## 2. Belief-updating mechanics (the force and its gates)

- Bayesian persuasion as information design (Kamenica and Gentzkow 2011); concavification extends to
  non-Bayesian updating (de Clippel and Zhang 2022); heterogeneous priors change when persuasion succeeds
  (Alonso and Camara 2016); coarse Bayesian updating formalizes documented deviations (Jakobsen 2026).
- Updating is asymmetric and confirmation-biased, and the asymmetry is traceable to prediction-error
  signaling in reward circuitry (Palminteri et al. 2022; Lefebvre et al. 2017; Coutts 2019). This grounds
  the asymmetric learning-rate gate (lambda_plus > lambda_minus) and ties it to the same reward-system
  signals the neural value term V reads.

## 3. Opinion dynamics and robustness (the anchor and the band)

- Friedkin-Johnsen: partial stubbornness, agents anchored to their initial opinion, factored into every
  update step; full or partial stubbornness produces opinion cleavage rather than consensus (Bernardo et
  al. 2024 survey). This is the susceptibility parameter s and the anchor term.
- Hegselmann-Krause bounded confidence: influence acts only within a confidence band of opinion
  similarity (Bernardo et al. 2024 survey). This is the gate g_d and the source of the small-step
  (foot-in-the-door) optimum.
- Networked-LLM opinion evolution is now itself a studied object (arXiv 2606.18276), confirming the
  opinion-dynamics frame transfers to LLM agents.

## 4. Resistance and inoculation (the defender's side)

- Inoculation theory confers resistance to persuasion across domains, including health and extremist
  propaganda, via reactance and credibility reduction (Compton et al. 2016; Braddock 2019).
- Attitude strength, ambivalence, and resistance are core moderators of persuadability (Crano and Prislin
  2005).

These map onto raising involvement alpha and anchoring (1 - s) and narrowing the band d, which is how the
model computes an inoculation dose (MODEL.md Section 5.2).

## 5. Dynamical-systems form (the geometry)

- The cusp catastrophe gives bistability and hysteresis in gradient systems (Thom; behavioral and
  perceptual applications widely documented). The specific application to attitudes is van der Maas,
  Kolstein, and van der Pligt (2003), who fit a cusp to sudden attitude transitions. This is the model's
  potential V(b) and the source of belief robustness as distance to the fold.

## 6. Control and optimization (the policy)

- Persuasive dialogue is naturally a POMDP over a belief state, with reinforcement learning for the
  policy (Natural Actor Belief Critic and successors; belief-state tracking plus RL). This is the frame
  for the persuasion-optimization problem (MODEL.md Section 5.1) and makes "potency" the controllability
  of the belief dynamics.

## 7. What the scan changed in cultist

1. Generalization is now a hierarchical-model claim on the random-effects distribution of w_n, with a
   prediction-interval kill criterion, directly answering "would it work on a broader corpus."
2. The instrument's neural term is embedded as one coefficient in a dynamical model, so its incremental
   value is interpretable as a force coupling, not just an AUC bump.
3. Belief robustness has a closed-form definition (well stiffness; distance to the fold; involvement^1.5
   scaling) and an inoculation dose, giving the defensive payoff a quantitative target.
