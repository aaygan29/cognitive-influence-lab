# A Formal Model of Belief Dynamics, Persuasion Potency, and Belief Robustness

Aayush Gandhi. Companion to the cultist instrument. No em dashes. Every modeling choice is anchored to a
cited result from the persuasion, opinion-dynamics, computational-neuroscience, or dynamical-systems
literature, so the model is a synthesis of the field rather than an ad hoc construction.

## 0. Why a formal model, and what the field requires of it

A single-corpus demonstration cannot establish that a persuasion effect generalizes. The meta-analytic
picture makes this concrete: across studies, large language models and humans do not differ in mean
persuasive effect (Hedges g = 0.02, not significant), but between-study heterogeneity is large
(I-squared = 76%), and study-level moderators (model, conversation design, domain) jointly explain most of
the variance (R-squared = 82%) (Holbling et al. 2025). Persuasion is therefore a moderated, context-heavy
effect, not a constant. Two consequences shape this model:

1. The object to model is not a fixed persuasion magnitude but a belief-dynamics process whose parameters
   vary across people and contexts. Generalization is a statement about a distribution of parameters, not
   a point estimate (Section 6).
2. The neural-grounded message quantity B(s) from the instrument enters as one term in the persuasive
   force, and its contribution is a parameter to be estimated with partial pooling, not assumed constant.

## 1. Belief as a state in a potential

Let a target hold a scalar belief b in R on a given issue (an attitude position; the scalar case extends
coordinate-wise to a vector of issues). Equilibrium beliefs are the minima of a potential V(b). We use the
cusp form, the lowest-order potential that produces bistability and hysteresis. The application of the
cusp catastrophe to attitudes is established prior art (van der Maas, Kolstein, and van der Pligt 2003,
who fit a cusp to sudden attitude transitions); strong-attitude resistance and conversion are reviewed in
Crano and Prislin 2005. We adopt their model form and add the neural-grounded force term of Section 2:

    V(b; alpha, beta) = (1/4) b^4 - (1/2) alpha b^2 - beta b.                         (1)

Gradient (relaxation) dynamics with noise:

    db/dt = -dV/db = -b^3 + alpha b + beta + xi(t),   xi ~ N(0, sigma^2).             (2)

- beta is the net persuasive pressure toward higher b (the "normal" factor of the cusp).
- alpha is attitude involvement or strength (the "splitting" factor). For alpha <= 0 the potential is
  monovalent (one stable belief); for alpha > 0 it is bistable (two entrenched positions separated by a
  barrier), which is the formal signature of a strong, identity-linked attitude (Kaplan et al. 2016
  locate belief resistance in self-referential default-mode cortex, the neural correlate of large alpha).

This single equation already yields three field-documented phenomena: graded change for weak attitudes
(alpha <= 0), resistance and sudden conversion for strong ones (alpha > 0, a fold crossing), and
hysteresis (a moved belief does not return when the pressure is removed).

## 2. The persuasive force, grounded in the instrument

The force beta is produced by a message m. We decompose it into a content component and the
neural-grounded component measured by the cultist instrument:

    beta_raw(m) = w_c * phi_content(m) + w_n * B(m),                                   (3)

where phi_content(m) is a content-only persuasion score and B(m) = E(m) + V(m) - R(m) is the neural
belief-formation signature (engagement E in medial prefrontal cortex, value V in nucleus accumbens,
resistance R in default-mode and dorsomedial prefrontal cortex; Falk et al. 2010; Genevsky et al. 2017;
Kaplan et al. 2016). The instrument's central hypothesis is w_n > 0 after partialling out phi_content
(the H-incremental test). Equation (3) is where the empirical instrument and the dynamical model meet:
the quantity the instrument validates is exactly the neural loading on the persuasive force.

## 3. Two gates the field requires: bounded confidence and asymmetric updating

A raw force does not act in full. Two well-established mechanisms gate it.

Bounded confidence (Hegselmann and Krause; Friedkin and Johnsen surveyed in Bernardo et al. 2024): a
message only moves a belief if its advocated position pos(m) is within a confidence band d of the current
belief. Outside the band the message is rejected or backfires.

    g_d(b, m) = 1 if |pos(m) - b| <= d, else 0.                                        (4)

Asymmetric, confirmation-biased updating (Palminteri et al. 2022; Lefebvre et al. 2017; Coutts 2019):
prior-confirming evidence is weighted more than disconfirming evidence, with the asymmetry traceable to
prediction-error signaling in reward circuitry. Let lambda_plus >= lambda_minus > 0:

    a(b, m) = lambda_plus  if sign(pos(m) - b) moves toward the person's prior pole,
              lambda_minus otherwise.                                                  (5)

The effective force is

    beta_eff(b, m) = a(b, m) * g_d(b, m) * beta_raw(m).                                (6)

## 4. Anchoring: the Friedkin-Johnsen robustness term

People do not fully assimilate; they stay partly anchored to their initial belief b0. The Friedkin-Johnsen
model captures this with a susceptibility s in [0, 1] (s = 1 fully open, s = 0 fully stubborn). Adding the
anchor to (2):

    db/dt = -b^3 + alpha b + s * beta_eff(b, m) + (1 - s) * kappa * (b0 - b) + xi.     (7)

Equation (7) is the full belief-dynamics model. Its parameters partition cleanly into a persuasion channel
(beta_raw, hence w_c, w_n, the message design) and a robustness channel (alpha, s, d, lambda asymmetry,
b0), which is what makes both optimization problems in Section 5 well posed.

## 5. The two optimization problems

### 5.1 Persuasion potency (the attacker's problem, characterized for defense)

Given a target with belief b_t and parameters, and a horizon T with message budget C, choose a message
sequence m_{1:T} to maximize the belief shift toward a goal position b_goal:

    maximize_{m_{1:T}}  E[ b(T) - b(0) ]  toward b_goal
    subject to          dynamics (7),  sum_t cost(m_t) <= C.                           (8)

Because the belief state is only partially observed (we see behavior and self-report, not b directly),
(8) is a partially observable Markov decision process over the belief state, the standard frame for
optimal persuasive dialogue (POMDP belief-state control; Natural Actor Belief Critic and successors). Two
predictions fall out and are falsifiable:

- Small-step optimality. The bounded-confidence gate (4) makes a single large jump (|pos(m) - b| > d)
  ineffective. The optimal policy advances the advocated position in increments that stay inside the band,
  the formal version of foot-in-the-door. This predicts sequential, belief-tracking persuasion beats
  one-shot messaging, and that microtargeting adds little once the steps are well placed, consistent with
  the null microtargeting effects in Argyle et al. 2025 and Hackenburg et al. 2025.
- Potency is the controllability of (7). The neural term w_n * B(m) raises potency only to the extent it
  enlarges the in-band, prior-confirming component of the force, which is exactly the quantity B is built
  to read.

### 5.2 Belief robustness (the defender's problem)

Define local robustness as the stiffness of the belief well at the current equilibrium b*:

    rho_local = V''(b*) = 3 b*^2 - alpha,                                              (9)

and the susceptibility (the inverse, the field's natural sensitivity measure):

    chi = d b* / d beta = 1 / rho_local = 1 / (3 b*^2 - alpha).                        (10)

Global robustness is the distance to the cusp fold, the net force needed to trigger a discontinuous jump.
The fold (double-root) set of b^3 - alpha b - beta = 0 is 4 alpha^3 = 27 beta^2, so the critical force at
involvement alpha is

    beta_crit(alpha) = sqrt( (4/27) * alpha^3 ) = (2 / (3 sqrt 3)) * alpha^(3/2),  alpha > 0,   (11)

and global robustness is the margin

    rho_global = beta_crit(alpha) - | s * beta_eff |.                                  (12)

The defender (inoculation) maximizes rho_global. Equations (9)-(12) say precisely how: raise involvement
alpha (attitude-strength and identity framing), lower susceptibility s and raise anchoring (1 - s)
(commitment, forewarning), and narrow the confidence band d (reactance, source-distrust priming). This is
the quantitative form of inoculation theory (Compton et al. 2016; Braddock 2019), and it yields an
inoculation dose: the increase in alpha or anchoring needed to push beta_crit above the strongest force an
adversary can deliver.

## 6. Statistical validity: the generalization model

The meta-analytic heterogeneity (I-squared = 76%, Holbling et al. 2025) is a warning that a single-corpus
estimate of w_n will not transfer. We therefore specify the estimation as a Bayesian hierarchical
(multilevel) model with partial pooling across corpora and domains. This is a specification for the
real-data run, not a model already fit to data:

    y_{ij}      = persuasive outcome for item i in corpus j
    y_{ij}     ~ Bernoulli( logit^{-1}( eta_{ij} ) )
    eta_{ij}   = w_c_j * phi_content(m_i) + w_n_j * B(m_i) + X_i' gamma                (13)
    w_n_j      ~ Normal( mu_wn, tau^2 )         (random slope per corpus)
    mu_wn, tau, w_c_j, gamma  ~ weakly informative priors,

with corpus-level moderators (model, conversation design, domain) entered as covariates because they
explain most between-study variance (R-squared = 82%). The generalization claim is not "w_n > 0 in one
corpus." It is that the posterior mean mu_wn is positive and, more stringently, that the predictive
interval for a new corpus w_n_{new} ~ Normal(mu_wn, tau^2) excludes zero. Reporting the prediction
interval rather than the confidence interval is the honest test of whether the neural increment survives
into an unseen corpus, and tau quantifies exactly the heterogeneity the meta-analysis flagged.

This converts "would it work on a broader corpus" from a hope into a measured quantity: the sign and mass
of the random-effects distribution of w_n, with the kill criterion stated on the prediction interval.

## 7. What is novel, stated honestly

None of the pieces is new. The cusp model of attitudes, Bayesian and non-Bayesian persuasion, bounded
confidence, Friedkin-Johnsen anchoring, asymmetric reinforcement-learning updates, POMDP persuasion, and
multilevel meta-analysis are all established. The contribution is their assembly into one model in which
(a) the persuasive force carries a neural-grounded term B(m) that an instrument can measure on content
alone, (b) belief robustness is a computable functional of the same dynamics (Eq. 9-12), and (c) the
generalization claim is pinned to the random-effects distribution of the neural loading (Eq. 13). The
model makes the instrument's H-incremental hypothesis a statement about a coefficient in a dynamical
system, and it makes inoculation a dose computed from the same equations that describe the attack.

## 8. Predictions to be tested empirically

The four results below are analytic consequences of the model, verified in the implementation
(src/belief_model.py, locked by tests): they confirm that the model is correctly built and internally
consistent, not that it is empirically true. Each is a falsifiable prediction for experiment, and
confirming or breaking it against data is the next step. The implementation values quoted are the model's
own output, not measurements of people.

1. Sequential in-band messaging produces larger total belief shift than a single large-step message of
   equal content strength (bounded-confidence gate, Eq. 4 and Section 5.1).
2. The neural loading w_n is positive in the random-effects distribution (Eq. 13); if its prediction
   interval includes zero, the neural increment does not generalize and the risk is content-level.
3. Robustness scales with involvement as beta_crit ~ alpha^{3/2} (Eq. 11); inoculation that raises alpha
   should raise the critical force super-linearly, a sharper prediction than "inoculation helps."
4. Hysteresis: a belief moved across the fold does not return when the force is removed, so persuasion and
   de-persuasion are asymmetric in cost. This is directly testable with reversal designs.

## References

Argyle, L. P., et al. (2025). Testing theories of political persuasion using AI. PNAS.
Bai, H., et al. (2025). LLM-generated messages can persuade humans on policy issues. Nature Communications.
Bernardo, C., et al. (2024). Bounded confidence opinion dynamics: A survey. Annual Reviews in Control.
Braddock, K. (2019). Vaccinating against hate: attitudinal inoculation vs extremist propaganda. Terrorism
and Political Violence.
Compton, J., et al. (2016). Persuading others to avoid persuasion: inoculation theory and resistant health
attitudes. Frontiers in Psychology.
Coutts, A. (2019). Good news and bad news are still news: experimental evidence on belief updating.
Experimental Economics.
Crano, W., and Prislin, R. (2005). Attitudes and persuasion. Annual Review of Psychology.
de Clippel, G., and Zhang, X. (2022). Non-Bayesian persuasion. Journal of Political Economy.
Falk, E., et al. (2010). Predicting persuasion-induced behavior change from the brain. Journal of
Neuroscience.
Friedkin, N., and Johnsen, E. (1990 and after). Social influence and opinions; the Friedkin-Johnsen model.
Genevsky, A., et al. (2017). When brain beats behavior: neuroforecasting crowdfunding outcomes. Journal of
Neuroscience.
Hegselmann, R., and Krause, U. (2002). Opinion dynamics and bounded confidence. JASSS.
Holbling, L., et al. (2025). A meta-analysis of the persuasive power of large language models. Scientific
Reports.
Kamenica, E., and Gentzkow, M. (2011). Bayesian persuasion. American Economic Review.
Kaplan, J., et al. (2016). Neural correlates of maintaining one's political beliefs. Scientific Reports.
Lefebvre, G., et al. (2017). Behavioural and neural characterization of optimistic reinforcement learning.
Nature Human Behaviour.
Palminteri, S., et al. (2022). The computational roots of positivity and confirmation biases in
reinforcement learning. Trends in Cognitive Sciences.
van der Maas, H., Kolstein, R., and van der Pligt, J. (2003). Sudden transitions in attitudes.
Sociological Methods and Research.
