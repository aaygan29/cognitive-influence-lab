# Agent-contamination sensitivity, aggregate arm (reviewer action #4)

AGEA (Özüdoğru et al. 2026, *Generalizable experiment agents threaten online
behavioral experiments*) shows a consumer LLM agent can complete online
behavioural experiments with human-range behaviour. The aggregate / market arms
here draw on online populations, so some "participants" may be agents whose
features look human but whose outcome does not follow the human
feature→outcome law. The person-level splits do not guard against this (an agent
is a whole fake person, split cleanly like any other). This treats the
contamination fraction `f` as a nuisance parameter and measures the aggregate
decode's degradation.

Script: [`scripts/agent_contamination_sensitivity.py`](../scripts/agent_contamination_sensitivity.py);
figure: [`figures/agent_contamination_sensitivity.pdf`](../figures/agent_contamination_sensitivity.pdf);
numbers: [`results/agent_contamination_sensitivity/sensitivity.json`](../results/agent_contamination_sensitivity/sensitivity.json).

## Model
Inject agent rows into the de-leaked Kiva arm until a fraction `f` of the dataset
is agents. Agent feature rows are **bootstrapped from real applicants** (human-like
marginals and correlations, as AGEA agents produce), but each agent's label is
drawn **independently at the base rate** -- the agent carries no genuine
feature→outcome signal. Both train and test are contaminated (agents are
indistinguishable participants). Balanced accuracy, StratifiedKFold, 25 seeds.

## Result

| fraction agents `f` | balanced accuracy |
| --- | --- |
| 0.00 (clean) | 0.605 ± 0.004 |
| 0.05 | 0.601 ± 0.005 |
| 0.10 | 0.594 ± 0.006 |
| 0.20 | 0.585 ± 0.006 |
| 0.30 | 0.574 ± 0.006 |
| 0.50 | 0.550 ± 0.007 |
| 0.70 | 0.528 ± 0.006 |

The decode degrades **roughly linearly** with contamination: the lift over
chance (0.105 at f=0) halves at **f ≈ 0.5** and is still marginally above chance
at f=0.7. So contamination is a graceful degradation, not a cliff -- the arm
tolerates a minority of agents, but a majority-agent population erases most of
this (already modest) records signal.

## What this means for the aggregate claim
- Any aggregate / market-level number should be reported with a
  contamination-sensitivity band, not as a point estimate, because the online
  population cannot be assumed agent-free.
- The practical exposure here is limited: the de-leaked Kiva arm is weak to begin
  with (0.605, below its 0.723 majority baseline), so there is little signal for
  contamination to threaten. The analysis matters more as a template for the
  stronger aggregate arms the framework is meant to generalise to.

## Caveats
- This is the **conservative** contamination model (agents as pure label noise
  on human-like features). Two directions change it: if agents are *detectable*
  (non-human feature marginals), a screen removes them and the degradation is
  milder; if agents *partially mimic* the human feature→outcome mapping (AGEA
  agents do reproduce condition effects), dilution is weaker and the curve is
  flatter -- but the aggregate outcome is then driven partly by synthetic
  behaviour, which is its own validity problem.
- Degradation was measured on the one loadable aggregate arm (Kiva). The
  crowdfunding/market arms the threat model names would show the same mechanism;
  the curve here is the method, not a universal constant.
