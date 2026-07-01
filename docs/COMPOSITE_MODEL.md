# Improving the integrated neuroforecasting composite

Aayush Gandhi. What the literature says, what we changed, and what the data said back. No em dashes.
Numbers from results/composite_model.json; reproduce with `python -m src.composite_model`.

## The question

Can the integrated brain-state composite be made more robust, predictive, and generalizable by pulling in
what recent neuroforecasting and small-sample prediction work has learned?

## What the literature says

- For AGGREGATE forecasting, only anticipatory-affect regions generalize: nucleus accumbens (approach,
  positive) and anterior insula (avoidance, negative). Medial prefrontal cortex predicts INDIVIDUAL choice
  but does not generalize to aggregate outcomes (Genevsky et al. 2017; Tong et al. 2020, PNAS). So the
  robust aggregate read-out is approach minus avoidance, NAcc - AIns.
- These predictors replicate and generalize across tasks and samples (Mortazavi et al. 2025, PNAS Nexus,
  combined N = 230, triple dissociation of NAcc/MPFC approach vs AIns avoidance).
- For small, correlated samples, ridge beats lasso, stepwise, and OLS, and parsimony wins the
  bias-variance tradeoff (methods-comparison literature). At n = 274 a high-dimensional learned model is
  expected to overfit.

## What we tested (cross-experiment, out-of-sample, permutation nulls)

| Model | out-of-sample r | perm p | reading |
|---|---|---|---|
| M0 NAcc alone | -0.219 | 1.00 | single region does not generalize (sign flips across experiments) |
| M1 E+V-R (NAcc+MPFC-AIns) | +0.182 | 0.003 | the prior composite, generalizes |
| M2 affect (NAcc-AIns), MPFC dropped | +0.180 | 0.002 | literature-preferred, ties M1 with fewer parts |
| M3 ridge, all ROI x TR features | +0.093 | 0.12 | multivariate overfits, not significant |
| M4 ridge, ROI window-means | -0.089 | 0.84 | worse |
| M5 shrinkage ensemble | +0.055 | 0.25 | worse |

Window robustness of the affect composite (all reported, none selected after the fact):
early (TR 2-5) +0.130, mid (3-8) +0.180, late (5-10) +0.110, wide (2-12) +0.141. Positive in every window.

## What the data said back

1. Parsimony wins. The two fixed low-dimensional composites (M1, M2) are the only significant
   generalizers. Every learned or multivariate model (M3, M4, M5) generalizes worse, exactly as the
   small-sample literature predicts. At this sample size, complexity hurts.
2. The affect composite NAcc - AIns matches the full E + V - R while dropping MPFC, which is the
   theoretically cleaner aggregate read-out (Genevsky, Tong: MPFC does not add to aggregate forecasts).
   We adopt NAcc - AIns (approach minus avoidance) as the canonical brain-state read-out.
3. The effect is robust across anticipatory time windows (positive in all four), so it is not knife-edge.

## The honest conclusion

The way to make this more robust and generalizable was NOT a fancier model. The data are clear that at
n = 274 the parsimonious approach-minus-avoidance composite is the best generalizer, and added complexity
overfits. The genuine improvement is threefold: a theoretically cleaner read-out (drop MPFC), a
demonstration that complexity does not help here (a real robustness result others can use), and a window
sensitivity check. Larger predictive gains require more data, not more parameters. This is the bias-variance
reality of small-sample neuroforecasting, and reporting it honestly is the contribution.

## The brain-state read-out, and optimization

The canonical measurable brain-state score is s = z(NAcc) - z(AIns), approach minus avoidance, computed
from anticipatory-window BOLD. In the formal model (paper/MODEL.md) this score is the neural term in the
persuasive force. Optimizing for a result means choosing content that maximizes the predicted s subject to
the bounded-confidence and robustness constraints. The read-out is validated here on real fMRI; the
content-to-s encoder that would close the optimization loop is the remaining piece and is not yet
validated, so optimization is specified, not yet demonstrated end to end.

## References

Genevsky, A., Yoon, C., Knutson, B. (2017). When brain beats behavior: neuroforecasting crowdfunding
outcomes. Journal of Neuroscience.
Tong, L. C., et al. (2020). Brain activity forecasts video engagement in an internet attention market.
PNAS.
Mortazavi, L., et al. (2025). Deconstructing neural predictors of risky choice. PNAS Nexus.
Kuhnen, C., Knutson, B. (2005). The neural basis of financial risk taking. Neuron.
