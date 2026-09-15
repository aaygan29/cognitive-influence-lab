# The real-data result (ChangeMyView, Tan et al. 2016)

Aayush Gandhi. What is real, what is null, what remains. No em dashes. Numbers from
results/realdata_cmv.json; reproduce with `python -m src.run_real`.

## Data

ChangeMyView / Winning Arguments corpus (Tan et al. 2016), via ConvoKit. Arguments that challenged an
original poster's view, labeled by whether the poster awarded a delta (view changed). n = 19571 labeled
arguments (12327 successful). The corpus provides matched pairs: a successful and an unsuccessful
challenger to the same original post. 3866 pairs have both sides with usable text. Within a pair, topic and
audience are fixed, which controls the confounds that dominate persuasion outcomes.

## Features

Deterministic, from text alone (src/realdata.py). A generic content baseline (length, lexical diversity,
questions, links, numbers) and three persuasion-neuroscience-grounded channels: engagement E (evidence and
analytic framing, the MPFC leg), value V (affect and social proof, the NAcc leg), resistance R (identity
threat and aggression, the DMN/dmPFC leg). Text-proxy signature B = z(E) + z(V) - z(R).

## Results

| Test | Result | Reading |
|---|---|---|
| Within-pair, engagement E | win rate 0.579, Wilcoxon p = 9e-25 (null 0.506 +/- 0.008) | REAL. The engagement channel is higher for the winning argument, topic held fixed. |
| Within-pair, value V | win rate 0.487, p = 0.07 | NULL. Affect does not win deltas in this corpus. |
| Within-pair, resistance R | successful arguments score higher on the R proxy | The R text-proxy runs opposite to the receiver-resistance theory. |
| Within-pair, composite B | win rate 0.535, p = 2e-5 | Real but dominated by E and dragged down by the R proxy. |
| Pooled incremental AUC (B over content) | content 0.569 -> content+B 0.569, delta not positive, grouped CV | NULL. No pooled increment over generic content features. |
| Permutation null | 0.506 +/- 0.008 | The tests are calibrated; the E effect is far outside it. |

## Honest conclusions

1. One channel of the neural-grounded signature, engagement, has a real, confound-controlled, highly
   significant relationship to genuine belief change (p = 9e-25). This is the medial-prefrontal
   persuasion-uptake leg (Falk et al. 2010) recovered on real deltas, and it matches the original study's
   finding that evidence and calibrated argument win.
2. The a priori composite was NOT re-weighted to fit the outcome. Reporting it as-is shows the value and
   resistance text-proxies do not transfer, which is a real finding about the operationalization, not a
   failure to hide.
3. The pooled increment over content is null, consistent with the meta-analytic heterogeneity
   (I-squared = 76%): pooled persuasion prediction is weak because context dominates, and the signal shows
   in the within-pair, context-controlled design.

## What remains (the two boundaries)

- The signature is a text proxy, not a learned brain encoder. Swapping in an encoder (TRIBE v2 or the
  author's own) is the registered next step and is a two-line change (docs/DATA.md).
- The pooled increment is null. Whether a learned encoder produces a pooled increment is the open
  empirical question the instrument exists to answer.
