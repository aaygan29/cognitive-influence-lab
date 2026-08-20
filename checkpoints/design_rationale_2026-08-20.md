# Design rationale and risk register (internal, non-paper)

Internal engineering note that fixes the experimental direction and the scope. Not a manuscript
artifact and never cited in the paper. Its only job is to make sure the experiments we run are the
right ones.

## Scope decision
Central scope = five claims C1–C5 (see README). The direction "predict an individual's actions from
their neurology" is **out of scope as a measured claim**: there is no neural ground truth for any
individual and the target decisions are researcher-coded, so it is not identifiable. It is
reformulated as C5 (predict where identifiable, abstain where confounded) and, via the attribution-
scale ladder (C6, extension), as an empirical question about *where* along the aggregate-to-individual
axis inference can survive.

## Claim scoping table
| # | Direction | Evidence tier the design targets | Disposition |
|---|---|---|---|
| C1 | AIM-DDM phenotype identifiable | T4/T5 (parameter recovery) | in scope |
| C2 | honesty layer: coverage + calibration + abstention | T4/T5 (proof + simulation) | in scope, machinery built |
| C3 | affective channel forecasts beyond behavior | T3->T4 (real neural data) | in scope, main empirical bet |
| C4 | pooling lowers MDES | T4 | in scope |
| C5 | predict-or-abstain on confounded targets | T4 | in scope |
| — | individual neurology as a measured quantity | T1 | out of scope; reformulated into C5/C6 |

## Risk register (rigor checks that steer the experiments)
- **Provenance** — real neural data (NARPS, DEAP) only; no synthetic stand-in reported as real; no
  individual is scanned.
- **Variance** — every headline is multi-seed (>=20) with an effect size and CI; source-field studies
  run at n~28-39 subjects and ~30-140 items, the regime where single-run effects are fragile.
- **Specificity** — the affective/integrative split must beat a content-blind 2-factor decomposition of
  equal capacity, or the neural framing is unearned. Required ablation in E3.
- **Confounds** — partial out the economic baseline (gain/loss magnitude), event severity, base rate.
- **External validity** — population->individual transfer is supported only population->population in the
  source field; treat it as the central open risk and test it with the scale ladder, do not assume it.
- **Measurement** — decision coding needs >=2 blind coders + Cohen's kappa before any C5/C6 number;
  information confound (advisor-filtered outcomes) and attribution confound (individual vs institution)
  are exactly what the scale ladder measures.
- **Reproducibility** — inherit the harness leakage guards (subject- and stimulus-grouped CV, nested
  out-of-fold weighting).
- **Analytic integrity** — preregister the temporal split and kill criteria before touching any
  decision corpus.

## Error analysis that sets the tests
- Affective-vs-behavioral transfer: Type-I risk from ROIs x arms; control with a permutation null +
  FDR across arms; settle with within-estimator affective ΔCRPS under stimulus-grouped CV.
- Individual-scope inverse: Type-I + Type-II; ~50-150 codable decisions per term -> wide MDES + analyst
  degrees of freedom; the abstention gate should FIRE at the individual scope. If it does not, the gate
  is miscalibrated, and that is a finding, not a success.

## The decisive experiment
Run E1 (synthetic parameter recovery) + E3 (within-estimator affective-vs-behavioral forecast on NARPS
ds001734), multi-seed with the harness permutation null. Clean recovery + affective ΔCRPS CI excluding 0
under stimulus-grouped CV means the C1–C4 core is solid and C5/C6 ride as the honest external-validity
probe. If affective transfer vanishes inside the unified estimator, C3 narrows and the paper leans on
C1/C2/C4 plus the scale-ladder result.
