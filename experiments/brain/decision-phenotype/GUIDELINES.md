# Guidelines — the rules every result must obey

These are not style preferences. They are the discipline that has killed real findings in
this program before (seed variance d=-1.20; a scale-confounded correlation that vanished on
partial correlation; a synthetic pilot that "fingerprinted" random seeds). Follow them or the
result does not get reported.

## Evidence tiers (a claim's language may not exceed its tier)

| Tier | Meaning |
|---|---|
| T0 | assertion, no evidence |
| T1 | single run / anecdote |
| T2 | descriptive stat, no inference |
| T3 | inferential with effect size + CI, no robustness/confound control |
| T4 | robust: multi-seed, confound-controlled, specificity-tested |
| T5 | mechanistic: necessity established, preregistered confirmatory |

"X causes Y" on T3 evidence is an overclaim. Write the tier next to every claim in `STATUS.md`.

## The gate ladder (run in order; a Gate 0 failure is terminal)

- **Gate 0 Provenance** — data real, not a synthetic stand-in presented as real; split fixed;
  chance/noise-ceiling baseline established; no train/test leakage.
- **Gate 1 Variance** — headline from >=20 seeds, effect size *and* CI, not p alone.
- **Gate 2 Spec robustness** — survives reasonable preprocessing/threshold/hyperparameter choices.
- **Gate 3 Specificity** — the *named* affective mechanism beats a generic equal-capacity alternative
  (the content-blind 2-factor ablation). Mandatory for any "neuro-grounded" claim.
- **Gate 4 Confounds** — survives partialing out the economic baseline (gain/loss magnitude),
  event severity, base rate, length, surprisal.
- **Gate 5 Mechanism** — necessity via lesioning the affective channel; does it carry the load?
- **Gate 6 Calibration** — language sized to tier. In-silico != in-vivo. Correlational != causal.
  For individuals: "revealed decision policy", never "neurology" or "psychology".
- **Gate 7 External validity** — holds across >=2 datasets / OOD, not asserted from one split.
- **Gate 8 Measurement** — instrument measures the construct, reliably. Human-coded decisions need
  >=2 blind coders + Cohen's kappa reported before modeling.
- **Gate 9 Reproducibility** — seeds, configs, splits, code, env, data versions released.
- **Gate 10 Ethics** — see below.
- **Gate 11 Analytic integrity** — preregister confirmatory analyses + kill criteria; separate
  confirmatory from exploratory; no HARKing.
- **Gate T Theory** — assumptions stated + realistic; proofs valid; bounds non-vacuous.

## The honesty layer (born into every reported number)

Every predictive number is emitted through `src/honesty.py` and carries:

1. **A strictly-proper score** (log-score / Brier / CRPS), never bare accuracy. Report calibration
   *and* sharpness.
2. **A conformal prediction set** at a stated coverage level (distribution-free, finite-sample).
3. **A bootstrap CI** and a **permutation null** (both already in the harness `evaluation/metrics.py`).
4. **An MDES check**: minimum detectable effect at this n; if the claimed effect < MDES, **abstain**
   ("unidentified") rather than report a point.
5. **A provenance stamp**: dataset id + version, split hash, seed, code commit.

Abstention is a valid, first-class output. A withheld number is stronger than a guessed one.

## Reproducibility + checkpoint discipline

- **Multi-seed by default** (>=20 where a seed exists). Report the distribution, not the best run.
- **Leakage guards are non-negotiable**: subject-grouped CV for individual claims, stimulus-grouped
  CV for aggregate/forecast claims. Never the same subject or stimulus on both sides of a fold.
- **Git commit at every checkpoint**; update `STATUS.md` in the *same* commit as any result.
  No Claude co-author trailer. Never push to `main` without an explicit ask.
- **Figures regenerate from `results/` via `figures/make_figures.py`.** No hand-edited figures;
  a figure always traces to its numbers.
- **`run_demo`-style positive controls stay green in CI.** If a positive control goes red, fix the
  pipeline; never relax the threshold.
- **README updated with every code change** (program convention): tests + docs move in the same PR.

## Ethics (Gate 10) — the individual-profiling clause

- No individual named in this work has been scanned. Never present or imply a measured neuroprofile
  of a real person. The neural grounding is a *population* property.
- Claims about a public figure are claims about their **office-level revealed decision policy** from
  the **public record only**, bounded by the information confound (advisor-filtered outcomes) and the
  attribution confound (individual vs institution). State both bounds wherever a figure is named.
- No psychodiagnosis, no clinical language, no private-trait inference. This is dual-use; the paper
  carries an honest broader-impacts section. If the framing drifts toward "profiling a person's mind",
  stop and rescope.
- **Inverse inference clause.** The inverse model outputs a posterior over a *latent computational
  state* (an affective/risk posture) that best explains a decision. That latent has a neural
  *interpretation* from the population forward map, stated as a hypothesis with its posterior width.
  It is never a claim of measured neural firing in a named individual ("decision X means region R
  fired in person P" is forbidden). At the individual attribution scope the gate is expected to
  abstain; report that, do not tune it away.

## Writing conventions

- No em dashes anywhere (use periods, commas, colons, parentheses).
- Every number in prose traces to a `results/*.json` produced by committed code.
- State what failed, what was skipped, and what is verified, plainly.
