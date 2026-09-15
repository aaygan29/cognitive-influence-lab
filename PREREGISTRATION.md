# Pre-registration: neural-grounded belief-imparting risk

Aayush Gandhi. Locked hypotheses and kill criteria for the real-data run. No em dashes.

This file fixes the confirmatory analysis before the real public-corpus data is touched. The apparatus
that runs these tests is validated on synthetic ground truth (see paper/manuscript.md, Section 3, and the
positive-and-negative-control test suite).

## Question

Does a neural belief-formation signature B(s), predicted from content alone by a frozen brain encoder,
carry information about whether that content imparts belief, over and above a strong content-only model,
when validated against measured human outcomes in public data?

## Signature

B = E + V - R, with E = MPFC engagement (Falk 2010), V = NAcc value (Genevsky 2017; Knutson 2018),
R = DMN + dmPFC resistance (Kaplan 2016; Kossowska 2026), read from a frozen encoder's predicted
activation profile via Neurosynth ROI masks (Yarkoni 2011).

## Outcome

Belief is a latent factor from a one-factor model over: self-reported attitude, truth discernment, MIST
susceptibility, attitude strength, and behavior (confirmatory). Primary outcome is the latent factor;
behavior is the confirmatory single indicator.

## Confirmatory hypotheses and kill criteria (locked)

- H-load: corr(B, latent belief factor) has a bootstrap 95% CI excluding 0 and a point value at or above
  0.15. Kill: fails either condition.
- H-incremental: adding B to a content-only logistic model raises held-out AUC for measured behavior;
  the paired delta-AUC bootstrap 95% CI excludes 0 and is positive. Kill: CI includes 0 or is negative.
  This is the load-bearing test.
- H-encoder: H-load and H-incremental hold with the same sign across at least two independent encoders
  (not weight-perturbations of one encoder). Kill: a verdict that flips on encoder swap is reported as
  encoder-specific and dropped.
- H-bound: the AUC of B alone for measured behavior, with bootstrap CI, reported as the risk magnitude.
  Descriptive, no kill criterion.

## Honesty layer

Every prediction is wrapped in split-conformal prediction; empirical coverage at the target level is
reported, and the instrument abstains under distribution shift rather than asserting.

## Data (public, no new collection)

ChangeMyView / Winning Arguments (Tan 2016), Persuasion-for-Good (Wang 2019), MIST norms (Maertens 2023),
Neurosynth (Yarkoni 2011), a frozen open encoder (e.g. TRIBE v2). Provenance in DATA_MANIFEST.json.

## Scope and ethics

The deliverable is a demonstration and quantification of risk and a measurement instrument, not a
persuasion optimizer. No targeting of named individuals. A negative result (B does not load or adds
nothing) is a valid, reportable bound.
