# Deviations & decision log

Any change to `PREREGISTRATION.md` after OSF timestamping is logged here with date + rationale.
This preserves the audit trail.

## 2026-06-08 — pre-registration hardening (adversarial review, round 1)
Six-reviewer adversarial panel, run **before any outcome data were examined**. Actions taken:

- **Decision rule reconciled with the registered statistics.** Family-level Benjamini-Hochberg
  FDR implemented in code (`validate.decide_family`); `Finding.decide` documented as a
  single-hypothesis primitive.
- **Leakage control encoded, not just promised.** Group-keyed splitter + frozen split hash +
  leakage test (`spikeprint/splits.py`, `tests/test_splits.py`) to prevent problem-level outcome
  leakage on aggregate data (choices13k).
- **CSI engine pinned.** `neurosignal @ e05bdb2` recorded in `csi_provenance.json`; learned-encoder
  weights to be SHA-256-hashed before any use.
- **Redistribution/PII guard added** (`scripts/check_no_raw_data.py`) and wired into CI.
- **Numeric MDES recorded** via `scripts/power.py`; **synthetic (LLM) arm** specified as separate
  and never pooled into the confirmatory human estimate.
- **Citation fixes:** "Botvinik-Nezer" (one *c*); covariate adjustment attributed to Rubin (1974),
  with Kleiner cited only for the indirect-vs-direct test distinction.

No outcome data were examined during this round.

## 2026-06-08 — analysis iteration + adversarial review round 2 (verdict: major-revisions)
Pillar-A pipeline added and re-reviewed. Fixed before commit:
- **Correctness bug:** `decide_family` applied a single null to mixed-null metrics (AUC null 0.5
  vs dAUC null 0.0), so an AUC CI that excludes 0.0 but includes 0.5 could wrongly pass. Moved
  `null` onto the `Finding`; `decide`/`decide_family` now use the per-Finding null. Added
  regression tests.
- **Construct validity:** H1 split into **H1a (value-framing, NARPS)** and **H1b (persuasive-text,
  CMV/P4G)** — reported separately, never pooled; cross-construct transfer labeled exploratory.
- **Resample integrity:** degenerate (single-class) resamples now counted and surfaced on every
  Finding; >1% flagged (registered threshold in `analysis.py`).
- **Reproducibility:** pytest `pythonpath=["."]` so tests do not depend on a flaky editable install.
- Citation/data: NARPS (CC0) is primary; choices13k deferred (no license at source).
No outcome data were examined during this round.
