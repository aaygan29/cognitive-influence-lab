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
