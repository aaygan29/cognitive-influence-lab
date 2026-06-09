# Exclusion rules (fixed before analysis)

Applied identically to CSI and all baselines; defined here **before** any outcome is examined.

- Drop records with non-finite or out-of-range fields (e.g., malformed gambles, empty text).
- Drop duplicate records by primary key (keep first by documented source order).
- Text corpora: drop deleted/removed posts and bot / automoderator content.
- choices13k: include only problems meeting the source's documented inclusion criteria; record
  pre/post counts.

Any deviation is logged in `DEVIATIONS.md`.
**Status:** to be finalized with per-dataset specifics before registration.
