# Raw ABIDE data removed

`data_abide/` (~89M, CC200 ROI timeseries .1D files, ~248 subjects) was deleted to reclaim
disk. This was always a runtime cache, not redistributed data — the README already
documents this:

> ABIDE (exp05) is fetched at runtime from the public `s3://fcp-indi` bucket (anonymous,
> no credentials), CC200 ROI timeseries, via the Preprocessed Connectomes Project.

Re-fetch by just re-running the experiment:
```
python3 exp05_abide_wiring_vs_weights.py   # re-fetches ~300 ABIDE subjects, ~2-3 min
```
`results/*.json` (already computed) are retained regardless, so downstream figures still
regenerate without re-fetching.
