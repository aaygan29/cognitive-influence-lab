# Ethics & human-subjects status

spikeprint collects **no** new human or animal data. It is a secondary analysis of public,
de-identified datasets (see `docs/DATA.md`).

- Under the US Common Rule (45 CFR 46), secondary analysis of publicly available,
  non-identifiable data is generally not human-subjects research. An IRB exemption/non-engagement
  determination **will be obtained and recorded here before analysis** — it is not self-certified.
- Naturalistic text corpora (ChangeMyView, Persuasion-for-Good) are used under their source
  terms. No raw post text or usernames are redistributed; only hashed IDs and derived features
  are stored (enforced by `scripts/check_no_raw_data.py`).
- IBL (animal) behavioral data are reused as a value-distortion analogue; this project runs no
  animals and makes no human claim from animal data.
- Intended use is measurement/audit research — **not** diagnosis and **not** deployment against
  individuals (see `PREREGISTRATION.md` §1, §10).
