# Data: from the synthetic stand-in to the real public-corpus run

No em dashes. All real data is public; none is committed. See `DATA_MANIFEST.json` for provenance.

## What ships in the repo

- `src/make_synthetic.py` generates a synthetic stand-in with a known ground truth. It is a
  pipeline-validation harness, not a result. It is deterministic given a seed.
- No raw third-party data is committed (`.gitignore` enforces this). Only the manifest is tracked.

## The two-line change to the real run

The analysis is encoder-agnostic and loader-agnostic. To run on real public data:

1. Replace the data source. Instead of `generate(...)`, load a public outcome corpus into the same
   structure: a list of `rows` (each a dict of the six surface content features plus, for the encoder
   path, the raw text), and a `behavior` array of measured outcomes. Suggested public corpora:
   - ChangeMyView / Winning Arguments (Tan et al. 2016): argument text with delta labels (belief change).
   - Persuasion-for-Good (Wang et al. 2019): donation-dialogue text with giving outcomes.
   - MIST (Maertens et al. 2023): validated misinformation-susceptibility norms for the discernment
     indicator.
   Each is downloaded by a small fetch script that writes a checksum into `DATA_MANIFEST.json`.

2. Replace the encoder. Instead of `get_encoder("reference")`, register and select a learned encoder
   (an open foundation text-to-activation model such as TRIBE v2, or the author's own encoders) that maps
   text to the four ROI activations the signature reads. Implement the projection in
   `src/encoders.py::TribeV2Encoder._predict_rois` and add it to `get_encoder`.

Everything else (the signature, the latent fusion, H-load / H-incremental / H-encoder / H-bound, the
kill criteria, the conformal layer, the positive-and-negative-control tests) is unchanged.

## Surface features

The six surface content features (emotional_intensity, evidence, identity_appeal, framing, concession,
social_proof) are computed from text by a feature extractor. In the stand-in they are sampled directly.
For the real run, compute them from the text with a documented, deterministic extractor (lexical and
discourse features), so the content-only baseline is reproducible and is a fair, strong comparison.

## Ethics

No new human-subjects data is collected. All outcome labels come from existing public, consented or
public-domain datasets. The work models content-level and population-level effects and does not target,
profile, or optimize against any named individual.
