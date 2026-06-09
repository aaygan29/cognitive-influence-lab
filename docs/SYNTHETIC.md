# Synthetic respondents — calibration gate (protocol)

LLM agents may be used as **additional** respondents, **never** pooled into the confirmatory
human estimate (`PREREGISTRATION.md` §6, §9). Admission requires passing a pre-set calibration
gate against human data:

- On overlapping choices13k problems, the LLM population's choice-distortion **direction** must
  match the human direction, and its **magnitude** must fall within a pre-registered tolerance
  band (numeric band to be fixed before any synthetic run).
- Substrate (human vs. LLM) is always an explicit covariate; synthetic results are reported as a
  separate, clearly-labeled arm.
- If the gate fails, synthetic data are excluded from confirmatory analysis and the failure is
  reported as a negative methods result.

Method and documented limits follow Horton (2023); Aher, Arriaga, & Kalai (2023);
Argyle et al. (2023); Binz & Schulz (2023).

**Status:** protocol stub; numeric tolerances to be set before registration.
