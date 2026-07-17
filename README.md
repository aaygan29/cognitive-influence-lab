# digital_brain

Subject-specific encoding models that map visual stimuli to predicted 7T fMRI responses (a "digital brain"), with a five-level protocol for testing how individual-specific those predictions really are.

## What it does

- Trains a per-subject model that maps CLIP ViT-L/14 image features to that subject's cortical fMRI responses.
- Evaluates it across five levels: encoding accuracy, self-vs-other representational geometry, RSA identity structure, subject identification (permutation test), and counterfactual consistency on held-out stimuli.
- Reports the honest full-sample finding: at N=8 (all subjects), encoding is strong (Pearson r = 0.14 to 0.39, positive in 25/25 ROIs) but **subject identification is at chance (0/25 ROIs)** under proper controls (dimension-independent RSA identity, model-free split-half, shuffle, exact permutation). Encoding accuracy is not representational identity.

An earlier N=4 pilot suggested subject "fingerprinting"; that did not survive the full-N re-run and is kept only as a superseded results file. A separate synthetic-data pilot is archived and must not be cited.

## Data & grounding

- Algonauts 2023 Challenge (Natural Scenes Dataset subset): 7T fMRI, all 8 subjects, 260 shared images, 25 ROIs. Download: https://algonautsproject.com/2023
- Stimulus features: CLIP ViT-L/14. Geometry evaluation via representational similarity analysis (RSA).

## License

MIT — see [LICENSE](LICENSE).
