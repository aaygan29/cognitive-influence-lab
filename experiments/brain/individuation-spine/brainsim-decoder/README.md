# BrainSim Decoder

An interactive Colab tool for fMRI-to-image decoding using real 7T neuroimaging data.

## What it does

Given an input image, the pipeline:

1. Encodes it into a CLIP ViT-L/14 embedding.
2. Predicts the whole-cortex fMRI response (~39k vertices, both hemispheres) via ridge regression trained on real trials.
3. Decodes the predicted fMRI back into CLIP space.
4. Renders an approximate retinotopic visual-field map of predicted activation.
5. Retrieves the top-5 most similar training images by decoded CLIP similarity.
6. Generates a caption from the decoded representation (BLIP).
7. Synthesizes an audio chord from the decoded embedding.

Optionally, GPU image reconstruction via Kandinsky 2.2 unCLIP (needs an A100/T4).

## Data & grounding

- Natural Scenes Dataset (NSD), subject 01, in Algonauts 2023 Challenge format (7T fMRI).
- NSD: https://naturalscenesdataset.org/ · Algonauts 2023: http://algonauts.csail.mit.edu/
- Stimulus features: CLIP ViT-L/14; captioning: BLIP; reconstruction: Kandinsky 2.2 unCLIP.

## License

MIT — see [LICENSE](LICENSE).
