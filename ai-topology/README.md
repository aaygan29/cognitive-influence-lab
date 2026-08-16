# neuro_topology

Tests whether making a transformer's attention graph more "brain-like" actually improves language modeling, or whether brain-like structure is just a byproduct of training.

## What it does

- Trains small language models (nanoGPT on TinyShakespeare) under a baseline and under small-world topology regularization, across 20 seeds and multiple topology conditions (120 training runs total).
- Measures the effect with Welch t-tests, Cohen's d, and bootstrap confidence intervals.
- Ablates the top- and bottom-10 "brain-similar" attention heads in GPT-2 to measure how load-bearing they are.

**Finding:** brain-like topology in early layers is real but emergent, not a performance driver. Regularizing toward it *hurts* performance (Cohen's d = -1.20 across 20 seeds), and brain-like heads are ~8.6x less load-bearing than the heads doing the actual work. The original single-run "improvement" was seed variance.

## Data & grounding

- TinyShakespeare corpus; nanoGPT (~0.81M params) and pretrained GPT-2 attention.
- Motivated by the hypothesis that better language models converge toward brain-like network structure; this repo tests the causal version of that claim.

## License

MIT — see [LICENSE](LICENSE).
