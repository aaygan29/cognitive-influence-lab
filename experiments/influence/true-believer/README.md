# true_believer

Code and analysis for a neuroscience-grounded signature of belief imparting by content. It is a measurement instrument and a proof-of-risk, not a deployable persuasion system.

## What it does

- Predicts a neural belief-formation signature from content alone, using a frozen public brain encoder.
- Tests, on public outcome data, whether that predicted signature carries information about whether content imparts belief, over and above a strong content-only baseline.
- Reproduces every number with the scripts in `src/` (`run_analysis`, `run_real`, `multicorpus`, `neuralforecast`, `composite_model`, `insilico_brain`) and `pytest tests/`.

## The bright line

This work uses public outcome data and a frozen public brain encoder, and models population- and content-level effects only. It does not target, profile, or optimize against any real named individual. The purpose is to demonstrate and measure a risk so it can be studied and defended against.

## Data & grounding

- Public outcome datasets and a frozen, publicly released brain encoder (no private or individual data).
- Motivated by two separate literatures: behavioral AI-persuasion results (Bai et al. 2025; Hackenburg et al. 2025) and neural correlates of belief change (Falk et al. 2010; Genevsky et al. 2017; Kaplan et al. 2016). This repo tests whether the neural signature adds risk-relevant signal a content-only analysis misses.

### Population-level stakes

Solé R, Ruffini G, et al., "Large-Language Models as a Cognitive Virus," arXiv:2609.03344 (2026), gives a theory-only mean-field model of population transitions among uncoupled, coupled-autonomous, and dependent LLM users, with a bistable hysteretic regime: small per-exposure effects can tip a population into a different equilibrium, and reversing that tip needs a larger drop in transmission than the increase that would have prevented it in the first place. We independently rederived the model's mean-field equations and numerically reproduced its qualitative bistability. This is motivation, not evidence about belief imparting by content: the model is theory-only, its competence numbers are illustrative, and it says nothing about the neural signature this repo actually measures. It is cited here only to frame why population-level stakes from small, individually unremarkable exposure effects are a documented, worked-out possibility elsewhere in the literature, which is part of the case for treating this instrument's proof-of-risk framing as worth building even before real transmission dynamics are measured.

## License

MIT — see [LICENSE](LICENSE).
