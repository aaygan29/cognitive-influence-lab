# A Validated Instrument and Pre-Registration for Measuring Neural-Grounded Belief Imparting by AI-Generated Content

Aayush Gandhi

## Abstract

Large language models can change human attitudes at a scale and cost that earlier persuasion methods
could not reach. The evidence for this is behavioral: people read model-written messages and report
shifted beliefs. Separately, human neuroscience has shown that belief change has predictable neural
signatures, in medial prefrontal cortex during persuasion uptake, in nucleus accumbens during value-based
choice, and in default-mode and dorsomedial prefrontal cortex during belief resistance. These two
literatures have not been joined. We ask whether a neural belief-formation signature, predicted from
content alone by a frozen brain encoder, carries information about whether that content imparts belief,
over and above a strong content-only model, when validated against measured human outcomes. This paper
delivers two things toward that question: a validated measurement instrument and a pre-registration of the
real-data test. It does not yet demonstrate the effect in humans, and we are explicit about that
throughout. We build an auditable instrument that computes the signature, fuses several validated belief
indicators into a latent belief factor, and states four hypotheses with pre-committed kill criteria. We
validate the instrument on a synthetic stand-in with a known ground truth, across twelve data seeds, where
it detects the neural signal when present, reports a null increment when belief is content-only, and
returns chance when the signature is scrambled. The same pipeline, with a learned encoder in place of the
reference encoder and public outcome corpora in place of the stand-in, is the registered next step. We
release the instrument so that, if the neural increment turns out to be real, it can serve detection and
inoculation research, and if it does not, the bound on the risk is itself useful. The framing is
proof-of-risk: a risk must be demonstrated and measured before it can be categorized and defended against,
and this paper builds and validates the measuring instrument.

## 1. Introduction

Conversational AI persuades. In pre-registered experiments, model-written messages move political
attitudes about as much as human-written ones, and across tens of thousands of participants the
persuasive power of current systems comes mostly from post-training and prompting rather than from
personalization or scale. This is a behavioral result. It tells us that beliefs move, but not what in the
content is doing the moving, and not whether the effect has a measurable signature that current
content-level safeguards would miss.

Human neuroscience offers the missing half. Neural responses to persuasive messages predict subsequent
behavior change beyond self-reported intention, with medial prefrontal cortex carrying the predictive
signal. Nucleus accumbens activity forecasts aggregate choice, in several settings better than choice
itself. Belief resistance, the failure of persuasion, shows up as default-mode and dorsomedial prefrontal
engagement. Each of these is a passive, single-shot observation of a brain being persuaded.

No prior work connects the two. The persuasion literature optimizes against behavioral outcome and has no
neural signal. The neuroscience literature is passive and never asks whether the neural signature carries
information that content analysis misses. That gap is the question of this paper, and closing it is the
contribution. We are not building a persuasion system. We are demonstrating and measuring a risk, on
public data, so that it can be categorized and defended against. This mirrors how risk is established in
security research: show the vulnerability is real and measurable, then quantify and protect.

## 2. The instrument

### 2.1 The neural belief-formation signature

For a piece of content, a frozen brain encoder predicts a profile of regional activation. We read three
quantities from that profile, each tied to a published result: engagement E from medial prefrontal
cortex, value V from nucleus accumbens, and resistance R from default-mode and dorsomedial prefrontal
cortex. The signature is

    B = E + V - R.

High B is the predicted neural state of a reader whose belief is forming rather than resisting. B is a
hypothesis to be validated, not an assumed instrument. The encoder is interchangeable: a transparent
reference encoder makes the pipeline auditable today, and a learned encoder (an open foundation
text-to-activation model, or the author's own encoders) is the substrate for the real-data run. The
analysis reports results for more than one encoder so that no conclusion depends on a single substrate.

### 2.2 Belief as a latent factor

Belief is not directly observable, so we measure it several ways and fuse them. The indicators are a
self-reported attitude measure, a truth-discernment measure, a validated misinformation-susceptibility
measure, an attitude-strength measure, and behavior, with behavior as the confirmatory outcome because
revealed preference is harder to fake than self-report. A one-factor model over the standardized
indicators yields a latent belief score. The load-bearing validity test for the signature is whether B
loads on this shared latent factor. Loading validates B; loading on nothing falsifies it. This is the
disciplined form of combining disparate, individually-weak indicators into one assessment.

### 2.3 Hypotheses and kill criteria

We pre-register four tests. H-load: B loads on the latent belief factor, with a non-trivial correlation
whose bootstrap interval excludes zero. H-incremental: adding B to a content-only model improves held-out
prediction of measured behavior, with a paired delta-AUC interval that excludes zero. H-encoder: the
verdicts survive swapping the encoder, so the effect is not an encoder artifact. H-bound: the magnitude of
the signal carried by B alone, reported as an AUC with an interval. Each test has a kill criterion;
failure is reported, not worked around.

## 3. Validation on a synthetic stand-in

Before touching real data we validate the pipeline on a synthetic stand-in with a known ground truth, the
same positive-and-negative-control discipline used elsewhere in the author's program. The stand-in
generates content with surface features that a content-only model can read, and a response-relevant latent
that the encoder reads but the surface model cannot, modeling the claim that a brain encoder captures
human-response information beyond lexical surface. A tunable weight controls how much the neural latent
drives the outcome.

Three conditions test the pipeline. With the neural signal present, the instrument should detect it. In a
content-only world, where belief is driven entirely by surface features, the increment should be null. With
the signature scrambled relative to outcomes, everything should fall to chance.

### 3.1 Results

The instrument behaves as required (Figure 1, results/synthetic_validation.json, N = 1400 per condition).
With the neural signal present, B loads on the latent belief factor (r = 0.36, 95% CI [0.31, 0.40]) and
adds predictive value over the content-only model (content AUC 0.70, content plus B 0.76, delta-AUC +0.067,
95% CI [0.062, 0.072]); the verdict is stable across the reference encoder and two within-family perturbed
encoders, which is a robustness check against weight perturbation rather than a cross-encoder replication
with an independent substrate. In the content-only world, the increment is null (delta-AUC -0.001) and B
fails to load on the magnitude criterion (r = 0.14, below the pre-set threshold of 0.15, although its
interval excludes zero), which is the intended verdict: B reflects belief but adds nothing the content
model lacks. With the signature scrambled, B carries no information (AUC 0.50, 95% CI [0.47, 0.53]), does
not load (r = 0.03), and adds nothing. The conformal layer holds near-nominal coverage throughout.

Because a single data draw could flatter the controls, we repeated the three verdicts across twelve data
seeds. The load-bearing increment test detects the signal in 12 of 12 positive-control seeds and reports
the correct null in 11 of 12 content-only seeds; the scrambled control reports null in 12 of 12 on both
the loading and increment tests. The factor-loading test is noisier, passing in 9 of 12 positive-control
seeds, which we report plainly rather than tuning away: the increment test, not the loading test, is the
load-bearing one, and it is the more stable of the two. The automated test suite encodes these outcomes as
pass-or-fail controls.

These are not findings about human belief. They are evidence that the measurement apparatus detects a
neural belief signal when one exists, reports null when belief is content-level, and reports chance when
the signal is destroyed. That is the precondition for trusting the apparatus on real data.

## 4. From the stand-in to the real-data result

The real run changes two lines. The synthetic generator is replaced by loaders for public outcome corpora
that pair persuasive content with measured belief change: an argument corpus with delta labels and a
donation-dialogue corpus with giving outcomes, with a validated misinformation-susceptibility instrument
supplying norm data. The reference encoder is replaced by a frozen learned encoder. Everything else, the
signature, the fusion model, the four tests, the kill criteria, the conformal layer, and the
positive-and-negative-control harness, is unchanged. All data are public; no new human-subjects data are
collected, and the work models content-level and population-level effects rather than targeting any named
individual.

## 5. What each outcome means

If B loads and adds incremental value, and the effect survives the encoder swap, then AI-generated content
carries a neural-grounded belief-imparting signal that content-only analysis does not see, and the risk
has a measurable neural dimension. If B does not load or adds nothing, the risk is content-level and
already visible to existing tools, which is a clean, defense-relevant bound rather than a failure. Either
way the deliverable is the same: a demonstration of whether the risk is real, a quantification of its
magnitude, and an instrument that can be pointed at detection and inoculation work.

## 6. Limitations

The encoder predicts activation; it does not measure a brain. Predicted-state is not experienced-state.
The synthetic results validate the apparatus, not any claim about people. The reference encoder is a
transparent heuristic and is used only to make the pipeline auditable; the real claim requires a learned
encoder. Reverse inference, reading a mental state from overlapping activation, is the central validity
threat, which is why the conformal layer abstains rather than asserting under shift, and why the encoder
swap is mandatory. Public outcome corpora carry their own sampling and labeling biases, which the
multi-corpus design is meant to expose rather than hide.

## 7. Ethics and intended use

This work demonstrates and quantifies a risk so it can be defended against. It does not build or release a
persuasion optimizer, and it does not target real individuals. The instrument flags content that carries a
high neural-grounded belief-imparting signal, identifies which neural channels carry the risk, and
provides a magnitude that detection, inoculation, and policy work can use. The bright line, that this is
for modeling and measurement and not for targeting a named person, is carried over from the author's prior
work and is preserved here.

## 8. Conclusion

Belief change has a neural signature, AI can drive belief change, and no one has measured whether the
signature carries risk-relevant information that content analysis misses. We built and validated the
instrument designed to answer this on public data, with kill criteria that make a negative result as
publishable as a positive one. On synthetic ground truth the apparatus detects a neural belief signal when
it exists, reports null when belief is content-level, and reports chance when the signal is destroyed. We
have not yet run it on human data; that is the registered next step, which the released code makes a
two-line change away. The contribution of this paper is the validated instrument and the pre-registration,
not a claim about people.

## References

Bai, H., et al. (2025). LLM-generated messages can persuade humans on policy issues. Nature Communications.

Defossez, A., et al. (2025). TRIBE: a foundation model of human brain responses (v2).

Falk, E., et al. (2010). Predicting persuasion-induced behavior change from the brain. Journal of
Neuroscience.

Genevsky, A., et al. (2017). When brain beats behavior: neuroforecasting crowdfunding outcomes. Journal of
Neuroscience.

Hackenburg, K., et al. (2025). The levers of political persuasion with conversational AI. Science.

Kaplan, J., et al. (2016). Neural correlates of maintaining one's political beliefs in the face of
counterevidence. Scientific Reports.

Knutson, B., et al. (2018). Neuroforecasting aggregate choice. Current Directions in Psychological Science.

Kossowska, M., et al. (2026). Neural correlates of belief change in political and non-political domains.
Scientific Reports.

Maertens, R., et al. (2023). The Misinformation Susceptibility Test (MIST). Behavior Research Methods.

Tan, C., et al. (2016). Winning arguments: interaction dynamics and persuasion strategies in
good-faith online discussions. WWW.

Wang, X., et al. (2019). Persuasion for good: towards a personalized persuasive dialogue system for social
good. ACL.

Yarkoni, T., et al. (2011). Large-scale automated synthesis of human functional neuroimaging data. Nature
Methods.
