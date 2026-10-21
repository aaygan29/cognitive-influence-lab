# The locus coeruleus as a handle on learning and decision-making

Anonymous. 2026-10-20. An evidence node, not an experiment: nothing here was run in this tree.
Every citation below is in [`corpus/references.json`](../../corpus/references.json) with a resolvable
identifier, the URL it was fetched from, and the date of that fetch. Only entries marked `verified`
support a claim.

## Why this node exists

The influence arm of this tree already leans on the locus coeruleus. `bounded-influence` uses
noradrenaline as the physiological knob that sets the gain of its belief dynamics and therefore the
height of its barrier, and proposes an arousal proxy as the specificity control for its
critical-slowing-down monitor. That was cited to a preprint and listed in that node's own
`PROVENANCE.md` as **not re-verified**. If the LC is going to carry weight in a threat model, the
weight has to be checked.

It is also the obvious candidate for the other direction. The LC is the brain's main noradrenergic
nucleus; it is the earliest site of tau pathology in Alzheimer's disease; there is a widely repeated
story that stimulating it — non-invasively, through the ear — helps people with Alzheimer's retain
memory and learn better. If that story held, it would be the first real human instance of the thing
this tree otherwise only simulates: a cheap, deployable knob on an individual's learning.

**It does not hold, in the form it is usually told.** This node records what is actually there.

## What is established

- **Pretangle tau appears in the LC very early, often before cortical tau, often in young adults.**
  Two large autopsy series (`braak2011-stages`, N=2,332; `braak2011-under30`, N=42) and independent
  stereology from a different group (`ehrenberg2017-lc-tau`; `theofilas2017-lc-volume`). About as
  solid as human neuropathology gets. Note it is cross-sectional: nobody has watched tau travel in a
  living human, and Braak's own inference is explicitly conditional on the pretangle state not being
  transient.
- **The LC loses many neurons in AD**, meta-analytically (`lyness2003-meta`, d=2.28 over 24 studies).
  But the same meta-analysis puts **nucleus basalis higher** (d=2.48). The common claim that the LC
  is the single hardest-hit nucleus is not what the meta-analysis says.
- **LC neuron density at autopsy correlates with antemortem cognitive decline**, replicated across
  cohorts (`wilson2013-neural-reserve`, N=165; `kelly2017-lc-progression`; `jacobs2021-lc-validation`).
  The foundational estimate is very small (0.003, SE 0.001), and the widely quoted "30% LC loss by
  MCI" rests on 11 versus 10 brains.
- **Removing LC noradrenaline worsens pathology and cognition in AD animal models**, across labs,
  species, toxins and chemogenetics (`heneka2010-lc-microglia`, `chalermpalanupap2018-dsp4`,
  `rorabaugh2017-chemogenetic`, `evans2024-ne-inhibition`).

## What is not established, and is usually claimed anyway

### 1. That transcutaneous vagus nerve stimulation engages the LC

This is the load-bearing claim for every "non-invasive LC stimulation" proposal, and the human
mechanistic evidence for it is **one N=12 brainstem fMRI study** (`frangos2015-tavns-brainstem`).

The markers that claim predicts are null:

| Marker | Result |
|---|---|
| Pupil, continuous/conventional protocol | **g = 0.002, BF01 = 21.9** — strong evidence *for the null*, 18 studies, 771 participants (`pervaz2025-pupil-meta`) |
| Pupil, independent single studies | null in `keute2019-null-pupil` (N=33), `warren2019-tavns-markers` (N=16), `burger2020-null-pupil` (3 studies), `contier2026-null` (N=40) |
| P300 | null across three labs; Bayes factor favouring the null 4.53 (`warren2019-tavns-markers`, `keute2019-null-pupil`, `contier2026-null`) |
| Salivary alpha-amylase | moves — but only under mixed models on pooled raw data, not under the summary-statistic meta-analysis (`giraudier2022-saa-mega`), and it is a *peripheral sympathetic* measure, not a central one |

`contier2026-null` is the sharpest of these: robust P3 and P600 effects were present in the data, so
the paradigm worked; taVNS moved neither, nor pupil, nor alpha-amylase. The authors conclude the
protocol may not have engaged LC-NE at all.

The one real moderator is **protocol**: pulsed/phasic taVNS *does* dilate the pupil (g = 0.36,
BF10 = 50.8, same meta-analysis; `sharon2021-phasic-pupil`). But `ludwig2024-perceived-intensity`
shows that effect attenuates once *perceived* stimulation intensity is controlled, and scales with
perceived intensity — so arousal-from-sensation is not excluded even there.

### 2. That taVNS improves cognition at the effect size claimed

The behavioural meta-analysis reports **g = 0.41** (`liu2026-cognition-meta`, 30 quantitative
studies, >1,500 participants) — and reports **no publication-bias test**. A moderate pooled
behavioural effect sitting on top of a mechanism literature that is null, in a field of small
single-blind studies, is the textbook small-study-effect signature. Treat g = 0.41 as an upper
bound.

The internal check is the damning one: `venturabort2025-memory` **replicated its own ERP findings and
failed to replicate its own behavioural memory effect**. The electrophysiology came back; the
behaviour did not.

### 3. That any of this helps people with Alzheimer's

**There is not one completed, double-masked taVNS trial in MCI or AD with posted positive results.**

- `nct03359902-mci-tavns` — the only one that is completed, double-masked *and* has results posted:
  N=59, randomized crossover. Primary outcome, Rey Auditory Verbal Learning Test delayed recall:
  sham 6.43 vs tVNS 7.83, **β = 1.4, p = 0.11. The prespecified primary endpoint was missed.**
- `nct04908358-ad-tavns` — the largest (N=126, MGH), completed, **no results posted**. Its outcome is
  unknown and must not be described either way.
- The remaining seven registered trials are recruiting or not yet started.

The one noradrenaline-augmentation trial in this indication, `levey2022-atomoxetine-mci` (N=39,
double-blind crossover, 12 months), **achieved target engagement, failed both prespecified primary
outcomes, and showed no significant effect on cognition or clinical outcomes.** It did reduce CSF tau
and pTau181 — as secondary/exploratory findings.

### 4. That more LC drive is good

The direction of effect is not fixed. `rorabaugh2017-chemogenetic` found **short-term** chemogenetic
LC activation rescued reversal learning in TgF344-AD rats. `nong2026-chronic-stim-worse` found
**long-term** LC stimulation *worsened* spatial memory, impaired contextual fear memory, reduced
hippocampal LTP and increased tau hyperphosphorylation in PS19 mice, and explicitly contrasts itself
with the short-term result.

Duration flips the sign. Any proposal to use LC drive as a lever has to specify duration and dose
before it can claim a direction at all.

Note also that the cognitive-reserve framing (`robertson2013-reserve-theory`,
`mather2016-lc-review`) has been substantially walked back **by its own leading proponent**:
`mather2021-retreat` states that noradrenaline "sometimes prevents and sometimes accelerates"
amyloid and tau, and that mental stimulation "may also have risks as well as benefits". Anyone citing
the 2016 review for a protective claim must cite the 2021 paper beside it.

## The measurement problem, which is worse than the effect problem

Most human LC work in aging rests on neuromelanin-sensitive MRI contrast. That measure is in trouble
on four independent fronts:

- **The validation is n=7 with no correlational test.** `keren2015-histologic-validation`, the study
  the field cites as its histological grounding, says so itself.
- **The name may be wrong.** `watanabe2019-not-neuromelanin` found the signal in 3-week-old mice and
  in dopamine-β-hydroxylase knockouts, in a species that largely lacks neuromelanin;
  `trujillo2017-iron-contrast` showed metal-free melanin has negligible magnetization-transfer
  effect. The contrast is substantially a **melanin-iron T1** effect, and possibly an **activity**
  marker rather than a pigment-accumulation one.
- **The field cannot agree where the LC is.** Seven published atlases overlap at **Dice 0.2–0.4**
  (`dmitrichenko2026-atlas-disagreement`); two leading masks overlap 51% after a published correction
  of a coding error (`liu2020-corrigendum`); inter-rater Dice is 0.54–0.64 and contrast ICC 0.53
  (`tona2017-testretest`); functional "LC" coordinates are scattered relative to neuroanatomy
  (`liu2017-systematic-review`).
- **It moves on a five-week timescale.** `bachman2023-hrv-lc-contrast`: slow paced breathing
  *reduced* left LC contrast in younger participants, proportional to how much they increased
  heart-rate oscillations. If a breathing exercise moves "LC structural integrity", the measure is
  not purely structural.

Add that LC contrast versus age is **quadratic, peaking around 60** (`liu2019-camcan-quadratic`,
N=605) and LC volume follows an inverted U (`dmitrichenko2026-atlas-disagreement`, N=2,393), and the
reading of LC contrast as cumulative neuron loss does not survive.

The field's own 2019 consensus (`betts2019-consensus`) framed LC imaging entirely prospectively —
"could help", "holds the promise". Seven years later the harmonisation steps it called for have not
been met.

## What this means for this tree

**For `bounded-influence`.** Its use of noradrenaline as a gain-and-noise knob is now cited to a
verified source: `su2026-lc-topography` (Nature 2026), the published version of the preprint that
node cited, whose abstract matches that node's description closely — dorsal LC neurons projecting to
isocortex activate on choice switches and carry reward-prediction-error signals; ventral LC activity
is higher when animals ignore reward-predicting stimuli. **It is a mouse result.** It licenses LC-NE
as a candidate mechanism for the model's gain parameter. It does not license a human magnitude.

**For the arousal covariate.** `bounded-influence` proposes pupil as the arousal proxy that
discriminates a genuine approach to the separatrix from an arousal shift. That control is weaker than
it looks. `joshi2016-pupil-lc` — the paper usually cited to license pupil-as-LC — **concludes the
relationship is not specific to LC**: colliculi and cingulate show it too. `megemont2022-pupil-not-lc`
shows pupil predicts only a small fraction of LC spiking moment to moment, with large variability
under *identical* optogenetic drive. And `nuiten2026-arousal-dissociation` shows the pharmacological
and pupillometric handles dissociate in humans — they are not two windows on one variable. A pupil
covariate controls for *something*, but not specifically for LC gain.

**For the steering arm generally.** The best-supported LC-NE-to-cognition link in this record is not
stimulation at all: it is that **noradrenaline reduces the weight of priors in decision-making**
(`degee2025-choice-history`, `marshall2016-pharma-fingerprints` at N=128,
`ocallaghan2025-parkinsons-rl`). That is a real, pharmacologically-manipulable decision parameter in
humans, and it is closer to this tree's concerns than anything in the taVNS literature. It is also
the closest human analogue to what the simulated manipulations in `bellwether` and
`persuasion-control` do.

Note the cautionary case for this tree's own theory habits: adaptive gain theory
(`astonjones2005-adaptive-gain`) is the standard framework for LC and exploration, and its one direct
human pharmacological test found the effect **in the opposite direction to prediction**
(`warren2017-atomoxetine-exploration`: atomoxetine *reduced* random exploration, N=22). The
inverted-U rescue the authors offer is unfalsifiable until someone runs a dose-response design.

## What would change this assessment

- A preregistered, adequately powered (N ≥ 150) continuous-taVNS study showing reliable pupil or P300
  modulation against a perceived-intensity-matched sham. It would have to beat BF01 = 21.9.
- Posted results from `nct04908358-ad-tavns` (N=126), or any adequately powered positive MCI/AD trial.
- A funnel/Egger/selection-model analysis of `liu2026-cognition-meta`'s 30 studies showing no
  small-study effect.
- An ex-vivo study with n ≫ 7 showing LC-MRI contrast correlates with stereological neuron counts in
  the same tissue, plus multi-site test-retest ICC > 0.8 on a harmonised mask.
- Independent replication of `nong2026-chronic-stim-worse` in a second tau model, which would make
  chronic LC stimulation therapeutically untenable as currently conceived.

The single highest-value study the field is missing: one adequately powered, preregistered,
double-blind design measuring a mechanism marker **and** a behavioural outcome **in the same
participants**, under both protocols. The contradiction has survived precisely because mechanism
studies and behaviour studies are different studies.

## Recorded as unverified

Three claims were searched for and could not be confirmed. They are in
[`corpus/references.json`](../../corpus/references.json) with `status: unverified` and the reason, so
that they are not quietly re-added by someone who half-remembers them:

- `clark1999-dose-numbers` — the paper is real and verified, but the famous inverted-U figures
  attached to it ("~35% at 0.5 mA, p<0.025") are **not in the abstract** and the full text was not
  read. Do not quote them.
- `optogenetic-lc-ad-rescue` — the AD-model causal rescue result is **chemogenetic**, not
  optogenetic. A query combining optogenetic, locus coeruleus and Alzheimer returned zero hits.
- `lc-transplant-ad-models` — LC grafting into normal rat hippocampus is real (1990); LC
  transplantation as an AD-model intervention could not be confirmed to exist.
- `alpha2-agonist-decision-studies` — no verifiable human clonidine or guanfacine study on
  exploration, learning rate or drift-diffusion parameters was found. Every noradrenergic
  pharmacology result verified here uses atomoxetine or propranolol.

`thompson2026-pupil-lc-modes` is a **preprint**: directly on point, pointing the same way as
`megemont2022-pupil-not-lc`, and marked as unreviewed at every use.
