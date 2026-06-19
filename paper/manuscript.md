# Wiring Is Not Weights: Identity as the Functional-Equivalence Class, and What It Would Take to Store It

**Aayush Gandhi**
Preprint draft, June 2026. Code: https://github.com/aaygan29/wiring-not-weights

---

## Abstract

Whole-brain preservation and "mind uploading" rest on an unexamined premise: that capturing a brain's
structure captures the person. We argue this premise is true for one sense of identity and false for another,
and we make the distinction measurable. We separate identity into three senses: *identification* (telling an
individual apart), *reconstruction* (regenerating the individual's stimulus-response function), and
*constitution* (whether a copy is the same person). We show, on synthetic ground truth with a validated
instrument, that reconstruction-identity is carried by the **functional-equivalence class of a network's
weights** rather than by its wiring diagram or by the exact weight values: a network's weights can be changed
by more than their own magnitude with zero loss of identity if the function is preserved (an exact nonlinear
symmetry), while an equal-magnitude change that alters the function destroys identity (Cohen's *d* = 33). We
quantify the *identity-sufficiency curve* — how much of the individuating weight-function must be retained to
recover the individual — and find identity is high-dimensional and distributed. On real human connectomes
(ABIDE, N = 248), we confirm that *identification* is a topology property: a binarized connectome fingerprints
individuals nearly as well as the weighted one (0.972 vs 0.980). A real-data *reconstruction* test (NSD, N = 8)
is underpowered and null, reproducing prior work. We connect these results to the connectome-constrained
network, degeneracy, and bits-per-synapse literatures to specify what storing identity would require: not the
wiring (non-identifying of function) and not the exact weights (degenerate), but the connectome plus enough
functional sampling to pin the correct equivalence class — on the order of 10^14–10^15 bits for the functional
content. We frame the single decisive experiment that would establish the dissociation in real human brains.

---

## 1. Introduction

If you copied a person's connectome perfectly, would you have the person? The connectomics, brain-preservation
(e.g., aldehyde-stabilized cryopreservation), and mind-uploading communities largely assume the answer is yes:
preserve the structure in enough detail and the person is, in principle, recoverable. We argue the question is
ill-posed because "identity" denotes three different things, and the structural premise is true for one and
unestablished for the others.

- **Identification (indexical):** can an individual be told apart from all others? Functional connectomes do
  this robustly [Finn2015].
- **Reconstruction (functional):** does the structure regenerate the individual's responses, dynamics, and
  behavior?
- **Constitution (the hard problem):** would instantiating the structure *be* the person — preserving the
  first-person and its continuity? [Chalmers2014, Parfit1984]

A fingerprint identifies you without reconstituting you. Our thesis concerns reconstruction:

> The connectome is the architecture; reconstruction-identity is the **wiring-scaffolded weight-function, up
> to its functional-degeneracy class.** The wiring alone does not determine the function; the exact weights are
> not the right unit (many weight sets implement the same function); the target is the functional-equivalence
> class.

This is not a new biological discovery — it is established at the circuit level that connectivity does not
determine function (Section 2). Our contributions are: (i) a three-sense operationalization that dissolves the
structural premise; (ii) a validated in-silico instrument — a capacity-matched, SNR-controlled, degeneracy-
aware ablation that measures an *identity-sufficiency curve*; (iii) a demonstration, including genuine
nonlinear compensatory degeneracy, that identity is the weights' functional-equivalence class; (iv) a real-
data result separating the identification axis (topology suffices) from the reconstruction axis (open); and
(v) a synthesis with information-theoretic and connectome-constrained-modeling results specifying what storing
identity would require.

## 2. Related work

**Connectivity does not determine function.** The *C. elegans* connectome has been complete since 1986, yet
behavior is not predictable from wiring without synaptic signs, weights, and neuromodulation [White1986,
Bargmann2012]. In real connectome-constrained recurrent networks, the connectome is insufficient to fix
dynamics: an *ensemble* of different weight solutions produces the same activity (non-identifiability), and
recordings from a subset of neurons are required to collapse the degeneracy [connectome-constrained2025,
Lappalainen2024]. This is the real-data form of our thesis.

**Degeneracy.** Disparate parameter sets produce near-identical circuit output: ~20 million model versions of
the crustacean pyloric network with widely varying conductances yield indistinguishable activity [Prinz2004].
Identity, if it is in the weights, must therefore be in their functional-equivalence class, not their values.

**Identification.** Functional connectomes fingerprint individuals at ~90%+ across sessions and tasks
[Finn2015]; this is the indexical sense, and it is a structural/topological property.

**Information content.** Synapses store ~4.7 bits each (26 distinguishable strengths) [Bartol2015], replicated
at 4.1–4.59 bits [Bartol2024]. Whole-brain-emulation accounting estimates ~50 bits/synapse to *address and
place* each synapse structurally [Sandberg2008]. Memory is moreover not purely synaptic: intrinsic excitability
and molecular/epigenetic state carry identity-relevant information [intrinsic2024, RNAengram2018].

**Preservation and uploading.** Structural brain preservation argues "memory is structural" from electro-
cerebral-silence evidence (deep hypothermic circulatory arrest): the brain can be electrically silenced and
recover, so live dynamics are not the bearer [DHCA]. We accept this — and note it does not bridge from
*pausable-and-resumable* living tissue to *fixed* tissue, nor from *traceable wiring* to *preserved function*.
Philosophical analyses frame what must be preserved as the substrate-independent functional pattern, leaving
constitution unresolved [Chalmers2014].

## 3. Framework and operational definitions

- **Wiring (architecture):** the connectivity graph at a stated resolution.
- **Weights:** the individuating parameters at the synaptic level and below (strengths, and the molecular state
  implementing them).
- **Functional-equivalence class:** the set of weight configurations producing indistinguishable individual-
  level function. All identity claims are about this class, never exact values.
- **Identity battery:** a multi-axis, SNR-controlled, conformally-calibrated instrument returning the
  probability that two model brains are the same individual. Axes: discriminability, reconstruction, dynamical
  signature, perturbation signature, self-model consistency. (This paper exercises the reconstruction axis.)
- **Identity-sufficiency representation:** the minimal retained weight-information that still passes the
  battery; by construction, the specification of what must be stored.

## 4. Methods

**The apparatus.** Individuals are models (linear maps in exp01–03a; ReLU MLPs in exp03b) sharing a population
backbone plus an individual weight-function. Twins are fit from training data. We construct capacity-matched
arms — identical parameter counts, differing only in whether and how individual weight-information is retained —
and identify individuals from held-out responses (Finn-style argmax matching; chance = 1/N).

**The ablation ladder.** We degrade the individual weight-function along a controlled axis and re-run the
battery: full weight-function -> within-manifold (function preserved) -> outside-manifold (function broken) ->
wiring-only. The break-point is the identity-sufficiency representation.

**Controls (each pre-registered).** (i) SNR/noise-ceiling matching; (ii) capacity matching across arms;
(iii) a single pre-committed axis to avoid multiple comparisons; (iv) degeneracy-aware perturbation (within-
vs outside-equivalence-class), never uniform randomization; (v) a label-shuffle null; (vi) replication across
R independent random cohorts and across SNR x signal regimes; (vii) bootstrap CIs and permutation p-values.

## 5. Results

**5.1 Apparatus validation (exp01–02).** The four-arm ordering FULL ~ WITHIN >> OUTSIDE ~ WIRING ~ chance
replicates in 20/20 random cohorts and in all 6 SNR x signal regimes, with the null control at chance and
equal parameter counts across arms. The instrument detects planted weight-function identity and reports its
absence under the null.

**5.2 The identity-sufficiency curve (exp03a, Fig 1).** In a sub-ceiling regime (FULL = 0.78, chance = 0.02,
N = 50, 15 cohorts), identification accuracy rises monotonically and convexly with the fraction of individuating
weight-dimensions retained; the minimal sufficient fraction is alpha* ~ 0.8. Identity is **distributed and
high-dimensional** — a few top features do not suffice. The exact-degeneracy control (null-space weight
perturbation) is perfectly flat across all perturbation magnitudes (range 0.000), while approximate-degeneracy
perturbation declines. Identity tracks the functional weight-dimensions and is invariant to weight changes
within the exact degeneracy class.

**5.3 Nonlinear compensatory degeneracy (exp03b, Fig 2).** Using ReLU-MLP twins and the exact ReLU symmetry
(hidden-unit permutation + positive rescaling), a PERM_SCALE arm changes the weights by ||ΔW||/||W|| = 1.57 —
more than their own magnitude — with **zero** change in identity (FULL = PERM_SCALE = 1.000). A FUNC_PERTURB
arm of the *same* weight-change magnitude, but in functional directions, drops identity to 0.243 (PERM_SCALE
vs FUNC_PERTURB Cohen's *d* = 33). Replacing the function (FUNC_RAND) or using the group function (WIRING)
gives chance. Identity is the function the weights compute, not the weight values; this holds under genuine
nonlinear, compensatory degeneracy, not only linear null spaces.

**5.4 Real connectomes, identification axis (exp05, Fig 3).** On ABIDE (Preprocessed Connectomes Project,
CC200 atlas, N = 248, public), within-subject split-half fingerprinting gives FULL_weighted = 0.980,
WIRING_binarized = 0.972, NODE_STRENGTH = 0.446 (chance = 0.004; all permutation p = 0.0005). **Binarized
topology fingerprints individuals almost as well as the weighted connectome.** Coarse topology (node strength)
is insufficient; the thresholded edge pattern is sufficient. This is the *identification* sense — the axis the
framework assigns to wiring — and it confirms wiring suffices there. Caveats: within-session split (inflates
vs cross-session); multi-site (scanner signature is a partial confound).

**5.5 Real reconstruction axis (exp04).** On NSD (N = 8, 260 shared images, 25 ROIs), neither the predicted-
representational-geometry fingerprint (mean acc 0.175) nor the encoder feature-tuning fingerprint (0.205)
individuates above chance (1/8 = 0.125; 0/25 and 1/25 ROIs significant, uncorrected). The degeneracy machinery
is verified (null-space weight perturbation changes predictions by 7e-15). This reproduces prior N = 8 nulls;
N = 8 is underpowered to resolve the reconstruction-level dissociation.

## 6. What it would take to store identity

The literature converges with our instrument on a concrete answer. The wiring alone is non-identifying of
function [connectome-constrained2025]; the exact weights are degenerate [Prinz2004] and unnecessary; the target
is the functional-equivalence class, fixed by the connectome **plus a sufficient sample of functional
activity** — and the theory even prioritizes which recordings most efficiently collapse the degeneracy
[connectome-constrained2025]. The storage budget brackets as: ~4.7 bits/synapse for the functional content
[Bartol2015] -> ~10^14–10^15 bits (~100 TB) for the weight-function; up to ~50 bits/synapse (~6 PB) for full
structural addressing [Sandberg2008]. "Weights" must include intrinsic-excitability and molecular state
[intrinsic2024, RNAengram2018]. A principled identity format therefore stores neither a wiring diagram (lossy)
nor exact weights (unmeasurable, unnecessary) but **sufficient statistics of the weight-function**: connectome
+ the minimal functional recordings that pin the equivalence class, at the resolution the sufficiency curve
demands.

## 7. Limitations and the decisive experiment

The strong dissociation is established **in silico** and corroborated by real-connectome modeling; it is not
yet shown on real human brains, where our identification result is powered (and confirms wiring suffices there)
but our reconstruction result is underpowered. The synthetic worlds are simplified (linear and small nonlinear
networks); the ABIDE result carries within-session and multi-site confounds; constitution is bracketed, not
addressed.

**The decisive experiment.** A powered *reconstruction* test: many subjects, each with rich stimulus-evoked
responses **and** a connectome, fitting connectome-conditioned individual encoders and running the capacity-
matched, degeneracy-controlled ablation. If a wiring-matched/weight-randomized twin fails reconstruction where
a within-equivalence-class twin succeeds, the dissociation holds in real human brains. The bottleneck is data:
such many-subject stimulus-response-plus-connectome datasets barely exist (NSD N=8, NeuroMod N=6). Building one
is the single highest-leverage step.

## 8. Conclusion

The connectome is the stage and the body of the actor; identity is the performance. You can copy the stage
perfectly and still not have the play — which is why "store the wiring" can never be "store the person."
Reconstruction-identity is the weights' functional-equivalence class; storing it requires the connectome plus
enough functional constraint to pin that class. We provide a validated instrument to measure exactly that, and
identify the one missing measurement that would settle it in humans.

## References

See `refs.bib`. Key anchors: Finn et al. 2015 (fingerprinting); Prinz, Bucher & Marder 2004 (degeneracy);
the connectome-constrained recurrent network non-identifiability result (Nat. Neurosci. 2025) and Lappalainen
et al. 2024 (fly visual system); Bartol & Sejnowski 2015 (bits per synapse); Sandberg & Bostrom 2008 (WBE
resolution ladder); Chalmers 2014 (uploading); intrinsic-excitability engram work (2024).
