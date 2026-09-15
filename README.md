# Structure Identifies, Function Individuates

**A power-controlled dissociation across brains and brain-inspired networks.**

Aayush Gandhi. This repository merges three previously separate repos
(`neurowire`/wiring-not-weights, `digital_brain`, `neuro_topology`) into the single evidence base
for one paper, the same way the underlying research program was consolidated from ~15 project
folders into one synthesized thesis. Each subdirectory keeps its full original commit history
(merged via `git subtree`); the old standalone repos are archived (not deleted) and point here.

## The thesis

> What makes a neural system *identifiable* is coarse structure, and it is cheap. What
> *individuates* it (its function, its behavior) is a different and harder quantity, and current
> human data cannot yet resolve how hard. The same lesson holds in artificial networks: brain-like
> *structure* is not the performance lever, function is. Bio-inspired AI should copy computation,
> not anatomy.

## The four claims, stress-tested (see `brainsim-decoder/       fMRI-to-image decoding on NSD 7T data (CLIP encoding, ridge whole-cortex
                       prediction, retrieval and captioning): computer vision applied to
                       individual neuroimaging, folded in 2026-09-15 with full history
rigor/STRENGTHENED_SYNTHESIS.md` for full detail)

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Identification is structural and cheap | **STRENGTHENED.** Corrected TOST equivalence (committed N=248: CI [-1.9%, +3.5%]); independently replicated fresh N=75 re-fetch; Marchenko-Pastur random-matrix null rejected (p≈0.005). Within-session confound named, not fixed (needs HCP between-session data). | `wiring-not-weights/` |
| 2 | Human reconstruction-individuation is **undetermined**, not "dear" | **Corrected.** digital-brain's N=4 "21/25 significant" is 0/25 after BH-FDR and Bonferroni; wiring-not-weights' N=8 null has MDES≈0.59 against an observed 0.175 — underpowered, not proof of a null. | `digital-brain/`, `wiring-not-weights/` |
| 3 | Where power is unlimited, reconstruction-identity is the functional-equivalence class of the weights | **STRENGTHENED existence proof.** Formalized via the Fisher Information Metric: the class is literally the FIM's zero-eigenvalue symmetry directions (permutation ⋊ positive-rescaling), confirmed numerically (curvature ≈1e-17). A stronger four-way numerical unification was attempted and **failed** — reported honestly, not hidden. | `wiring-not-weights/`, `rigor/fim_spectrum_results.json` |
| 4 | Brain-like structure is emergent, not causal, in artificial networks | **Survives unchanged.** Imposing brain-like attention topology on a transformer hurts performance, Cohen's d = -1.20, 120 runs / 20 seeds. The cleanest negative in the program. | `ai-topology/` |

## Layout

```
wiring-not-weights/   the spine: ABIDE N=248 identification + NSD N=8 reconstruction +
                       the synthetic functional-equivalence-class apparatus (full history)
digital-brain/         N=4 stimulus-evoked fingerprinting arm, now FDR-corrected (full history)
ai-topology/            the AI-side echo: brain-like topology hurts transformer performance (full history)
rigor/                  the stress-test pass: math framework (FIM/sloppy-model theory, Lie-group
                       quotient manifolds, manifold capacity, random matrix theory, rate-distortion),
                       corrected statistics, adversarial review, and a second-substrate (RNN)
                       generalization test of the framework
```

## Honesty discipline

Every negative result here is controlled (positive-and-negative apparatus checks, permutation
nulls, FDR/Bonferroni correction, TOST equivalence, MDES reporting) and every number is traceable
to a committed result file. This program has caught and corrected its own overclaims twice
(the digital-brain N=4 fingerprint, and a failed FIM-unification conjecture) rather than hide them.
See `rigor/RIGOR_FINDINGS.md` and `rigor/STRENGTHENED_CLAIMS.md` for the full audit trail.

## What's still open

The within-session identification confound and the powered human reconstruction test both need
data this repo does not have (HCP gated access; a properly powered pooled 7T/3T human study).
Ready-to-execute protocols for both live in the source research folder
(`PREREGISTERED_PROTOCOLS_PENDING_DATA.md`) and will be added here once run.

## License
MIT (each subdirectory retains its original license file).
