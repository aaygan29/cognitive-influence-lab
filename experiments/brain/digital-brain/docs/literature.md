# Related work (added 2026-10 from Delete Later/papers)

Grounding for the geometry-aware individual fMRI encoder. Source PDFs in
`~/Desktop/Delete Later/papers`.

### Zhang, Finn, Sabuncu & Kuceyeski (2026), *Communications Biology*
**A view-engage-predict framework for enhancing brain-behavior mapping with
naturalistic movie-watching fMRI.**
[10.1038/s42003-026-10411-9](https://doi.org/10.1038/s42003-026-10411-9)

A deep-network framework predicts cognitive scores and sex from functional
connectivity during naturalistic movie viewing, and shows movie-watching FC
outperforms resting-state FC, with the gain tied to how strongly the content
synchronises brain activity. Relevance: supports the choice of naturalistic
stimuli over rest for individual encoding, and gives an external benchmark for
brain-behaviour mapping accuracy the digital-brain encoder should be compared
against (not just within-subject reconstruction).

### Chen, M. (2025), *MENDR: Manifold Explainable Neural Data Representations*
Georgia Tech, MS thesis (PDF: `~/Desktop/Delete Later/papers/CHEN-THESIS-2025.pdf`)

Manifold-based, explainable representations of neural data. Relevance: directly
on-point for the encoder's geometry-aware / manifold claims -- a comparison
point for whether the learned representation is explainable on the manifold it
is claimed to respect, and a source of manifold-explainability methods to borrow.

### Feldman & Skolnick (2026), *A Shortcut to Structure in AlphaFold 3*
arXiv:2610.08937 (PDF: `~/Desktop/Delete Later/papers/2610.08937v1.pdf`)

Uses causal interventions on internal representations and per-block probing to
trace how global structure forms in AF3, finding the MSA acts as a structural
shortcut. Relevance: a clean template for the *causal-intervention* style of
mechanistic analysis the digital-brain encoder's internal representations should
get, rather than correlational probing alone.
