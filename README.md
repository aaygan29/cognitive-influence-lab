# neurowire

An in-silico apparatus for a single question: if you copied a brain's wiring (its connectome) perfectly, would you have the person, and if not, what else would you have to store?

## What it does

- Separates three things usually conflated as "identity": identification (telling people apart), reconstruction (regenerating an individual's function), and constitution (whether a copy is literally you).
- Runs an identity-sufficiency curve and a compensatory-degeneracy test in silico, plus real-data checks, to see what the wiring alone does and does not determine.

**Finding:** wiring/topology is enough to *identify* people (even a binarized connectome fingerprints individuals), but the exact weights are not needed to reconstruct function (they are degenerate), and topology alone is non-identifying of function. Storing an identity requires the connectome plus enough functional constraint to pin the right functional-equivalence class. Constitution is left explicitly unresolved.

## Data & grounding

- ABIDE resting-state functional connectomes (CC200 atlas, N = 248) for the real-data identification test.
- Natural Scenes Dataset (N = 8) for a second real-data check.
- In-silico experiments generate their own controlled networks; figures and per-experiment results JSON are included.

## License

MIT — see [LICENSE](LICENSE).
