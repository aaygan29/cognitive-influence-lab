# Independent replication and adversarial audit

Every claim in this repository was re-checked by three **blind** agents that built
their own code from scratch in isolated temporary directories, with no access to
`lean/CliqueBetti.lean`, `verify.py`, or `empirical.py`, and without being told the
expected answers. Their scripts are reproduced here (verbatim except data paths) so
anyone can re-run them.

Run them with:

```bash
python3 ../empirical.py                       # downloads data/ once
python3 minimality_and_nonmonotonicity.py     # ~1 min: exhaustive n<=6 brute force
python3 empirical_blind.py
python3 empirical_adversarial.py
```

## Task 1 — math re-derivation (`minimality_and_nonmonotonicity.py`)

Is clique-complex b1 monotone under edge addition? Exhaustive brute force over all
simple graphs on 3..6 vertices.

- **32,296** of 251,084 edge-additions strictly **decrease** clique-complex b1.
- The smallest graph with clique b1 > 0 is the 4-cycle C4, so the minimal
  non-monotone instance is C4 + one chord: clique b1 `1 -> 0`, cyclomatic `1 -> 2`.
- clique b1 confirmed three ways (gudhi, hand GF(2) rank, sympy over Q) - all agree.

Result: claim confirmed and shown minimal.

## Task 2 — empirical re-run (`empirical_blind.py`)

Independent parse + filtration of the C. elegans connectome.

- 297 nodes, 2148 undirected edges (2359 directed records symmetrized).
- Full network: b1_graph = 1852, b1_clique = 139, ratio 13.32.
- b1_clique non-monotone: peak 224 at density ~0.026, final 139.
- Component trajectory 252 -> ... -> 13 -> 1: the graph is disconnected until the
  final edge.

Result: numbers reproduce.

## Task 3 — adversarial falsification (`empirical_adversarial.py`)

Tasked with breaking both claims. Could not falsify either, but sharpened two
framings, both now corrected in the main analysis:

1. **The affine-ness is not about "connectivity."** The connectome is disconnected
   until the last edge, so there is exactly one connected-regime point and a
   "connected-regime correlation" is not a statistic. The robust statement is the
   exact identity `b1_graph = E - V + c` (max deviation 0 over all steps): b1_graph
   is a deterministic function of two elementary counts, so it holds no higher-order
   information regardless of connectivity.
2. **The 13x overcount is a full-density phenomenon, not a flat order of magnitude.**
   The ratio is density-dependent: roughly 2.3-2.5x through the sparse/intermediate
   regime, climbing to 13.3x only at full density (numerator grows monotonically,
   denominator falls after its peak).

Result: claims survive; two caveats folded into `README.md` and `empirical.py`.
