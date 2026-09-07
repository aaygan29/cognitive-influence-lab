# Two things called "the number of cycles" in a connectome are not the same invariant

A Lean-verified counterexample, corroborated four independent ways and by a blind
adversarial audit, plus a real-connectome demonstration: the "number of cycles" used
as a topological feature in some brain-network studies is, by definition, a function
of two elementary graph counts (edges and connected components) and does not measure
the higher-order topological loops it is sometimes interpreted as measuring.

- **Certificate:** [`lean/CliqueBetti.lean`](lean/CliqueBetti.lean) (Lean 4, no Mathlib, depends on no axioms)
- **Four-route check:** [`verify.py`](verify.py)
- **Real-connectome demo:** [`empirical.py`](empirical.py), figure in [`results/`](results/connectome_betti.png)
- **Blind independent replication + adversarial audit:** [`replication/`](replication/REPLICATION.md)

## The two invariants

Network-neuroscience TDA uses "loops" / "cycles" / "first Betti number" for two
genuinely different objects.

1. **Graph cyclomatic number** `b1(G) = E - V + c` (E edges, V vertices, c connected
   components): the first Betti number of the graph as a 1-dimensional complex. It is
   monotone non-decreasing under edge addition. Chung et al. (2019) define "the number
   of cycles" this way and prove the monotonicity.
2. **Clique-complex first Betti number** `b1(X(G))`, where `X(G)` is the clique (flag /
   Vietoris-Rips) complex: every k-clique becomes a (k-1)-simplex. This is the "cavity"
   count of Giusti et al. and Sizemore et al. Filling triangles kills 1-cycles, so it is
   **not** monotone: an added edge can complete triangles that destroy an existing loop.

These are different invariants. This repository pins the difference to a minimal,
machine-checked certificate and measures its consequence on a real brain.

## The statement

For any finite graph G:

```
b1(X(G))  <=  b1(G) = E - V + c
```

with equality iff G has no filled triangles. Moreover:

- `b1(G)` is monotone non-decreasing under edge addition.
- `b1(X(G))` is **not** monotone.
- The gap is unbounded: for K_n, `b1(G) = C(n,2) - n + 1 -> infinity` while
  `b1(X(G)) = 0` (a simplex is contractible).

**Minimal witness (Lean-verified, no axioms):** from the 4-cycle `C4 = 0-1-2-3-0`, add
the single diagonal `{0,2}`.

| quantity                 | C4 | C4 + {0,2} | on the +1 edge |
|--------------------------|----|-----------|----------------|
| graph cyclomatic `b1(G)` | 1  | 2         | up (monotone)  |
| clique `b1(X(G))`        | 1  | 0         | **down**       |

Adding a connection *reduced* the topological loop count from 1 to 0, because `{0,1,2}`
and `{0,2,3}` become filled triangles (a disk). An exhaustive search over all graphs on
up to 6 vertices (`replication/`) confirms this is the smallest possible instance: C4 is
the unique smallest carrier of a hole, and 32,296 of 251,084 edge-additions strictly
decrease clique b1.

## Scope of the novelty

The underlying topological fact (filled triangles kill H1) is textbook. The contribution
here is narrow and specific:

- a machine-checked minimal certificate (4 vertices, 1 edge) with **zero axiom
  dependencies**, from first-principles GF(2) homology with no external library;
- corroboration by four routes sharing no code, plus three blind agents including an
  adversarial falsifier;
- an empirical demonstration that the distinction bites on a real connectome.

This is not a refutation of a famous open conjecture, and Chung et al. are not wrong
about their own object: their monotonicity proof is correct. The critique is about
interpretation and redundancy, made precise below.

## Verification: four independent routes (`verify.py`)

The fragile part of the Lean proof is a hand-rolled GF(2) rank. It is cross-checked
against three computations that agree exactly.

| route | field / tool                         | C4 clique b1 | C4+{0,2} clique b1 |
|-------|--------------------------------------|--------------|--------------------|
| 1     | Lean, GF(2), own rank                | 1            | 0                  |
| 2     | Python, signed boundary maps over Q  | 1            | 0                  |
| 3     | Python, independent GF(2) rank       | 1            | 0                  |
| 4     | `gudhi` (field-standard TDA library) | 1            | 0                  |

Boundary ranks agree too (rank d1 = 3/3, rank d2 = 0/2). Route 2 works over the rationals
with real simplex orientations; these complexes are torsion-free, so rational Betti
numbers equal the true Betti numbers.

## Empirical demonstration on a real connectome (`empirical.py`)

**Data.** The C. elegans nervous system (White et al. 1986; compiled by Watts and
Strogatz 1998; distributed in M. Newman's network-data collection as
`celegansneural.gml`), a directed weighted graph symmetrized to 297 nodes and 2148
undirected edges (2359 directed synaptic records summed). `empirical.py` downloads it
automatically. A strongest-edge-first weight-threshold filtration is swept, and at each
density both invariants are computed: graph cyclomatic b1, and clique-complex b1 (`gudhi`,
flag 2-skeleton, which determines b1 exactly).

Results (`results/connectome_betti.png`):

- **Full connectome:** graph cyclomatic `b1 = 1852`, clique `b1 = 139`.
- **Clique b1 is non-monotone:** it rises to about 225 near density 0.026, then falls to
  139 as triangles fill loops. Graph cyclomatic b1 climbs monotonically throughout.
- **Graph cyclomatic b1 is an exact function of two elementary counts:**
  `b1(G) = E - V + c` holds with maximum deviation 0 across the filtration. It encodes no
  higher-order or simplicial structure beyond the edge count and the component count.
  (The network stays disconnected until the final edge, so both E and c vary; both are
  first-order quantities.) The clique-complex b1 is not such a function: the same edge and
  component counts, a different value, and non-monotone behavior. That difference is the
  higher-order information graph cyclomatic b1 misses.

### Honest caveats

- Chung et al.'s monotonicity result is correct for their invariant. The contribution is
  the interpretation gap and the redundancy with elementary counts, not a math error.
- The full-density overcount is 13x, but the ratio is **density-dependent, not a flat
  order of magnitude**: it is roughly 2.3-2.5x through the sparse and intermediate regime
  and climbs to 13.3x only at full density (numerator grows monotonically, denominator
  falls after its peak).
- One connectome is a demonstration, not a survey. The exact identity holds for any graph;
  the specific magnitudes and the peak location are C.-elegans-specific and depend on the
  symmetrization choice (sum vs max vs binary).

## Reproduce

```bash
# Lean certificate (Lean 4.33.1; no Mathlib)
lean lean/CliqueBetti.lean            # prints: depends on no axioms

# Four-route triangulation of the counterexample
pip install numpy sympy networkx gudhi matplotlib
python3 verify.py

# Real-connectome demonstration (auto-downloads the data)
python3 empirical.py                  # writes results/connectome_betti.png

# Blind independent + adversarial replication
python3 replication/minimality_and_nonmonotonicity.py
python3 replication/empirical_blind.py
python3 replication/empirical_adversarial.py
```

## Sources

Method claims scrutinized:

- M. K. Chung, H. Lee, V. Solo, R. J. Davidson, S. C. Pollak, "Statistical inference on
  the number of cycles in brain networks," *IEEE Int. Symp. Biomedical Imaging (ISBI)*,
  2019. doi:10.1109/ISBI.2019.8759222
- M. K. Chung, H. Lee, A. DiChristofano, H. Ombao, V. Solo, "Exact topological inference
  of the resting-state brain networks in twins," *Network Neuroscience* 3(3):674-694,
  2019. doi:10.1162/netn_a_00091

Higher-order / clique-complex strand:

- A. E. Sizemore, C. Giusti, A. Betzel, R. F. Betzel, D. S. Bassett, "Cliques and cavities
  in the human connectome," *J. Comput. Neurosci.* 44:115-145, 2018.
  doi:10.1007/s10827-017-0672-6
- C. Giusti, E. Pastalkova, C. Curto, V. Itskov, "Clique topology reveals intrinsic
  geometric structure in neural correlations," *PNAS* 112(44):13455-13460, 2015.
  doi:10.1073/pnas.1506407112
- C. Giusti, R. Ghrist, D. S. Bassett, "Two's company, three (or more) is a simplex,"
  *J. Comput. Neurosci.* 41:1-14, 2016. doi:10.1007/s10827-016-0608-6

Data:

- J. G. White, E. Southgate, J. N. Thomson, S. Brenner, "The structure of the nervous
  system of the nematode Caenorhabditis elegans," *Phil. Trans. R. Soc. Lond. B*
  314:1-340, 1986.
- D. J. Watts, S. H. Strogatz, "Collective dynamics of small-world networks," *Nature*
  393:440-442, 1998. doi:10.1038/30918
- M. E. J. Newman, network data collection, `celegansneural`,
  http://www-personal.umich.edu/~mejn/netdata/

Tools: Lean 4 (`leanprover/lean4:v4.33.1`); GUDHI (https://gudhi.inria.fr/);
NetworkX; SymPy; NumPy.
