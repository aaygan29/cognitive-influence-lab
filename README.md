# Two things called "the number of cycles" in a connectome are not the same invariant

A Lean-verified counterexample plus a real-connectome demonstration showing that the
"number of cycles" reported in some topological-data-analysis (TDA) studies of brain
networks is, once the network is connected, an affine function of the edge count, and
therefore not the higher-order topological quantity it is sometimes interpreted as.

## The two invariants

Network-neuroscience TDA papers use the phrase "loops" / "cycles" / "first Betti number"
for two genuinely different objects:

1. **Graph cyclomatic number** `b1(G) = E - V + c` (E edges, V vertices, c connected
   components). This is the first Betti number of the graph seen as a 1-dimensional
   complex. It is provably monotone non-decreasing under edge addition. Chung et al.
   (2019) define "the number of cycles" this way and prove the monotonicity.

2. **Clique-complex first Betti number** `b1(X(G))`, where `X(G)` is the clique (flag /
   Vietoris-Rips) complex: every k-clique of G becomes a (k-1)-simplex. This is the
   "cavity" count of Giusti et al. and Sizemore et al. Filling triangles kills 1-cycles,
   so this quantity is **not** monotone: adding an edge can complete triangles that
   destroy an existing loop, and b1 drops.

These are different invariants. This repository pins the difference to a minimal,
machine-checked certificate and then measures it on a real brain.

## The statement

For any finite graph G:

```
b1(X(G))  <=  b1(G) = E - V + c
```

with equality iff G has no filled triangles. Moreover:

- `b1(G)` is monotone non-decreasing under edge addition.
- `b1(X(G))` is **not** monotone.
- The gap is unbounded: for the complete graph K_n, `b1(G) = C(n,2) - n + 1 -> infinity`
  while `b1(X(G)) = 0` (a simplex is contractible).

**Minimal witness (Lean-verified, `lean/CliqueBetti.lean`, depends on no axioms):**
start from the 4-cycle `C4 = 0-1-2-3-0` and add the single diagonal edge `{0,2}`.

| quantity                 | C4 | C4 + {0,2} | direction on +1 edge |
|--------------------------|----|-----------|----------------------|
| graph cyclomatic `b1(G)` | 1  | 2         | up (monotone)        |
| clique `b1(X(G))`        | 1  | 0         | **down**             |

Adding a connection *reduced* the topological loop count from 1 to 0, because `{0,1,2}`
and `{0,2,3}` become filled triangles (a disk). The graph cyclomatic count went the
other way.

## What is and is not novel here

The underlying topological fact (filled triangles kill H1) is textbook. What this
repository contributes is narrow and specific:

- a machine-checked minimal certificate (4 vertices, 1 edge) with **zero axiom
  dependencies**, built from first-principles GF(2) homology with no external library;
- corroboration by four independent routes that share no code (see below);
- an empirical demonstration that the distinction is not academic on a real connectome.

This is not a refutation of a famous open conjecture, and Chung et al. are not wrong
about their own object: their monotonicity proof is correct. The critique is about
interpretation and redundancy, made precise in the empirical section.

## Verification: four independent routes (`verify.py`)

The fragile part of the Lean proof is a hand-rolled GF(2) rank. It is cross-checked
against three other computations that agree exactly:

| route | field / tool                         | C4 clique b1 | C4+{0,2} clique b1 |
|-------|--------------------------------------|--------------|--------------------|
| 1     | Lean, GF(2), own rank                | 1            | 0                  |
| 2     | Python, signed boundary maps over Q  | 1            | 0                  |
| 3     | Python, independent GF(2) rank       | 1            | 0                  |
| 4     | `gudhi` (field-standard TDA library) | 1            | 0                  |

The boundary ranks agree as well (rank d1 = 3/3, rank d2 = 0/2). Route 2 works over the
rationals with real simplex orientations; these complexes are torsion-free, so rational
Betti numbers equal the true Betti numbers.

## Empirical demonstration on a real connectome (`empirical.py`)

Data: the C. elegans nervous system (Newman 2006 mirror of White et al. / Watts-Strogatz),
297 neurons, 2148 undirected edges after symmetrizing the directed synaptic graph. A
weight-threshold filtration is swept (strong edges first), exactly as weight-filtration
Betti-curve analyses do. At each density both invariants are computed: graph cyclomatic
b1 and clique-complex b1 (`gudhi`, flag 2-skeleton, which determines b1 exactly).

Results (see `results/connectome_betti.png`):

- At the full connectome: graph cyclomatic `b1 = 1852`, clique `b1 = 139`. A **13x**
  overcount.
- Clique `b1` is non-monotone: it rises to 227 near density 0.026, then falls to 139 as
  triangles fill loops. Graph cyclomatic `b1` climbs monotonically the whole way.
- `corr(graph b1, edge count) = 1.000000`.

The correlation is definitional, and that is the point. Once a graph is connected, `c = 1`
and `b1(G) = E - V + 1`, an affine function of the edge count. On this connectome the
network becomes connected almost immediately, so that regime spans essentially the entire
filtration. The graph cyclomatic "number of cycles" therefore carries no information
beyond edge density across the real range: any heritability or group effect it detects is
statistically indistinguishable from an effect of edge count. The genuinely higher-order
quantity (clique b1) is a different, non-monotone, roughly 13x smaller number that density
does not determine.

### Caveats

- Chung et al.'s monotonicity result is mathematically correct for their invariant. The
  contribution here is the interpretation gap and the density-redundancy, not a math error.
- One connectome is a demonstration, not a survey. The affine/definitional point holds for
  any connected network; the 13x magnitude and the peak location are specific to this data.

## Reproduce

```bash
# Lean certificate (Lean 4.33.1; no Mathlib needed)
lean lean/CliqueBetti.lean          # prints: depends on no axioms

# Four-route triangulation of the counterexample
pip install numpy sympy networkx gudhi
python3 verify.py

# Real-connectome demonstration (auto-downloads the data)
python3 empirical.py                # writes results/connectome_betti.png
```

## References

- M. K. Chung et al., "Statistical inference on the number of cycles in brain networks,"
  IEEE ISBI, 2019.
- M. K. Chung et al., "Exact topological inference of the resting-state brain networks in
  twins," Network Neuroscience 3(3), 2019.
- A. E. Sizemore et al., "Cliques and cavities in the human connectome," J. Comput.
  Neurosci., 2018.
- C. Giusti, E. Pastalkova, C. Curto, V. Itskov, "Clique topology reveals intrinsic
  geometric structure in neural correlations," PNAS, 2015.
- Data: M. Newman network data collection; C. elegans neural network from J. G. White et
  al. and D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998).
