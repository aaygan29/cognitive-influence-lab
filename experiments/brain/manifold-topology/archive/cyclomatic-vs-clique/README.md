# Graph cyclomatic number and clique-complex b₁ are different invariants: a verified demonstration

This repository is a small, machine-checked **teaching / verification artifact** about a
distinction that is well known in topological data analysis (TDA): the graph cyclomatic
number `b1(G) = E - V + c` and the first Betti number of the clique complex `b1(X(G))` are
different invariants and behave differently under edge addition. It provides a Lean-verified
minimal example, a four-route cross-check, and a real-connectome illustration.

**It is not a critique of any paper, and not new mathematics.** See "What this is not" below.
An earlier version of this README overstated all of those things; the "Corrections" section
records what was wrong and why.

- Certificate: [`lean/CliqueBetti.lean`](lean/CliqueBetti.lean) (Lean 4, no Mathlib)
- Four-route cross-check: [`verify.py`](verify.py)
- Real-connectome illustration: [`empirical.py`](empirical.py), figure in [`results/`](results/connectome_betti.png)
- Independent + adversarial replication: [`replication/`](replication/REPLICATION.md)

## The two invariants

1. **Graph cyclomatic number** `b1(G) = E - V + c` (E edges, V vertices, c components): the
   first Betti number of the graph as a 1-dimensional complex. Monotone non-decreasing under
   edge addition. This is what Chung et al. (2019) call "the number of cycles"; they state
   explicitly that it is a graph Betti number and prove the monotonicity. Their proof is
   correct.
2. **Clique-complex first Betti number** `b1(X(G))`, where `X(G)` is the clique (flag)
   complex: every k-clique becomes a (k-1)-simplex. This is the "cavity" count of Giusti et
   al. and Sizemore et al. Filling triangles kills 1-cycles, so it is **not** monotone.

Both facts are standard. The non-monotonicity of clique-complex Betti numbers is, in fact, a
core motivation for persistent homology, which tracks the birth and death of features rather
than counting them at a single threshold.

## The elementary relationship

For any finite graph G:

```
b1(X(G))  <=  b1(G) = E - V + c
```

equality iff G has no filled triangles. `b1(G)` is monotone under edge addition; `b1(X(G))`
is not; and the gap is unbounded (for K_n, `b1(G) = C(n,2) - n + 1 -> infinity` while
`b1(X(G)) = 0`, since a simplex is contractible).

**Minimal witness (Lean-verified).** From the 4-cycle `C4 = 0-1-2-3-0`, add the diagonal
`{0,2}`:

| quantity                 | C4 | C4 + {0,2} | on the +1 edge |
|--------------------------|----|-----------|----------------|
| graph cyclomatic `b1(G)` | 1  | 2         | up (monotone)  |
| clique `b1(X(G))`        | 1  | 0         | down           |

`{0,1,2}` and `{0,2,3}` become filled triangles (a disk), so the clique-complex loop is
filled. An exhaustive search over all graphs on up to 6 vertices (`replication/`) confirms C4
is the smallest carrier of a hole, so this is the minimal instance; 32,296 of 251,084
edge-additions strictly decrease clique b1.

## Verification: four independent routes (`verify.py`)

The one non-trivial ingredient in the Lean file is a hand-rolled GF(2) rank, cross-checked
against three other computations that agree exactly.

| route | field / tool                         | C4 clique b1 | C4+{0,2} clique b1 |
|-------|--------------------------------------|--------------|--------------------|
| 1     | Lean, GF(2), own rank                | 1            | 0                  |
| 2     | Python, signed boundary maps over Q  | 1            | 0                  |
| 3     | Python, independent GF(2) rank       | 1            | 0                  |
| 4     | `gudhi` (field-standard TDA library) | 1            | 0                  |

Boundary ranks agree too (rank d1 = 3/3, rank d2 = 0/2). Route 2 uses oriented boundaries over
Q; the complexes are torsion-free, so rational Betti numbers equal the true Betti numbers.

## Real-connectome illustration (`empirical.py`)

To show the two invariants differ on real, triangle-rich data (not to test any paper's
method), `empirical.py` loads the C. elegans nervous system (White et al. 1986; Watts and
Strogatz 1998; Newman netdata `celegansneural.gml`), symmetrized to 297 nodes / 2148 edges,
and sweeps a weight-threshold filtration. At each density it reports both invariants.

- Full network: graph cyclomatic `b1 = 1852`, clique `b1 = 139`. These differ by ~13x because
  the connectome has many triangles; this is a difference between two invariants, **not** an
  error and **not** an "overcount."
- Clique b1 is non-monotone: it rises to about 225 near density 0.026, then falls to 139.
  Graph cyclomatic b1 climbs monotonically. The ratio is density-dependent (~2.3x sparse,
  ~13x at full density), not a fixed factor.
- Pointwise, `b1(G) = E - V + c` exactly (max deviation 0 over the filtration).

**What this illustration does NOT show.** It does not show that graph cyclomatic b1 is
uninformative. Chung et al. do not report a single number; they compute the whole Betti
*curve* over the filtration and compare curves with a Kolmogorov-Smirnov distance. The curve's
shape depends on the full edge-weight ordering and the evolving component count, so it is not a
function of total edge count, and the KS distance between two subjects' curves is not reducible
to an edge-count comparison. An earlier version of this repo claimed the opposite; that claim
was wrong and is retracted (see Corrections).

## What this is not

- **Not a critique of Chung et al. or anyone else.** No cited paper conflates the two
  invariants. Chung et al. compute the graph Betti curve deliberately and correctly.
- **Not new mathematics.** `b1(X(G)) <= b1(G)`, the non-monotonicity, and the K_n gap are
  textbook. The distinction is already understood in the TDA community.
- **Not an impressive formalization.** The Lean certificate is a concrete 4-vertex
  linear-algebra computation; its small axiom footprint reflects Lean evaluating finite
  arithmetic, not mathematical depth. It is included for reproducibility, not as a result.

The legitimate content is narrow: a cleanly verified, independently replicated demonstration
that these two invariants are different objects, with a minimal certificate and a real-data
illustration of how far apart they can be.

## Corrections

The initial version of this repository (commits before the "reframe" commit) made several
claims that were unsupported or false, retracted here:

1. It framed clique-complex b1 monotonicity as "a belief asserted generally in the TDA
   literature." No one asserts this; the opposite is a motivation for persistent homology.
2. It implied the two invariants are conflated in the literature and framed the work as a
   counterexample to that. Chung et al. explicitly compute the graph invariant.
3. It claimed graph cyclomatic b1 "carries no information beyond edge density" and that "any
   heritability or group effect is statistically indistinguishable from an edge-count effect."
   This is false: Betti curves and their KS distances depend on the weight distribution and
   component structure, not just total edge count, and no such statistical control was run.
4. It called the 297-node ratio an "overcount," implying error where there is only a
   difference between two invariants.
5. It overstated the novelty of the result and the significance of the zero-axiom certificate.

## Reproduce

```bash
lean lean/CliqueBetti.lean            # Lean 4.33.1, no Mathlib
pip install numpy sympy networkx gudhi matplotlib
python3 verify.py
python3 empirical.py                  # auto-downloads data, writes results/connectome_betti.png
python3 replication/minimality_and_nonmonotonicity.py
python3 replication/empirical_blind.py
python3 replication/empirical_adversarial.py
```

## Sources

- M. K. Chung, H. Lee, V. Solo, R. J. Davidson, S. C. Pollak, "Statistical inference on the
  number of cycles in brain networks," IEEE ISBI, 2019. doi:10.1109/ISBI.2019.8759222
- M. K. Chung et al., "Exact topological inference of the resting-state brain networks in
  twins," Network Neuroscience 3(3):674-694, 2019. doi:10.1162/netn_a_00091
- A. E. Sizemore et al., "Cliques and cavities in the human connectome," J. Comput. Neurosci.
  44:115-145, 2018. doi:10.1007/s10827-017-0672-6
- C. Giusti, E. Pastalkova, C. Curto, V. Itskov, "Clique topology reveals intrinsic geometric
  structure in neural correlations," PNAS 112(44):13455-13460, 2015. doi:10.1073/pnas.1506407112
- C. Giusti, R. Ghrist, D. S. Bassett, "Two's company, three (or more) is a simplex," J.
  Comput. Neurosci. 41:1-14, 2016. doi:10.1007/s10827-016-0608-6
- Data: J. G. White et al., Phil. Trans. R. Soc. Lond. B 314:1-340, 1986; D. J. Watts and S. H.
  Strogatz, Nature 393:440-442, 1998; M. E. J. Newman network data collection (`celegansneural`).

Tools: Lean 4 (`leanprover/lean4:v4.33.1`), GUDHI, NetworkX, SymPy, NumPy.
