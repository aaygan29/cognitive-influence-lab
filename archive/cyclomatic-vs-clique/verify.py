"""
Independent verification of the clique-complex Betti counterexample, sharing no
code with the Lean file. Four routes over different fields/libraries must agree.
Ground truth: G_small (4-cycle) -> graph b1=1, clique b1=1; G_big (+diag {0,2})
-> graph b1=2, clique b1=0.
"""
import numpy as np, networkx as nx
from sympy import Matrix
from itertools import combinations

V = [0, 1, 2, 3]
E_small = [(0,1),(1,2),(2,3),(0,3)]
E_big   = [(0,1),(1,2),(2,3),(0,3),(0,2)]

def triangles(edges):                       # 3-cliques = flag 2-simplices, derived
    eset = {frozenset(e) for e in edges}
    verts = sorted({x for e in edges for x in e})
    return [(a,b,c) for a,b,c in combinations(verts, 3)
            if {frozenset({a,b}),frozenset({b,c}),frozenset({a,c})} <= eset]

def graph_b1(edges):                         # cyclomatic = E - V + components
    G = nx.Graph(); G.add_nodes_from(V); G.add_edges_from(edges)
    c = nx.number_connected_components(G)
    return len(edges) - len(V) + c, c

def betti_signed(edges):                     # ROUTE 2: signed boundaries over Q
    tris = triangles(edges)
    D1 = np.zeros((len(V), len(edges)), int)
    for j,(a,b) in enumerate(edges): D1[a,j], D1[b,j] = -1, 1
    ei = {e:i for i,e in enumerate(edges)}
    D2 = np.zeros((len(edges), len(tris)), int)
    for j,(a,b,c) in enumerate(tris):        # ∂[a,b,c] = [b,c]-[a,c]+[a,b]
        D2[ei[(b,c)],j] += 1; D2[ei[(a,c)],j] -= 1; D2[ei[(a,b)],j] += 1
    assert (D1 @ D2 == 0).all(), "chain condition ∂1∂2=0 failed"
    r1 = Matrix(D1).rank(); r2 = Matrix(D2).rank() if D2.size else 0
    return len(V)-r1, (len(edges)-r1)-r2, r1, r2

def rank_gf2(rows):                          # mod-2 rank by row reduction (rank is
    A = [list(map(int, r)) for r in rows]    # transpose-invariant, so rows are fine)
    r = 0; ncol = len(A[0]) if A else 0
    for c in range(ncol):
        piv = next((i for i in range(r, len(A)) if A[i][c] & 1), None)
        if piv is None: continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(len(A)):
            if i != r and A[i][c] & 1: A[i] = [x ^ y for x, y in zip(A[i], A[r])]
        r += 1
    return r

def betti_gf2(edges):                        # ROUTE 3: independent GF(2) rank
    tris = triangles(edges)
    d1 = [[1 if v in e else 0 for v in V] for e in edges]          # rows = edges
    d2 = [[1 if edges[i] in [(a,b),(b,c),(a,c)] else 0 for i in range(len(edges))]
          for (a,b,c) in tris]                                     # rows = triangles
    r1 = rank_gf2(d1); r2 = rank_gf2(d2) if d2 else 0
    return len(V)-r1, (len(edges)-r1)-r2, r1, r2

for name, E in [("G_small (4-cycle)", E_small), ("G_big (+diag {0,2})", E_big)]:
    gb1, c = graph_b1(E)
    b0q, b1q, r1q, r2q = betti_signed(E)
    b0g, b1g, r1g, r2g = betti_gf2(E)
    print(f"\n{name}: edges={len(E)} triangles={triangles(E)}")
    print(f"  graph cyclomatic b1 = {gb1} (components={c})")
    print(f"  ROUTE 2 (Q,signed) : clique b0={b0q} b1={b1q}  [rank d1={r1q}, d2={r2q}]")
    print(f"  ROUTE 3 (GF2,indep): clique b0={b0g} b1={b1g}  [rank d1={r1g}, d2={r2g}]")

try:                                         # ROUTE 4: gudhi (field-standard)
    import gudhi
    print()
    for name, E in [("G_small", E_small), ("G_big", E_big)]:
        st = gudhi.SimplexTree()
        for s in [[v] for v in V] + [list(e) for e in E] + [list(t) for t in triangles(E)]:
            st.insert(s)
        st.compute_persistence(persistence_dim_max=True)
        b = st.betti_numbers()
        print(f"  ROUTE 4 (gudhi) {name}: betti={b}")
except Exception as ex:
    print("gudhi route unavailable:", ex)

print("\nExpected: small clique b1=1, big clique b1=0 (graph b1 goes 1->2).")
