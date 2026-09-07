"""
Independent verification of the clique-complex Betti counterexample.
Shares NO code with the Lean file. Multiple routes, different fields/libraries.
Expected ground truth:
  G_small = 4-cycle 0-1-2-3-0:        graph b1 = 1, clique b1 = 1, b0 = 1
  G_big   = + diagonal {0,2}:          graph b1 = 2, clique b1 = 0, b0 = 1
"""
import numpy as np
from sympy import Matrix
import networkx as nx
from itertools import combinations

V = [0, 1, 2, 3]
E_small = [(0,1),(1,2),(2,3),(0,3)]
E_big   = [(0,1),(1,2),(2,3),(0,3),(0,2)]

def triangles(edges):
    """3-cliques = flag-complex 2-simplices, derived (not hand-picked)."""
    eset = {frozenset(e) for e in edges}
    tris = []
    for a,b,c in combinations(sorted(set([x for e in edges for x in e])), 3):
        if {frozenset({a,b}),frozenset({b,c}),frozenset({a,c})} <= eset:
            tris.append((a,b,c))
    return tris

def graph_b1(edges):
    """Cyclomatic number = Chung et al.'s 'number of cycles' = E - V + components."""
    G = nx.Graph(); G.add_nodes_from(V); G.add_edges_from(edges)
    comps = nx.number_connected_components(G)
    return len(edges) - G.number_of_nodes() + comps, comps

# ---------- ROUTE 2: signed boundary maps over Q (rational rank, sympy) ----------
def d1_signed(edges):
    M = np.zeros((len(V), len(edges)), dtype=int)
    for j,(a,b) in enumerate(edges):
        M[a,j] = -1; M[b,j] = +1          # ∂[a,b] = b - a
    return M

def d2_signed(edges, tris):
    ei = {e:i for i,e in enumerate(edges)}
    M = np.zeros((len(edges), len(tris)), dtype=int)
    for j,(a,b,c) in enumerate(tris):     # ∂[a,b,c] = [b,c] - [a,c] + [a,b]
        M[ei[(b,c)],j] += 1
        M[ei[(a,c)],j] -= 1
        M[ei[(a,b)],j] += 1
    return M

def betti_signed(edges):
    tris = triangles(edges)
    D1 = d1_signed(edges); D2 = d2_signed(edges, tris)
    # chain condition ∂1∂2 = 0
    assert (D1 @ D2 == 0).all(), "chain condition failed!"
    r1 = Matrix(D1).rank(); r2 = Matrix(D2).rank() if D2.size else 0
    b0 = len(V) - r1
    b1 = (len(edges) - r1) - r2
    return b0, b1, r1, r2

# ---------- ROUTE 3: independent GF(2) rank on the SAME matrices as Lean ----------
def rank_gf2(M):
    """Fresh mod-2 Gaussian elimination, unrelated to Lean's rankGF2."""
    A = [[int(x)&1 for x in row] for row in M]
    rows, cols = len(A), (len(A[0]) if A else 0)
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r,rows) if A[i][c]), None)
        if piv is None: continue
        A[r],A[piv] = A[piv],A[r]
        for i in range(rows):
            if i!=r and A[i][c]:
                A[i] = [(x^y) for x,y in zip(A[i],A[r])]
        r += 1
    return r

def d1_gf2(edges):   # unsigned, over GF(2) — matches Lean's construction
    M = np.zeros((len(V), len(edges)), dtype=int)
    for j,(a,b) in enumerate(edges): M[a,j]=1; M[b,j]=1
    return M
def d2_gf2(edges,tris):
    ei={e:i for i,e in enumerate(edges)}
    M=np.zeros((len(edges),len(tris)),dtype=int)
    for j,(a,b,c) in enumerate(tris):
        for e in [(a,b),(b,c),(a,c)]: M[ei[e],j]=1
    return M

def betti_gf2(edges):
    tris=triangles(edges)
    D1=d1_gf2(edges); D2=d2_gf2(edges,tris)
    r1=rank_gf2(D1.T.tolist()); r2=rank_gf2(D2.T.tolist()) if D2.size else 0
    # rank is transpose-invariant; feed columns as given
    r1=rank_gf2([list(D1[:,j]) for j in range(D1.shape[1])])
    r2=rank_gf2([list(D2[:,j]) for j in range(D2.shape[1])]) if D2.size else 0
    return len(V)-r1, (len(edges)-r1)-r2, r1, r2

print("="*70)
for name,E in [("G_small (4-cycle)",E_small),("G_big (+diag {0,2})",E_big)]:
    tris = triangles(E)
    gb1,comps = graph_b1(E)
    b0q,b1q,r1q,r2q = betti_signed(E)
    b0g,b1g,r1g,r2g = betti_gf2(E)
    print(f"\n{name}: edges={len(E)}  triangles(3-cliques)={tris}")
    print(f"  ROUTE graph  : cyclomatic (Chung 'number of cycles') b1 = {gb1}   (components={comps})")
    print(f"  ROUTE 2 (Q,signed) : clique b0={b0q} b1={b1q}   [rank d1={r1q}, rank d2={r2q}]")
    print(f"  ROUTE 3 (GF2,indep): clique b0={b0g} b1={b1g}   [rank d1={r1g}, rank d2={r2g}]")

# ---------- ROUTE 4: gudhi (field-standard simplicial homology) ----------
try:
    import gudhi
    print("\n"+"="*70+"\nROUTE 4: gudhi SimplexTree persistent Betti\n"+"="*70)
    for name,E in [("G_small",E_small),("G_big",E_big)]:
        st = gudhi.SimplexTree()
        for v in V: st.insert([v])
        for e in E: st.insert(list(e))
        for t in triangles(E): st.insert(list(t))   # flag complex 2-skeleton
        st.compute_persistence(persistence_dim_max=True)
        b = st.betti_numbers()
        print(f"  {name}: gudhi betti = {b}  (b0={b[0]}, b1={b[1] if len(b)>1 else 0})")
except Exception as ex:
    print("gudhi route unavailable:", ex)

print("\n"+"="*70)
print("EXPECTED: small -> graph b1=1, clique b1=1 ; big -> graph b1=2, clique b1=0")
print("="*70)
