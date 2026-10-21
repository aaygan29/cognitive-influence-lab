"""
Blind replication A + adversarial claim (i): is clique-complex b1 monotone under
edge addition? Exhaustive brute force over ALL simple graphs on 3..6 vertices,
computing clique b1 (gudhi flag expansion) before and after every possible edge
addition, counting decreases and locating the smallest carrier of a hole.

This is the independent script produced by a blind subagent (it shares no code
with lean/CliqueBetti.lean or verify.py). Reproduced here verbatim for external
validation. Needs only networkx + gudhi.

Expected: many edge-additions strictly DECREASE clique b1; the smallest graph
with clique b1 > 0 is the 4-cycle C4, so C4 + one chord (4 vertices, 4->5 edges)
is the minimal non-monotone instance (clique b1 1->0 while cyclomatic 1->2).
"""
import itertools, networkx as nx, gudhi

def clique_b1(G):
    st = gudhi.SimplexTree()
    for v in G.nodes(): st.insert([v])
    for u, v in G.edges(): st.insert([u, v])
    st.expansion(3)                      # dim>=2 triangles suffice for b1; 3 to be safe
    st.compute_persistence(persistence_dim_max=True)
    return st.betti_numbers()

def cyclomatic(G):
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

def all_graphs(n):
    nodes = list(range(n)); pairs = list(itertools.combinations(nodes, 2))
    for mask in range(1 << len(pairs)):
        G = nx.Graph(); G.add_nodes_from(nodes)
        for i, p in enumerate(pairs):
            if mask >> i & 1: G.add_edge(*p)
        yield G

found = checked = 0; best = None
for n in range(3, 7):
    for G in all_graphs(n):
        b = clique_b1(G); b1 = b[1] if len(b) > 1 else 0
        for e in list(nx.non_edges(G)):
            H = G.copy(); H.add_edge(*e)
            bh = clique_b1(H); b1h = bh[1] if len(bh) > 1 else 0
            checked += 1
            if b1h < b1:
                found += 1
                if best is None or H.number_of_nodes() <= best[0]:
                    best = (H.number_of_nodes(), sorted(G.edges()), e, b1, b1h,
                            cyclomatic(G), cyclomatic(H))
    print(f"n<={n} done, decreasing edge-additions so far={found}, checked={checked}")

print("\nCLAIM (i): found", found, "edge-additions that STRICTLY DECREASE clique b1")
if best:
    print("smallest counterexample: n =", best[0])
    print("  base edges:", best[1], " added edge:", best[2])
    print("  clique b1:", best[3], "->", best[4], "   cyclomatic:", best[5], "->", best[6])
