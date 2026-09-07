"""
Blind replication B: independent re-run of the C. elegans connectome analysis,
written from scratch by a blind subagent (shares no code with ../empirical.py).
Reproduced here verbatim except the data path (points at ../data). Run
`python3 empirical.py` once first to download the data, or fetch it into ../data.

Reports, from scratch: full-network b1_graph vs b1_clique and their ratio,
whether b1_clique is monotone, and the component trajectory.
"""
import os, re, numpy as np, networkx as nx, gudhi
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GML = os.path.join(HERE, "..", "data", "celegansneural.gml")
assert os.path.exists(GML), "run `python3 empirical.py` first to download the data into data/"

txt = open(GML).read()
edge_blocks = re.findall(r'edge\s*\[(.*?)\]', txt, re.S)
w = defaultdict(float); nodes = set()
for b in edge_blocks:
    s = int(re.search(r'source\s+(\d+)', b).group(1))
    t = int(re.search(r'target\s+(\d+)', b).group(1))
    vm = re.search(r'value\s+(\d+)', b); val = float(vm.group(1)) if vm else 1.0
    a, c = (s, t) if s <= t else (t, s)
    w[(a, c)] += val; nodes.add(s); nodes.add(t)

node_ids = sorted(nodes); V = len(node_ids)
print("nodes V =", V, " undirected weighted edges =", len(w),
      " directed edge records =", len(edge_blocks))
edges = sorted(w.items(), key=lambda kv: -kv[1]); M = len(edges)

idxs = [i for i in sorted(set(int(round(M*(i+1)/40)) for i in range(40))) if i > 0]
results = []
for k in idxs:
    incl = edges[:k]
    G = nx.Graph(); G.add_nodes_from(node_ids)
    for (a, c), val in incl: G.add_edge(a, c)
    m = G.number_of_edges(); ncomp = nx.number_connected_components(G)
    b1_graph = m - V + ncomp
    st = gudhi.SimplexTree()
    for n in node_ids: st.insert([n])
    for (a, c), val in incl: st.insert([a, c])
    st.expansion(2); st.compute_persistence(persistence_dim_max=True)
    betti = st.betti_numbers(); b1_clique = betti[1] if len(betti) > 1 else 0
    results.append((k, m, ncomp, b1_graph, b1_clique, m/(V*(V-1)/2)))

k, m, nc, bg, bc, d = results[-1]
print("\n=== FULL NETWORK ===")
print("b1_graph =", bg, " b1_clique =", bc, " ratio =", round(bg/bc, 2))
bcs = [r[4] for r in results]; ds = [r[5] for r in results]
peak = max(bcs); pi = bcs.index(peak)
mono = all(bcs[i+1] >= bcs[i] for i in range(len(bcs)-1))
print("b1_clique peak =", peak, "at density", round(ds[pi], 4), " final =", bcs[-1],
      " monotone?", mono, " rises-then-falls?", (not mono) and peak > bcs[-1])
print("component counts across filtration:", [r[2] for r in results])
