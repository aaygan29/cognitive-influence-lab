"""
Adversarial replication (claim ii): independent connectome analysis written from
scratch by a blind subagent tasked with FALSIFYING the empirical claims. Shares no
code with ../empirical.py. Reproduced verbatim except the data path (../data).
Run `python3 empirical.py` once first to download the data.

Surfaces the two caveats that were then folded into the main analysis:
  - the graph is disconnected until the final edge, so a 'connected-regime'
    correlation is a single point (not meaningful); and
  - the overcount ratio is density-dependent (dips to ~2.5x mid-filtration,
    ~13x only at full density), not a flat order of magnitude.
"""
import os, re, numpy as np, networkx as nx, gudhi

HERE = os.path.dirname(os.path.abspath(__file__))
GML = os.path.join(HERE, "..", "data", "celegansneural.gml")
assert os.path.exists(GML), "run `python3 empirical.py` first to download the data into data/"

txt = open(GML).read()
edge_re = re.compile(r"edge\s*\[(.*?)\]", re.S)
src_re = re.compile(r"source\s+(\d+)"); tgt_re = re.compile(r"target\s+(\d+)")
val_re = re.compile(r"value\s+(\d+)")

edges = {}; n_raw = 0
for blk in edge_re.findall(txt):
    s = int(src_re.search(blk).group(1)); t = int(tgt_re.search(blk).group(1))
    m = val_re.search(blk); wv = int(m.group(1)) if m else 1
    n_raw += 1
    if s == t: continue
    key = (min(s, t), max(s, t)); edges[key] = edges.get(key, 0) + wv

nodes = sorted({n for e in edges for n in e})
print("raw edge lines:", n_raw, " undirected unique pairs:", len(edges), " nodes:", len(nodes))
order = sorted(edges.items(), key=lambda kv: -kv[1]); E = len(order)

def clique_b1(elist, nodeset):
    st = gudhi.SimplexTree()
    for v in nodeset: st.insert([v])
    for (u, v), _ in elist: st.insert([u, v])
    st.expansion(3); st.compute_persistence(persistence_dim_max=True)
    b = st.betti_numbers(); return b[1] if len(b) > 1 else 0

steps = sorted(set(int(x) for x in np.linspace(1, E, 40)))
rows = []
for k in steps:
    sub = order[:k]
    G = nx.Graph(); G.add_nodes_from(nodes)
    for (u, v), wv in sub: G.add_edge(u, v)
    comps = nx.number_connected_components(G); V = G.number_of_nodes()
    cyc = k - V + comps; cb1 = clique_b1(sub, nodes)
    rows.append((k, comps, cyc, cb1))
    print(f"E={k:4d} comps={comps:3d} cyclomatic={cyc:5d} clique_b1={cb1:5d} ratio={cyc/max(cb1,1):.2f}")

rows = np.array(rows, float)
conn = rows[rows[:, 1] == 1]
print("\n# connected-regime points:", len(conn),
      " (a correlation needs >=2; strong-edge filtration stays fragmented until the end)")
full = rows[-1]
print("At full density: cyclomatic=%d clique_b1=%d ratio=%.2f"
      % (full[2], full[3], full[2]/max(full[3], 1)))
ratios = rows[:, 2]/np.maximum(rows[:, 3], 1)
print("ratio range across filtration: min=%.1fx  max=%.1fx  => density-dependent, not flat"
      % (ratios.min(), ratios.max()))
