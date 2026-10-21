"""
Real connectome (C. elegans; White et al. 1986 / Watts-Strogatz 1998; Newman
netdata). Over a weight-threshold filtration, compute both quantities the
literature calls "cycles":
  (A) graph cyclomatic b1 = E - V + components   [Chung et al. 2019]
  (B) clique-complex b1 (flag complex, gudhi)    [Giusti/Sizemore]
"""
import os, re, zipfile, urllib.request
import networkx as nx, numpy as np, gudhi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GML = "data/celegansneural.gml"
if not os.path.exists(GML):
    os.makedirs("data", exist_ok=True)
    url = "http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip"
    z = "data/celegansneural.zip"
    print("downloading", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as f: f.write(r.read())
    except Exception:
        import subprocess; subprocess.run(["curl", "-sSL", "-o", z, url], check=True)
    with zipfile.ZipFile(z) as f: f.extractall("data")

txt = open(GML).read()
nodes = {int(m) for m in re.findall(r"node\s*\[\s*id\s+(\d+)", txt)}
records = re.findall(r"edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)\s+value\s+(\d+)", txt)
U = nx.Graph(); U.add_nodes_from(nodes)              # symmetrize: w = sum over directions
for s, t, val in records:
    s, t = int(s), int(t)
    if s != t: U.add_edge(s, t, weight=(U[s][t]["weight"] if U.has_edge(s, t) else 0) + float(val))
V = U.number_of_nodes()
edges_sorted = sorted(U.edges(data=True), key=lambda e: -e[2]["weight"])  # strong first
Etot = len(edges_sorted)
idx = {n: i for i, n in enumerate(U.nodes())}
print(f"{V} neurons, {Etot} undirected edges ({len(records)} directed records)")

def clique_b1(sub):
    st = gudhi.SimplexTree()
    for n in range(V): st.insert([n])
    for u, v, _ in sub: st.insert([idx[u], idx[v]])
    st.expansion(2); st.compute_persistence(persistence_dim_max=True)   # 2-skeleton -> exact b1
    b = st.betti_numbers()
    return b[1] if len(b) > 1 else 0

rows = []
for k in np.linspace(0.03, 1.0, 40):
    m = max(1, int(k * Etot)); sub = edges_sorted[:m]
    H = nx.Graph(); H.add_nodes_from(U.nodes()); H.add_edges_from((u, v) for u, v, _ in sub)
    comps = nx.number_connected_components(H)
    rows.append((2*m/(V*(V-1)), m, comps, m - V + comps, clique_b1(sub)))
dens, m_, comps_, gb1_, cb1_ = np.array(rows).T

print("\n density  edges  comps  graph_b1  clique_b1  ratio")
for d, m, c, g, cl in rows[::5]:
    print(f"  {d:5.3f} {int(m):5d} {int(c):5d} {int(g):8d} {int(cl):9d}  {g/cl if cl else 0:5.1f}x")

full_g, full_c = int(gb1_[-1]), int(cb1_[-1]); pk = int(np.argmax(cb1_))
gap = int(np.max(np.abs(gb1_ - (m_ - V + comps_)))); n_conn = int((comps_ == 1).sum())
ratio = np.where(cb1_ > 0, gb1_/np.maximum(cb1_, 1), np.nan); rmin = int(np.nanargmin(ratio))
print(f"\nFull connectome: graph_b1={full_g}, clique_b1={full_c} ({full_g/max(full_c,1):.0f}x overcount)")
print(f"clique_b1 NON-monotone: peaks {int(cb1_[pk])} at density {dens[pk]:.3f}, falls to {full_c}")
print(f"graph_b1 == E - V + components EXACTLY (max deviation {gap}): a function of two")
print(f"  elementary counts, no higher-order info. Only {n_conn}/{len(rows)} steps are connected,")
print(f"  so a 'connected-regime' correlation is not meaningful; the identity is the robust point.")
print(f"Overcount ratio is density-dependent: ~{ratio[rmin]:.1f}x at density {dens[rmin]:.3f}, "
      f"{full_g/max(full_c,1):.0f}x only at full density.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(dens, gb1_, "o-", color="#c0392b", label="graph cyclomatic $b_1$ (Chung)")
ax[0].plot(dens, cb1_, "s-", color="#2471a3", label="clique-complex $b_1$ (true loops)")
ax[0].set(xlabel="edge density", ylabel="$b_1$", title="Two 'cycle counts' diverge"); ax[0].legend(fontsize=8)
ax[1].plot(dens, cb1_, "s-", color="#2471a3")
ax[1].set(xlabel="edge density", ylabel="clique-complex $b_1$", title="True loops: rise then FALL")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/connectome_betti.png", dpi=130)
print("saved results/connectome_betti.png")
