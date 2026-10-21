"""
Strengthened stats for the geometric-vs-temporal attribution of the grid torus.
n=12 seeds. Two readouts per persistence diagram:
  (1) gap significance: gapH1 = 2nd/3rd longest H1 bar; gapH2 = 1st/2nd H2 bar.
  (2) bottleneck toroidality Gamma_k = mean bottleneck distance from the diagram to
      the empirical NO-STRUCTURE null (destroy-G diagrams). Large Gamma = far from
      'no torus'. Point clouds are scale-normalized so distances are comparable.
Nonparametric Mann-Whitney U tests REAL vs destroy-G for each metric.
"""
import numpy as np, time, json, gudhi
from ripser import ripser
from numpy.linalg import svd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

N = 500
ks = np.array([[np.cos(a), np.sin(a)] for a in [0, np.pi/3, 2*np.pi/3]])
lam = 0.30; kf = 2*np.pi/lam
RNG = np.random.default_rng(7)
phases = RNG.random((N, 2)); phi0 = phases[0].copy()
dt_f = 0.01; Ttot = 120.0; nf = int(Ttot/dt_f)
pos = np.zeros((nf, 2)); v = RNG.normal(0, 0.003, 2)
for t in range(1, nf):
    v = 0.95*v + RNG.normal(0, 0.0009, 2); pos[t] = pos[t-1] + v
    for d in range(2):
        while pos[t, d] < 0 or pos[t, d] > 1.5:
            pos[t, d] = -pos[t, d] if pos[t, d] < 0 else 3.0 - pos[t, d]
tf = np.arange(nf)*dt_f
osc = 0.6*np.cos(2*np.pi*8*tf) + 0.4*np.cos(2*np.pi*4*tf)
print("speed %.3f m/s" % (np.sqrt((np.diff(pos, axis=0)**2).sum(1)).mean()/dt_f))

def gridS(ph_):
    return np.stack([np.clip((sum(np.cos(kf*((pos-ph_[i]*lam)@k)) for k in ks)+1.5)/4.5, 0, None)
                     for i in range(N)], 1)
S_full = gridS(phases); S_one = gridS(np.tile(phi0, (N, 1)))

def spikes(S, m, seed):
    c = np.random.default_rng(seed).poisson(np.clip(90.0*S*(1+m*osc)[:, None]*dt_f, 0, None))
    return [np.repeat(tf, c[:, i]) for i in range(N)]

def diagrams(times, jitter, dt_bin=0.02, ksm=13, npts=500, dim=6, seed=1):
    rng = np.random.default_rng(seed); nb = int(Ttot/dt_bin); edges = np.arange(nb+1)*dt_bin
    X = np.zeros((nb, N))
    for i, ts in enumerate(times):
        tj = ts + rng.normal(0, jitter, ts.shape) if jitter > 0 else ts
        X[:, i] = np.histogram(np.clip(tj, 0, Ttot-1e-9), edges)[0]
    ker = np.ones(ksm)/ksm; X = np.apply_along_axis(lambda u: np.convolve(u, ker, "same"), 0, X)
    idx = np.argsort(-X.sum(1))[:npts*2]; X = X[idx]
    X = X[rng.choice(len(X), min(npts, len(X)), replace=False)]
    U, Sv, _ = svd(X - X.mean(0), full_matrices=False)
    Z = U[:, :dim]*Sv[:dim]
    Z = Z / np.sqrt((Z**2).sum(1)).mean()                 # scale-normalize
    d = ripser(Z, maxdim=2)['dgms']
    return {k: d[k][np.isfinite(d[k][:, 1])] for k in (1, 2)}

def gapof(D):
    g = {}
    for k in (1, 2):
        L = np.sort(D[k][:, 1]-D[k][:, 0])[::-1]; L = np.concatenate([L, [0]*5])
        g[k] = (L[1]/(L[2]+1e-9)) if k == 1 else (L[0]/(L[1]+1e-9))
    return g[1], g[2]

def bott(Da, Db, k):
    return gudhi.bottleneck_distance(Da[k].tolist(), Db[k].tolist())

SEEDS = list(range(12)); JIT = [0.0, 0.10, 0.30, 0.50]
t0 = time.time()
D = {"REAL": {}, "m0": {}, "G": {}}
Djit = {j: {} for j in JIT}
for s in SEEDS:
    rsp = spikes(S_full, 1.0, s)
    D["REAL"][s] = diagrams(rsp, 0.0, seed=s)
    D["m0"][s] = diagrams(spikes(S_full, 0.0, s), 0.0, seed=s)
    D["G"][s] = diagrams(spikes(S_one, 1.0, s), 0.0, seed=s)
    for j in JIT:
        Djit[j][s] = D["REAL"][s] if j == 0.0 else diagrams(rsp, j, seed=s)
    print(f"  seed {s} ({time.time()-t0:.0f}s)")

nullseeds = SEEDS[:6]
def gamma(diag, k):     # bottleneck distance to the destroy-G null (mean over null seeds)
    return np.mean([bott(diag, D["G"][ns], k) for ns in nullseeds])

def collect(dset):
    g1 = []; g2 = []; G1 = []; G2 = []
    for s, dg in dset.items():
        a, b = gapof(dg); g1.append(a); g2.append(b)
        G1.append(gamma(dg, 1)); G2.append(gamma(dg, 2))
    return dict(g1=np.array(g1), g2=np.array(g2), G1=np.array(G1), G2=np.array(G2))

M = {c: collect(D[c]) for c in ("REAL", "m0", "G")}
def line(c):
    m = M[c]
    return (f"{c:5s}  gapH1 {m['g1'].mean():.2f}±{m['g1'].std()/np.sqrt(len(m['g1'])):.2f}  "
            f"gapH2 {m['g2'].mean():.2f}±{m['g2'].std()/np.sqrt(len(m['g2'])):.2f}  "
            f"GammaH1 {m['G1'].mean():.3f}  GammaH2 {m['G2'].mean():.3f}")
print("\n(mean ± sem, n=12)"); [print(" ", line(c)) for c in ("REAL", "m0", "G")]

def mw(a, b):
    u, p = mannwhitneyu(a, b, alternative="greater"); return p
print("\nMann-Whitney (REAL > destroyG), p-values:")
for metric in ("g1", "g2", "G1", "G2"):
    print(f"  {metric}: p = {mw(M['REAL'][metric], M['G'][metric]):.4f}")

jitstats = {}
for j in JIT:
    c = collect(Djit[j]); jitstats[j] = {k: [float(c[k].mean()), float(c[k].std()/np.sqrt(len(c[k])))] for k in c}
print("\njitter sweep (mean±sem):  Δt   gapH1        gapH2        GammaH1")
for j in JIT:
    s = jitstats[j]; print(f"  {int(j*1000):4d}  {s['g1'][0]:.2f}±{s['g1'][1]:.2f}  "
                           f"{s['g2'][0]:.2f}±{s['g2'][1]:.2f}  {s['G1'][0]:.3f}±{s['G1'][1]:.3f}")

json.dump({"main": {c: {k: M[c][k].tolist() for k in M[c]} for c in M}, "jit": jitstats},
          open("results/stats_results.json", "w"), indent=2)

# figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
conds = ["REAL", "m0", "G"]; labs = ["REAL\n(G+T)", "destroy T\n(m=0)", "destroy G\n(1 phase)"]
for arr, col, off, lb in [("G1", "#2471a3", -0.2, "Γ H1 loops"), ("G2", "#c0392b", 0.2, "Γ H2 void")]:
    mu = [M[c][arr].mean() for c in conds]; se = [M[c][arr].std()/np.sqrt(12) for c in conds]
    ax[0].bar(np.arange(3)+off, mu, 0.38, yerr=se, capsize=4, color=col, label=lb)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labs)
ax[0].set_ylabel("bottleneck distance to no-structure null (Γ)")
ax[0].set_title("Attribution (n=12): distance from 'no torus'"); ax[0].legend(fontsize=8)
jx = [int(j*1000) for j in JIT]
ax[1].errorbar(jx, [jitstats[j]["g1"][0] for j in JIT], yerr=[jitstats[j]["g1"][1] for j in JIT],
               marker="o", color="#2471a3", label="gapH1")
ax[1].errorbar(jx, [jitstats[j]["G1"][0] for j in JIT], yerr=[jitstats[j]["G1"][1] for j in JIT],
               marker="^", color="#1abc9c", label="Γ H1")
ax[1].axhline(1.0, ls="--", c="k", lw=1); ax[1].set_xlabel("spike-time jitter Δt (ms)")
ax[1].set_title("Temporal jitter effect (n=6/level)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/attribution_stats.png", dpi=130)
print("\nsaved attribution_stats.png ; total %.0fs" % (time.time()-t0))
