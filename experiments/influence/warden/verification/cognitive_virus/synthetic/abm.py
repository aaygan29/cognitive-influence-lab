"""(iii) Agent-based stochastic version of the same transition rules.

Per-agent hazards (tau-leaping, dt small):
  U -> C : lam * c_i          (c_i = fraction of contacts in C; global fraction if well-mixed)
  C -> U : rho + kap * u_i^2  (u_i = fraction of contacts in U)
  C -> D : mu
  D -> C : sig
In the well-mixed N -> infinity limit these reproduce Eqs. (1)-(3) exactly.
Note: E_U (no C, no D) is absorbing for any finite population.
"""
import json, sys, time, numpy as np, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tqdm import tqdm
from model import lam_sn, lam_tc, physical_coupled

rho, kap, mu, sig = 0.1, 0.4, 0.2, 0.1
DT, T = 0.1, 800.0
rng = np.random.default_rng(7)


def p_of(rate):
    return 1.0 - np.exp(-rate * DT)


def well_mixed(N, lam, init, T=T):
    nU, nC = int(round(init[0] * N)), int(round(init[1] * N)); nD = N - nU - nC
    for _ in range(int(T / DT)):
        if nC == 0 and nD == 0: break
        u, c = nU / N, nC / N
        a = rng.binomial(nU, p_of(lam * c))
        # competing exits from C: multinomial
        rC = rho + kap * u * u + mu; pexit = p_of(rC)
        out = rng.binomial(nC, pexit)
        back, toD = (rng.multinomial(out, [(rho + kap * u * u) / rC, mu / rC]) if out else (0, 0))
        rec = rng.binomial(nD, p_of(sig))
        nU += back - a; nC += a - back - toD + rec; nD += toD - rec
    return nU / N, nC / N, nD / N


def er_graph(N, kmean):
    m = int(N * kmean / 2)
    i = rng.integers(0, N, m); j = rng.integers(0, N, m); keep = i != j
    A = sp.coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    return A


def ba_graph(N, m):
    targets = list(range(m)); repeated = []; rows = []; cols = []
    for v in range(m, N):
        for t in set(targets):
            rows += [v, t]; cols += [t, v]
        repeated += list(set(targets)) + [v] * m
        targets = [repeated[k] for k in rng.integers(0, len(repeated), m)]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    return (A > 0).astype(float)


def network(A, lam, init, T=T):
    N = A.shape[0]; deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    s = rng.choice(3, size=N, p=[init[0], init[1], 1 - init[0] - init[1]])  # 0=U 1=C 2=D
    for _ in range(int(T / DT)):
        isU, isC, isD = s == 0, s == 1, s == 2
        if not isC.any() and not isD.any(): break
        c = A @ isC.astype(float) / deg; u = A @ isU.astype(float) / deg
        r = rng.random(N)
        new = s.copy()
        new[isU & (r < p_of(lam * c))] = 1
        rC = rho + kap * u * u + mu; pex = p_of(rC)
        ex = isC & (r < pex)
        r2 = rng.random(N)
        new[ex & (r2 < (rho + kap * u * u) / rC)] = 0
        new[ex & (r2 >= (rho + kap * u * u) / rC)] = 2
        new[isD & (r < p_of(sig))] = 1
        s = new
    return (s == 0).mean(), (s == 1).mean(), (s == 2).mean()


def main(quick=False):
    lams = np.round(np.linspace(0.30, 0.70, 17), 3)
    inits = {"near_uncoupled": (0.98, 0.02), "coupled": (0.2, 0.3)}
    reps = 2 if quick else 3
    res = dict(params=dict(rho=rho, kap=kap, mu=mu, sig=sig, dt=DT, T=T), lam_SN=lam_sn(rho, kap), lam_TC=lam_tc(rho, kap),
               meanfield_U_minus={str(l): physical_coupled(l, rho, kap) for l in lams}, runs=[])
    N_net = 2000
    graphs = {"ER_k10": er_graph(N_net, 10), "BA_m5": ba_graph(N_net, 5)}
    for g, A in graphs.items():
        d = np.asarray(A.sum(1)).ravel(); res[f"{g}_degree"] = dict(mean=float(d.mean()), max=float(d.max()), cv=float(d.std() / d.mean()))
    configs = [("wellmixed", 1000), ("wellmixed", 10000), ("ER_k10", N_net), ("BA_m5", N_net)]
    jobs = [(kind, N, lam, iname, rep) for kind, N in configs for lam in lams for iname in inits for rep in range(reps)]
    t0 = time.time()
    for kind, N, lam, iname, rep in tqdm(jobs, desc="ABM"):
        init = inits[iname]
        out = well_mixed(N, lam, init) if kind == "wellmixed" else network(graphs[kind], lam, init)
        res["runs"].append(dict(kind=kind, N=N, lam=float(lam), init=iname, rep=rep, U=out[0], C=out[1], D=out[2]))
    res["wall_s"] = time.time() - t0
    # summaries
    summ = {}
    for kind, N in configs:
        key = f"{kind}_N{N}"; summ[key] = {}
        for iname in inits:
            rows = [(r["lam"], r["U"]) for r in res["runs"] if r["kind"] == kind and r["N"] == N and r["init"] == iname]
            meanU = {l: float(np.mean([u for ll, u in rows if ll == l])) for l in lams}
            absorbed = {l: float(np.mean([u > 0.999 for ll, u in rows if ll == l])) for l in lams}
            summ[key][iname] = dict(meanU={str(k): v for k, v in meanU.items()}, frac_absorbed_U1={str(k): v for k, v in absorbed.items()})
            # empirical thresholds: first lam where majority of runs leave U=1 (up) / last lam where coupled runs return (down)
            if iname == "near_uncoupled":
                lv = [l for l in lams if absorbed[l] < 0.5]; summ[key]["empirical_invasion_lam"] = float(lv[0]) if lv else None
            else:
                rt = [l for l in lams if absorbed[l] >= 0.5]; summ[key]["empirical_collapse_to_U1_upto_lam"] = float(max(rt)) if rt else None
    res["summary"] = summ
    json.dump(res, open("results/abm.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    mf = [physical_coupled(l, rho, kap) for l in lams]
    for ax, (kind, N) in zip(axes, configs):
        key = f"{kind}_N{N}"
        for iname, col in (("near_uncoupled", "b"), ("coupled", "r")):
            ax.plot(lams, list(summ[key][iname]["meanU"].values()), col + "o-", ms=3, label=f"ABM from {iname}")
        ax.plot(lams, [1 if l < lam_tc(rho, kap) else np.nan for l in lams], "k-", lw=0.7)
        ax.plot(lams, [m if m is not None else np.nan for m in mf], "k--", lw=0.7, label="mean-field U-")
        ax.axvline(lam_sn(rho, kap), color="gray", lw=0.5); ax.axvline(lam_tc(rho, kap), color="gray", lw=0.5, ls="--")
        ax.set_title(key); ax.set_xlabel("lambda")
    axes[0].set_ylabel(f"U at t={T:g} (mean of reps)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/abm.png", dpi=130)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
