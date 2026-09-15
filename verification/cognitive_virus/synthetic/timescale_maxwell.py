"""(iv) Timescale separation behind the 1D reduction, and the Maxwell point under alternative reductions."""
import json, numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *

rho, kap, mu, sig = 0.1, 0.4, 0.2, 0.1
s = sig / (mu + sig)


def maxwell(weight):
    def G(lam):
        Um = roots(lam, rho, kap)[0]
        return quad(lambda U: weight(U) * f(U, lam, rho, kap), Um, 1.0)[0]
    return brentq(G, lam_sn(rho, kap) + 1e-9, lam_tc(rho, kap) - 1e-9)


def main():
    out = {}
    out["maxwell_slow_manifold_(1-U)f"] = maxwell(lambda U: (1 - U))  # dU/dt = s(1-U) f(U)
    out["maxwell_time_rescaled_f_only"] = maxwell(lambda U: 1.0)       # dU/dtau = f(U), tau = int C dt
    out["maxwell_note"] = ("Equal-depth point depends on which 1D flow is integrated; both keep the same fixed points "
                           "and thresholds. Gamma is affine in U so using Gamma instead of U does not change it.")
    # timescale ratio at the stable coupled state across lambda
    rows = []
    for lam in tqdm(np.linspace(lam_sn(rho, kap) + 1e-4, 0.8, 60), desc="timescales"):
        Um = roots(lam, rho, kap)[0]
        ev = np.linalg.eigvals(jacobian_UC_interior(Um, lam, rho, kap, mu, sig)).real
        slow, fast = sorted(np.abs(ev))
        rows.append(dict(lam=float(lam), U_minus=float(Um), slow_rate=float(slow), fast_rate=float(fast), ratio=float(fast / slow)))
    out["timescale_rows"] = rows
    out["ratio_at_lam_0.45"] = [r for r in rows if abs(r["lam"] - 0.45) == min(abs(q["lam"] - 0.45) for q in rows)][0]
    out["min_ratio_on_grid"] = min(r["ratio"] for r in rows)
    # E_U side: eigenvalues of (C,D) Jacobian
    out["EU_eigs_lam_0.45"] = sorted(np.linalg.eigvals(jacobian_CD_at_EU(0.45, rho, kap, mu, sig)).real.tolist())
    # reduction error: full 2D vs slow-manifold 1D from the same U0 started ON the manifold
    errs = []
    for lam in (0.42, 0.45, 0.6):
        for U0 in (0.2, 0.5, 0.8):
            y0 = [U0, s * (1 - U0), (1 - s) * (1 - U0)]
            T = 400; te = np.linspace(0, T, 2001)
            full = solve_ivp(rhs_ucd, (0, T), y0, args=(lam, rho, kap, mu, sig), t_eval=te, rtol=1e-9, atol=1e-12)
            red = solve_ivp(lambda t, u: s * (1 - u) * f(u, lam, rho, kap), (0, T), [U0], t_eval=te, rtol=1e-9, atol=1e-12)
            W = full.y[1] - s * (1 - full.y[0])
            errs.append(dict(lam=lam, U0=U0, max_abs_U_err=float(np.max(np.abs(full.y[0] - red.y[0]))),
                             max_abs_offmanifold_W=float(np.max(np.abs(W))), U_final_full=float(full.y[0, -1]), U_final_1D=float(red.y[0, -1])))
    out["reduction_error"] = errs
    json.dump(out, open("results/timescale_maxwell.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy([r["lam"] for r in rows], [r["ratio"] for r in rows]); ax[0].set_xlabel("lambda"); ax[0].set_ylabel("fast/slow rate at U-")
    ax[0].axvline(out["maxwell_slow_manifold_(1-U)f"], ls=":", color="k")
    for lam in (0.36, 0.40, out["maxwell_slow_manifold_(1-U)f"], 0.47, 0.52):
        Us = np.linspace(0.2, 1.0, 300)
        V = [-quad(lambda u: s * (1 - u) * f(u, lam, rho, kap), 1.0, x)[0] for x in Us]
        ax[1].plot(gamma_star(Us, mu, sig), V, label=f"lambda={lam:.3f}")
    ax[1].set_xlabel("<Gamma>"); ax[1].set_ylabel("V (slow-manifold, U-coordinate)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/timescale_potential.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k != "timescale_rows"}, indent=2))


if __name__ == "__main__":
    main()
