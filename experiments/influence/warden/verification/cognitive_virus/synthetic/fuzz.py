"""(ii) Random-parameter fuzz: closed forms vs numerics."""
import json, numpy as np
from tqdm import tqdm
from model import *

rng = np.random.default_rng(20260915)


def draw():
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    p = dict(rho=lu(0.01, 1), kap=lu(0.01, 2), mu=lu(0.01, 1), sig=lu(0.01, 1))
    p["lam"] = float(rng.uniform(0.2, 1.8)) * max(lam_tc(p["rho"], p["kap"]), 1e-3)
    return p


def main(n=1200):
    stats = dict(n=n, max_conservation=0.0, max_eq_residual=0.0, max_dshare_err=0.0, max_gamma_err=0.0,
                 max_jac_eig_err=0.0, EU_stability_mismatch=0, attractor_mismatch=0, U_plus_le_1_violations_in_bistable=0,
                 bistable_draws=0, kap_le_rho_but_bistable=0, skipped_slow=0)
    fails = []
    for _ in tqdm(range(n), desc="fuzz"):
        p = draw(); lam, rho, kap, mu, sig = p["lam"], p["rho"], p["kap"], p["mu"], p["sig"]
        y = rng.dirichlet([1, 1, 1])
        stats["max_conservation"] = max(stats["max_conservation"], abs(sum(rhs_ucd(0, y, lam, rho, kap, mu, sig))))
        r = roots(lam, rho, kap)
        if r is not None:
            for Us in r:
                eq = coupled_eq(Us, mu, sig)
                res = np.max(np.abs(rhs_ucd(0, eq, lam, rho, kap, mu, sig)))
                stats["max_eq_residual"] = max(stats["max_eq_residual"], res / max(1, abs(1 - Us)))
                stats["max_dshare_err"] = max(stats["max_dshare_err"], abs(eq[2] / (eq[1] + eq[2]) - mu / (mu + sig)) if Us != 1 else 0)
                GU = GAMMA["Gu"] * eq[0] + GAMMA["Gc"] * eq[1] + GAMMA["Gd"] * eq[2]
                stats["max_gamma_err"] = max(stats["max_gamma_err"], abs(GU - gamma_star(Us, mu, sig)))
                ev = np.sort(np.linalg.eigvals(jacobian_UC_interior(Us, lam, rho, kap, mu, sig)).real)
                # numerical Jacobian of the full 2D system
                def F(z):
                    U, C = z; return np.array(rhs_ucd(0, [U, C, 1 - U - C], lam, rho, kap, mu, sig)[:2])
                z0 = np.array(eq[:2]); h = 1e-7
                Jn = np.column_stack([(F(z0 + h * e) - F(z0 - h * e)) / (2 * h) for e in np.eye(2)])
                evn = np.sort(np.linalg.eigvals(Jn).real)
                stats["max_jac_eig_err"] = max(stats["max_jac_eig_err"], float(np.max(np.abs(ev - evn))))
        # E_U stability
        ev = np.linalg.eigvals(jacobian_CD_at_EU(lam, rho, kap, mu, sig)).real
        if (np.max(ev) < 0) != (lam < rho + kap): stats["EU_stability_mismatch"] += 1
        bist = physical_coupled(lam, rho, kap) is not None and lam < rho + kap
        if bist:
            stats["bistable_draws"] += 1
            if r[1] > 1: stats["U_plus_le_1_violations_in_bistable"] += 1
            if kap <= rho: stats["kap_le_rho_but_bistable"] += 1
        # attractor prediction vs integration from several ICs (skip near-degenerate cases)
        slow = min(abs(lam - rho - kap), abs(lam**2 - 4 * kap * rho)) / max(lam, 1e-9)
        if slow < 0.03:
            stats["skipped_slow"] += 1; continue
        pred = set()
        if lam < rho + kap: pred.add("EU")
        Um = physical_coupled(lam, rho, kap)
        if Um is not None: pred.add("coupled")
        seen = set()
        for y0 in ([0.999, 0.001, 0], [0.05, 0.9, 0.05], [0.3, 0.3, 0.4], [0.9, 0.0, 0.1], [0.6, 0.2, 0.2]):
            yT = integrate(np.array(y0, float), p, T=4000 / min(1, mu + sig, abs(lam - rho - kap) + 1e-3) if False else 20000)
            if yT[0] > 0.995: seen.add("EU")
            elif Um is not None and abs(yT[0] - Um) < 1e-3: seen.add("coupled")
            else: seen.add(f"other:{yT[0]:.4f}")
        if not seen <= pred:
            stats["attractor_mismatch"] += 1; fails.append(dict(p=p, pred=sorted(pred), seen=sorted(seen)))
    stats["failures_sample"] = fails[:10]
    json.dump(stats, open("results/fuzz.json", "w"), indent=2); print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
