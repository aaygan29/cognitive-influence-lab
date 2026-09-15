"""(v) Table I sanity: signs of finite-difference effects of each parameter."""
import json, numpy as np
from model import *

G = GAMMA


def summary(p):
    lam, rho, kap, mu, sig = p["lam"], p["rho"], p["kap"], p["mu"], p["sig"]
    Um = physical_coupled(lam, rho, kap)
    Us = Um if Um is not None else 1.0
    U, C, D = coupled_eq(Us, mu, sig) if Um is not None else (1.0, 0.0, 0.0)
    return dict(lam_SN=lam_sn(rho, kap), lam_TC=lam_tc(rho, kap), width=lam_tc(rho, kap) - lam_sn(rho, kap),
                bistable=(kap > rho), U_coupled=Us, D_share=(mu / (mu + sig)), Gamma_coupled=G["Gu"] * U + G["Gc"] * C + G["Gd"] * D,
                EU_invadable=lam >= rho + kap)


def main():
    base = dict(BASE); b = summary(base); h = 1e-3
    eff = {}
    for k in ["lam", "rho", "kap", "mu", "sig"]:
        q = dict(base); q[k] += h; s = summary(q)
        eff[k] = {m: float(np.sign(round((s[m] - b[m]) / h, 9))) for m in ("lam_SN", "lam_TC", "width", "U_coupled", "D_share", "Gamma_coupled")}
    checks = {
        "decrease lam: no threshold shift, prevents invasion only below lam_TC": eff["lam"]["lam_SN"] == 0 and eff["lam"]["lam_TC"] == 0,
        "increase rho raises both thresholds": eff["rho"]["lam_SN"] > 0 and eff["rho"]["lam_TC"] > 0,
        "increase rho reduces hysteresis width (kap>rho)": eff["rho"]["width"] < 0,
        "increase kap raises lam_TC": eff["kap"]["lam_TC"] > 0,
        "increase kap widens hysteresis (kap>rho)": eff["kap"]["width"] > 0,
        "mu, sig do not move thresholds": all(eff[k][m] == 0 for k in ("mu", "sig") for m in ("lam_SN", "lam_TC")),
        "decrease mu reduces D share": eff["mu"]["D_share"] > 0,
        "decrease mu raises Gamma (needs Gc>Gd)": eff["mu"]["Gamma_coupled"] < 0,
        "increase sig reduces D share and raises Gamma at fixed U*": eff["sig"]["D_share"] < 0 and eff["sig"]["Gamma_coupled"] > 0 and eff["sig"]["U_coupled"] == 0,
        "bistability disappears for rho>=kap": summary(dict(base, rho=0.4))["width"] <= 1e-12 and summary(dict(base, rho=0.5))["bistable"] is False,
    }
    # counter-check: if Gc < Gd (a scaffolding regime) the mu/sigma Gamma effect reverses
    g1 = g_of(0.2, 0.1, Gc=0.1, Gd=0.5); g2 = g_of(0.2, 0.1 + h, Gc=0.1, Gd=0.5)
    out = dict(base=base, base_summary=b, finite_difference_signs=eff, table_checks=checks,
               sigma_effect_on_g_if_Gc_lt_Gd=float(np.sign(g2 - g1)))
    json.dump(out, open("results/interventions.json", "w"), indent=2, default=float); print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
