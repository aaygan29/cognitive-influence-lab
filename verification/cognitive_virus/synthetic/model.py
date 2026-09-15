"""Mean-field model of Sole et al. (arXiv:2609.03344), Eqs. (1)-(3), plus closed forms."""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(lam=0.45, rho=0.10, kap=0.40, mu=0.20, sig=0.10)
GAMMA = dict(Gu=1.0, Gc=0.5, Gd=0.1)


def rhs_ucd(t, y, lam, rho, kap, mu, sig):
    U, C, D = y
    dU = -lam * U * C + rho * C + kap * U**2 * C
    dC = lam * U * C - (mu + rho) * C + sig * D - kap * U**2 * C
    dD = mu * C - sig * D
    return [dU, dC, dD]


def f(U, lam, rho, kap):
    return rho - lam * U + kap * U**2


def roots(lam, rho, kap):
    disc = lam**2 - 4 * kap * rho
    if disc < 0:
        return None
    s = np.sqrt(disc)
    return (lam - s) / (2 * kap), (lam + s) / (2 * kap)


def lam_sn(rho, kap):
    return 2 * np.sqrt(kap * rho)


def lam_tc(rho, kap):
    return rho + kap


def coupled_eq(Ustar, mu, sig):
    return Ustar, sig / (mu + sig) * (1 - Ustar), mu / (mu + sig) * (1 - Ustar)


def g_of(mu, sig, Gc=GAMMA["Gc"], Gd=GAMMA["Gd"]):
    return (Gc * sig + Gd * mu) / (mu + sig)


def gamma_star(U, mu, sig, **G):
    g = g_of(mu, sig, **G)
    return g + (1 - g) * U


def physical_coupled(lam, rho, kap):
    """Return the stable coupled U- if it lies in (0,1), else None."""
    r = roots(lam, rho, kap)
    if r is None:
        return None
    Um = r[0]
    return Um if 0 < Um < 1 else None


def integrate(y0, p, T=4000.0, rtol=1e-9, atol=1e-12):
    sol = solve_ivp(rhs_ucd, (0, T), y0, args=(p["lam"], p["rho"], p["kap"], p["mu"], p["sig"]),
                    method="LSODA", rtol=rtol, atol=atol)
    y = sol.y[:, -1]
    return np.clip(y, 0, 1) / np.clip(y, 0, 1).sum()


def jacobian_CD_at_EU(lam, rho, kap, mu, sig):
    return np.array([[lam - rho - kap - mu, sig], [mu, -sig]])


def jacobian_UC_interior(Ustar, lam, rho, kap, mu, sig):
    C = sig / (mu + sig) * (1 - Ustar)
    fp = -lam + 2 * kap * Ustar
    return np.array([[C * fp, 0.0], [-C * fp - sig, -(mu + sig)]])
