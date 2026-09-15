"""(i) Continuation sweeps of lambda up then down; measure jump points vs lambda_SN, lambda_TC."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tqdm import tqdm
from model import integrate, lam_sn, lam_tc, physical_coupled

EPS = 1e-3  # perturbation injected at each step so a marginally unstable E_U can be left


def sweep(rho, kap, mu=0.2, sig=0.1, n=161):
    lo, hi = 0.5 * min(lam_sn(rho, kap), lam_tc(rho, kap)), 1.5 * max(lam_sn(rho, kap), lam_tc(rho, kap))
    lams = np.linspace(lo, hi, n)
    y = np.array([1 - EPS, EPS, 0.0]); up = []
    for lam in lams:
        y = integrate(y + np.array([-EPS, EPS, 0]) * (y[0] > 1 - EPS), dict(lam=lam, rho=rho, kap=kap, mu=mu, sig=sig), T=6000)
        up.append(y[0])
    down = []
    for lam in lams[::-1]:
        y = integrate(y, dict(lam=lam, rho=rho, kap=kap, mu=mu, sig=sig), T=6000)
        down.append(y[0])
    return lams, np.array(up), np.array(down[::-1])


def jump(lams, U, thr=0.98, direction="up"):
    hi = U > thr
    if direction == "up":
        idx = np.where(~hi)[0]
        return float(lams[idx[0]]) if len(idx) and hi[0] else None
    # down sweep stored in ascending-lambda order: return point = largest lambda at which U is back near 1
    idx = np.where(hi)[0]
    return float(lams[idx[-1]]) if len(idx) else None


def main():
    grid = [(0.1, 0.4), (0.1, 0.2), (0.05, 0.5), (0.2, 0.3), (0.1, 0.1), (0.2, 0.2), (0.3, 0.1), (0.4, 0.1), (0.2, 0.05)]
    out = []
    fig, axes = plt.subplots(3, 3, figsize=(11, 9)); axes = axes.ravel()
    for k, (rho, kap) in enumerate(tqdm(grid, desc="hysteresis grid")):
        lams, up, down = sweep(rho, kap)
        dl = lams[1] - lams[0]
        lu, ld = jump(lams, up, direction="up"), jump(lams, down, direction="down")
        width = (lu - ld) if (lu is not None and ld is not None) else None
        # smooth (no-jump) case: max step change
        max_step = float(np.max(np.abs(np.diff(up))))
        rec = dict(rho=rho, kap=kap, regime=("kap>rho" if kap > rho else "kap=rho" if kap == rho else "kap<rho"),
                   lam_SN=float(lam_sn(rho, kap)), lam_TC=float(lam_tc(rho, kap)), theory_width=float((np.sqrt(kap) - np.sqrt(rho))**2) if kap > rho else 0.0,
                   measured_leave_U1_up=lu, measured_return_U1_down=ld, measured_width=width, grid_step=float(dl),
                   max_abs_dU_between_steps_up=max_step,
                   up_down_max_gap=float(np.max(np.abs(up - down))),
                   lams=lams.tolist(), U_up=up.tolist(), U_down=down.tolist())
        out.append(rec)
        ax = axes[k]
        mf = [physical_coupled(l, rho, kap) for l in lams]
        ax.plot(lams, up, "b-", label="sweep up"); ax.plot(lams, down, "r--", label="sweep down")
        ax.plot(lams, [m if m is not None else np.nan for m in mf], "k:", lw=1, label="U- theory")
        ax.axvline(lam_sn(rho, kap), color="gray", lw=0.6); ax.axvline(lam_tc(rho, kap), color="gray", lw=0.6, ls="--")
        ax.set_title(f"rho={rho}, kappa={kap}", fontsize=9); ax.set_ylim(-0.02, 1.05)
        if k == 0: ax.legend(fontsize=7)
    for ax in axes[6:]: ax.set_xlabel("lambda")
    fig.suptitle("Continuation sweeps (solid gray: lambda_SN, dashed: lambda_TC)"); fig.tight_layout()
    fig.savefig("figures/hysteresis_grid.png", dpi=130)
    json.dump(out, open("results/hysteresis.json", "w"), indent=2)
    for r in out: print({k: v for k, v in r.items() if k not in ('lams', 'U_up', 'U_down')})


if __name__ == "__main__":
    main()
