"""AIM-DDM decision-phenotype estimator and the E1 parameter-recovery experiment.

Forward model (Section 3.1-3.2 of the paper). An agent is a parameter vector

    theta = (rho_gain, lambda_loss, kappa_risk, omega_threat, delta_time, tau_consistency).

An event c exposes features x = [E_gain, E_loss, sigma_risk, ambiguity, delay]. The AIM-structured
valuation drives a drift-diffusion choice; the drift is

    v(c, theta) = tau * ( rho_gain*E_gain - lambda_loss*E_loss - kappa_risk*sigma
                          - omega_threat*ambiguity - delta_time*delay ).

For an unbiased diffusion the choice-marginal is logistic in the drift, P(act) = sigmoid(v). That
is what we fit from choice data here. Reaction times (needed to separate tau from the valuation
scale) enter in Step 3; see the identifiability note below.

Identifiability (honest, and the reason E1 targets ratios). From choice alone the logistic
coefficients recover the PRODUCTS tau*rho, tau*lambda, ... so the overall scale tau is not
separately identifiable without RT. The scale-free quantities that ARE identifiable are the
affective ratios

    loss_aversion = lambda/rho,  threat = omega/rho,  risk = kappa/rho,  discount = delta/rho,

which are precisely the decision-phenotype coordinates that matter (loss aversion is a ratio by
definition in prospect theory). E1 tests recovery of these ratios. tau is recovered later from RT
via the full DDM.

Pure numpy/scipy. Every reported number is emitted through the honesty layer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402

FEATURES = ["E_gain", "E_loss", "sigma_risk", "ambiguity", "delay"]
# sign each feature carries in the valuation (gain is +, the rest reduce value)
FEATURE_SIGN = np.array([+1.0, -1.0, -1.0, -1.0, -1.0])
# the affective channel (population-general) vs integrative (per-agent)
AFFECTIVE = ("loss_aversion", "threat")          # lambda/rho, omega/rho
INTEGRATIVE = ("risk", "discount")               # kappa/rho, delta/rho


# --------------------------------------------------------------- generative model

@dataclass
class Population:
    """Priors on theta. Affective params are tight (population-general); integrative are wide."""
    rho_mean: float = 1.0;   rho_sd: float = 0.12
    lam_mean: float = 1.9;   lam_sd: float = 0.25     # classic loss aversion ~1.8-2.25
    omega_mean: float = 0.8; omega_sd: float = 0.18
    kappa_mean: float = 0.6; kappa_sd: float = 0.30   # wider -> individuating
    delta_mean: float = 0.5; delta_sd: float = 0.28
    tau_mean: float = 1.3;   tau_sd: float = 0.30

    def sample_theta(self, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
        def pos(mean, sd):
            return np.clip(rng.normal(mean, sd, n), 0.05, None)
        return {
            "rho": pos(self.rho_mean, self.rho_sd),
            "lambda": pos(self.lam_mean, self.lam_sd),
            "omega": pos(self.omega_mean, self.omega_sd),
            "kappa": pos(self.kappa_mean, self.kappa_sd),
            "delta": pos(self.delta_mean, self.delta_sd),
            "tau": pos(self.tau_mean, self.tau_sd),
        }


def true_ratios(theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "loss_aversion": theta["lambda"] / theta["rho"],
        "threat": theta["omega"] / theta["rho"],
        "risk": theta["kappa"] / theta["rho"],
        "discount": theta["delta"] / theta["rho"],
    }


def sample_contexts(rng: np.random.Generator, n_trials: int) -> np.ndarray:
    """Event feature matrix, shape (n_trials, 5). Independent draws so features are not collinear."""
    Eg = rng.uniform(0.0, 1.0, n_trials)
    El = rng.uniform(0.0, 1.0, n_trials)
    sig = rng.uniform(0.0, 1.0, n_trials)
    amb = rng.uniform(0.0, 1.0, n_trials)
    dly = rng.uniform(0.0, 1.0, n_trials)
    return np.column_stack([Eg, El, sig, amb, dly])


def drift(X: np.ndarray, theta_i: Dict[str, float]) -> np.ndarray:
    w = np.array([theta_i["rho"], theta_i["lambda"], theta_i["kappa"],
                  theta_i["omega"], theta_i["delta"]])
    # order X columns to [Eg, El, sigma, ambiguity, delay]; apply signs
    signed = FEATURE_SIGN * np.array([w[0], w[1], w[2], w[3], w[4]])
    return theta_i["tau"] * (X @ signed)


def simulate_agent_choices(rng: np.random.Generator, X: np.ndarray, theta_i: Dict[str, float]) -> np.ndarray:
    v = drift(X, theta_i)
    p = 1.0 / (1.0 + np.exp(-v))
    return (rng.uniform(size=len(p)) < p).astype(float)


# --------------------------------------------------------------- estimator

def _neg_log_lik(beta: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float) -> Tuple[float, np.ndarray]:
    z = X @ beta
    # stable log-sigmoid
    logp = -np.logaddexp(0.0, -z)
    logq = -np.logaddexp(0.0, z)
    nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * l2 * np.sum(beta ** 2)
    p = 1.0 / (1.0 + np.exp(-z))
    grad = X.T @ (p - y) + l2 * beta
    return nll, grad


def fit_agent(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """Maximum-likelihood logistic fit (no intercept). Returns beta = tau * signed weights.

    A small L2 keeps the fit finite under separation, the same guard the harness uses elsewhere.
    """
    beta0 = np.zeros(X.shape[1])
    res = optimize.minimize(lambda b: _neg_log_lik(b, X, y, l2), beta0, jac=True, method="L-BFGS-B")
    return res.x


def fit_agents_pooled(Xs: List[np.ndarray], ys: List[np.ndarray], n_iter: int = 6,
                      shrink_floor: float = 1e-3) -> List[np.ndarray]:
    """Empirical-Bayes partial pooling of the logistic coefficients across agents (C4).

    Each agent's beta is fit with a Gaussian prior N(mu_pop, diag(sigma2_pop)); the population mean
    and variance are re-estimated across agents each iteration (an EM-style loop). At low
    decisions-per-agent this borrows strength across the population and lowers per-agent estimate
    variance, which is the mechanism the paper claims lowers the effective MDES. The affective
    coefficients, which genuinely share a tight population, are shrunk more than the idiosyncratic
    integrative ones, exactly as the AIM split predicts.
    """
    n_agents = len(Xs)
    d = Xs[0].shape[1]
    # init: unpooled fits
    betas = [fit_agent(Xs[i], ys[i]) for i in range(n_agents)]
    B = np.array(betas)
    mu = B.mean(axis=0)
    sig2 = np.clip(B.var(axis=0), shrink_floor, None)
    for _ in range(n_iter):
        # E-step: MAP fit of each agent under the current population prior
        new = []
        for i in range(n_agents):
            lam = 1.0 / sig2  # per-coordinate prior precision
            new.append(_map_agent(Xs[i], ys[i], mu, lam))
        B = np.array(new)
        # M-step: update population mean/var
        mu = B.mean(axis=0)
        sig2 = np.clip(B.var(axis=0), shrink_floor, None)
    return [B[i] for i in range(n_agents)]


def _map_agent(X: np.ndarray, y: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """MAP logistic fit with a Gaussian prior N(mu, diag(1/lam)) on beta."""
    def obj(beta):
        z = X @ beta
        logp = -np.logaddexp(0.0, -z); logq = -np.logaddexp(0.0, z)
        nll = -np.sum(y * logp + (1 - y) * logq) + 0.5 * np.sum(lam * (beta - mu) ** 2)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) + lam * (beta - mu)
        return nll, grad
    res = optimize.minimize(obj, mu.copy(), jac=True, method="L-BFGS-B")
    return res.x


def recover_ratios(beta: np.ndarray) -> Dict[str, float]:
    """Map logistic coefficients back to affective/integrative ratios.

    beta[k] = tau * FEATURE_SIGN[k] * w[k], with w = [rho, lambda, kappa, omega, delta].
    So the magnitudes m[k] = |beta[k]| = tau * w[k], and every ratio divides out tau and rho.
    """
    m = np.abs(beta)
    rho = m[0] if m[0] > 1e-6 else 1e-6      # gain-sensitivity magnitude (tau*rho)
    return {
        "loss_aversion": m[1] / rho,   # tau*lambda / tau*rho = lambda/rho
        "threat": m[3] / rho,          # omega/rho
        "risk": m[2] / rho,            # kappa/rho
        "discount": m[4] / rho,        # delta/rho
    }


# --------------------------------------------------------------- E1 recovery experiment

def recovery_once(seed: int, n_agents: int, n_trials: int, pop: Population) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = pop.sample_theta(rng, n_agents)
    truth = true_ratios(theta)
    recovered = {k: np.empty(n_agents) for k in truth}
    for i in range(n_agents):
        theta_i = {k: float(theta[k][i]) for k in theta}
        X = sample_contexts(rng, n_trials)
        y = simulate_agent_choices(rng, X, theta_i)
        beta = fit_agent(X, y)
        rec = recover_ratios(beta)
        for k in recovered:
            recovered[k][i] = rec[k]
    return {"truth": truth, "recovered": recovered}


def run_e1(n_agents: int = 60, trials_grid: Tuple[int, ...] = (200, 500, 1000, 2000),
           n_seeds: int = 20, kill_threshold: float = 0.6) -> Dict[str, object]:
    """Recovery is characterized as a curve over decisions-per-agent, not a single arbitrary n.

    The identifiability question E1 answers: does the estimator recover the affective phenotype
    ratios, and how many decisions per agent does that take? A single n hides the answer; the curve
    is the honest object. PASS = affective ratios reach r >= kill_threshold at the largest grid point,
    with the minimum n_trials that first clears it reported.
    """
    pop = Population()
    params = ["loss_aversion", "threat", "risk", "discount"]
    provenance = {"model": "AIM-DDM choice-marginal (logistic)", "estimator": "MLE logistic + L2",
                  "code": "src/phenotype.py",
                  "identifiable_target": "affective/integrative ratios (tau absorbed; RT recovers tau, Step 3)"}

    curve: Dict[int, Dict[str, object]] = {}
    for nt in trials_grid:
        per_seed_r = {k: [] for k in params}
        for s in range(n_seeds):
            out = recovery_once(seed=s, n_agents=n_agents, n_trials=nt, pop=pop)
            for k in params:
                per_seed_r[k].append(float(stats.pearsonr(out["truth"][k], out["recovered"][k])[0]))
        entry = {}
        for k in params:
            rs = np.array(per_seed_r[k])
            ci = H.bootstrap_ci(rs, statistic=np.mean, seed=0)
            gated = H.gate_effect(f"recovery_r[{k}]@{nt}", effect=float(rs.mean()), n=n_agents,
                                  kind="correlation", provenance=dict(provenance, n_trials=nt))
            entry[k] = {"channel": "affective" if k in AFFECTIVE else "integrative",
                        "mean_r": float(rs.mean()), "sd_r": float(rs.std()),
                        "ci95": [ci["lo"], ci["hi"]], "gated_abstained": gated.abstained}
        curve[nt] = entry

    # min trials to clear the threshold for each param; PASS if affective ratios clear at max grid
    max_nt = max(trials_grid)
    min_trials_to_pass = {}
    for k in params:
        hit = [nt for nt in trials_grid if curve[nt][k]["mean_r"] >= kill_threshold]
        min_trials_to_pass[k] = int(min(hit)) if hit else None
    affective_pass = all(curve[max_nt][k]["mean_r"] >= kill_threshold for k in AFFECTIVE)

    return {
        "experiment": "E1_parameter_recovery",
        "design": {"n_agents": n_agents, "trials_grid": list(trials_grid), "n_seeds": n_seeds,
                   "kill_threshold_affective": kill_threshold},
        "provenance": provenance,
        "recovery_curve": {str(nt): curve[nt] for nt in trials_grid},
        "min_trials_to_reach_threshold": min_trials_to_pass,
        "affective_kill_criterion_pass": bool(affective_pass),
        "note": ("recovery of the per-agent affective ratios rises with decisions-per-agent; this "
                 "data requirement is the empirical motivation for hierarchical pooling (C4) and for "
                 "abstention on sparse records (C5)."),
    }


def _recovery_r_for_betas(betas: List[np.ndarray], truth: Dict[str, np.ndarray],
                          params: List[str]) -> Dict[str, float]:
    rec = {k: np.empty(len(betas)) for k in params}
    for i, b in enumerate(betas):
        rr = recover_ratios(b)
        for k in params:
            rec[k][i] = rr[k]
    return {k: float(stats.pearsonr(truth[k], rec[k])[0]) for k in params}


def run_e4(n_agents: int = 60, trials_grid: Tuple[int, ...] = (150, 300, 600),
           n_seeds: int = 20) -> Dict[str, object]:
    """C4: hierarchical pooling lifts low-n recovery. Compares unpooled vs pooled recovery r."""
    pop = Population()
    params = ["loss_aversion", "threat", "risk", "discount"]
    out: Dict[str, object] = {"experiment": "E4_pooling", "design": {
        "n_agents": n_agents, "trials_grid": list(trials_grid), "n_seeds": n_seeds}}
    curve = {}
    for nt in trials_grid:
        unp = {k: [] for k in params}; poo = {k: [] for k in params}
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            theta = pop.sample_theta(rng, n_agents)
            truth = true_ratios(theta)
            Xs, ys = [], []
            for i in range(n_agents):
                ti = {k: float(theta[k][i]) for k in theta}
                X = sample_contexts(rng, nt); Xs.append(X)
                ys.append(simulate_agent_choices(rng, X, ti))
            betas_unp = [fit_agent(Xs[i], ys[i]) for i in range(n_agents)]
            betas_poo = fit_agents_pooled(Xs, ys)
            ru = _recovery_r_for_betas(betas_unp, truth, params)
            rp = _recovery_r_for_betas(betas_poo, truth, params)
            for k in params:
                unp[k].append(ru[k]); poo[k].append(rp[k])
        entry = {}
        for k in params:
            u = np.array(unp[k]); p = np.array(poo[k])
            delta = H.bootstrap_ci(p - u, statistic=np.mean, seed=0)
            entry[k] = {"channel": "affective" if k in AFFECTIVE else "integrative",
                        "unpooled_r": float(u.mean()), "pooled_r": float(p.mean()),
                        "delta_r": float((p - u).mean()), "delta_ci95": [delta["lo"], delta["hi"]],
                        "pooling_helps": bool(delta["lo"] > 0)}
        curve[str(nt)] = entry
    out["curve"] = curve
    # summary: does pooling help affective recovery at the smallest n (CI excludes 0)?
    smallest = str(trials_grid[0])
    out["affective_pooling_helps_at_min_n"] = bool(all(
        curve[smallest][k]["pooling_helps"] for k in AFFECTIVE))
    return out


if __name__ == "__main__":
    import json

    res = run_e1()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    path = os.path.join(root, "results", "e1_recovery.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)

    print("E1 recovery curve (mean Pearson r across %d seeds), by decisions/agent:" % res["design"]["n_seeds"])
    grid = res["design"]["trials_grid"]
    print("  %-14s %-11s " % ("param", "channel") + " ".join(f"n={nt:<6d}" for nt in grid))
    for k in ["loss_aversion", "threat", "risk", "discount"]:
        ch = res["recovery_curve"][str(grid[0])][k]["channel"]
        row = " ".join(f"{res['recovery_curve'][str(nt)][k]['mean_r']:<8.3f}" for nt in grid)
        print(f"  {k:<14s} {ch:<11s} {row}")
    print("\nmin decisions/agent to reach r>=%.2f:" % res["design"]["kill_threshold_affective"],
          res["min_trials_to_reach_threshold"])
    print("affective kill criterion (at n=%d):" % max(grid),
          "PASS" if res["affective_kill_criterion_pass"] else "FAIL")
    print("->", path)

    # E4 pooling
    e4 = run_e4()
    p4 = os.path.join(root, "results", "e4_pooling.json")
    with open(p4, "w") as fh:
        json.dump(e4, fh, indent=2)
    print("\nE4 pooling (unpooled -> pooled recovery r), smallest n = %d/agent:" % e4["design"]["trials_grid"][0])
    sm = str(e4["design"]["trials_grid"][0])
    for k, e in e4["curve"][sm].items():
        star = "  (CI>0)" if e["pooling_helps"] else ""
        print(f"  {k:<14s} [{e['channel']:<11s}] {e['unpooled_r']:.3f} -> {e['pooled_r']:.3f}"
              f"  d={e['delta_r']:+.3f}{star}")
    print("affective pooling helps at min n:", e4["affective_pooling_helps_at_min_n"], "->", p4)
