"""The formal belief-dynamics model (paper/MODEL.md), made runnable.

Implements the cusp belief potential, the gated/anchored dynamics, the robustness functionals, and the
four falsifiable predictions: hysteresis, sequential-beats-single-shot, super-linear robustness scaling,
and the susceptibility curve. This is a low-dimensional theoretical model; it stands on its own as a
formal object and connects to the instrument through the persuasive-force term beta = w_c*phi + w_n*B.

No em dashes.
"""

from __future__ import annotations

import numpy as np


# ---------- potential, force, equilibria ----------

def dV_db(b, alpha, beta):
    """Gradient of the cusp potential V (Eq. 1-2). Stable belief = root with -dV/db = 0, V'' > 0."""
    return b ** 3 - alpha * b - beta


def force(b, alpha, beta):
    return -dV_db(b, alpha, beta)


def stiffness(b, alpha):
    """Local robustness rho_local = V''(b) = 3 b^2 - alpha (Eq. 9)."""
    return 3.0 * b ** 2 - alpha


def equilibria(alpha, beta):
    """Real roots of b^3 - alpha b - beta = 0; return stable ones (V'' > 0), sorted."""
    roots = np.roots([1.0, 0.0, -alpha, -beta])
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    stable = [b for b in real if stiffness(b, alpha) > 0]
    return sorted(stable)


def beta_crit(alpha):
    """Critical force at the fold (Eq. 11); 0 for alpha <= 0 (no barrier).

    Fold set of b^3 - alpha b - beta = 0 is the double-root locus 4 alpha^3 = 27 beta^2, so
    beta_crit = sqrt(4 alpha^3 / 27) = (2 / (3 sqrt 3)) alpha^{3/2}.
    """
    return np.sqrt((4.0 / 27.0) * alpha ** 3) if alpha > 0 else 0.0


# ---------- simulation of the gated, anchored dynamics (Eq. 7) ----------

def simulate(
    b_init,
    alpha,
    messages,           # list of (pos, beta_raw) per step
    b0=None,            # FJ anchor; defaults to b_init
    s=0.7,              # susceptibility (1 open, 0 stubborn)
    kappa=1.0,
    d=0.6,              # bounded-confidence band (Eq. 4)
    lam_plus=1.0,       # confirming learning rate (Eq. 5)
    lam_minus=0.5,      # disconfirming learning rate
    dt=0.05,
    steps_per_msg=40,
    sigma=0.0,
    seed=0,
):
    """Integrate Eq. 7 under a message sequence. Returns the belief trajectory."""
    rng = np.random.default_rng(seed)
    b = float(b_init)
    b0 = float(b_init if b0 is None else b0)
    traj = [b]
    for pos, beta_raw in messages:
        for _ in range(steps_per_msg):
            in_band = 1.0 if abs(pos - b) <= d else 0.0
            toward_pole = np.sign(pos - b) == np.sign(b - b0) or abs(b - b0) < 1e-9
            a = lam_plus if toward_pole else lam_minus
            beta_eff = a * in_band * beta_raw
            db = (-(b ** 3) + alpha * b + s * beta_eff + (1 - s) * kappa * (b0 - b))
            b = b + dt * db + (sigma * np.sqrt(dt) * rng.normal() if sigma > 0 else 0.0)
            traj.append(b)
    return np.asarray(traj)


# ---------- the falsifiable predictions ----------

def pred_hysteresis(alpha=1.2, beta_max=1.2, n=60):
    """Sweep beta up then down; return (betas_up, b_up, betas_down, b_down). Loop area > 0 = hysteresis."""
    betas_up = np.linspace(-beta_max, beta_max, n)
    betas_down = betas_up[::-1]
    b = -1.5
    up = []
    for be in betas_up:
        for _ in range(200):
            b += 0.02 * force(b, alpha, be)
        up.append(b)
    down = []
    for be in betas_down:
        for _ in range(200):
            b += 0.02 * force(b, alpha, be)
        down.append(b)
    return betas_up, np.array(up), betas_down, np.array(down)


def pred_sequential_vs_single(alpha=0.5, target=1.6, total_strength=3.0, d=0.6,
                              s=0.9, kappa=0.3, K=8):
    """Adaptive sequential in-band persuasion vs one large out-of-band step of equal total strength.

    The sequential policy is the foot-in-the-door optimum: each message advocates a position just inside
    the confidence band of the CURRENT belief, so the gate (Eq. 4) stays open as the belief is walked
    across the barrier. The single message advocates the full distance at once and falls outside the band.
    Returns (b_final_sequential, b_final_single). Prediction: sequential ends much closer to target.
    """
    b_start = -1.2
    # single: one out-of-band message with all the strength
    single = simulate(b_init=b_start, alpha=alpha, d=d, s=s, kappa=kappa,
                      messages=[(target, total_strength)])

    # sequential: place each message adaptively just inside the band of the current belief
    b = b_start
    per = total_strength / K
    traj_end = b
    for _ in range(K):
        pos = min(target, b + 0.9 * d)
        out = simulate(b_init=b, alpha=alpha, d=d, s=s, kappa=kappa, b0=b,
                      messages=[(pos, per)])
        b = float(out[-1])
        traj_end = b
    return float(traj_end), float(single[-1])


def pred_robustness_scaling(alphas=(0.2, 0.4, 0.8, 1.2, 1.6)):
    """beta_crit vs alpha; should scale as alpha^{3/2} (Eq. 11)."""
    a = np.array(alphas, dtype=float)
    bc = np.array([beta_crit(x) for x in a])
    # fit log bc = c + p log a; p should be ~1.5
    mask = bc > 0
    p = np.polyfit(np.log(a[mask]), np.log(bc[mask]), 1)[0]
    return a, bc, float(p)


def susceptibility_curve(alpha=1.0, betas=None):
    """chi = 1/(3 b*^2 - alpha) along the METASTABLE branch being destabilized; diverges at the fold.

    For beta > 0 the lower (metastable) well is the one that vanishes at the fold. We follow that branch by
    continuation from beta = 0, choosing at each step the stable root closest to the previous one. As beta
    approaches beta_crit the well stiffness goes to zero and chi diverges (Eq. 10).
    """
    if betas is None:
        top = beta_crit(alpha) * 0.999 if alpha > 0 else 1.0
        betas = np.linspace(0.0, top, 60)
    out = []
    for be in betas:
        eqs = equilibria(alpha, be)
        if not eqs:
            continue
        # for beta > 0 the metastable (vanishing) well is the smallest-b stable root
        b_star = eqs[0]
        out.append((float(be), 1.0 / stiffness(b_star, alpha)))
    return np.array(out)
