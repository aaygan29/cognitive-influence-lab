"""Drift-diffusion reduction of the value decision, with a learning rule.

The attractor decision circuit used in `bellwether` and `persuasion-control`
reduces, near its symmetric point, to a one-dimensional drift-diffusion process:
evidence accumulates with drift ``v`` toward one of two boundaries separated by
``a``, from a relative start point ``w`` in (0, 1), after a non-decision time
``t0``. The first-passage time and which boundary is hit are the reaction time
and the choice. This module provides:

  * ``wfpt_logpdf`` : the Wiener first-passage-time log-density (Navarro & Fuss,
    2009), in JAX, so it is differentiable for NUTS. This is the analytic
    likelihood the DDM-reduction route buys us.
  * ``simulate_ddm`` : an Euler-Maruyama simulator, used only to validate the
    analytic density against Monte-Carlo and to generate synthetic data with
    known ground truth.
  * ``simulate_subject`` : a trial sequence in which a Rescorla-Wagner learning
    rule reshapes the drift across trials, so a subject is defined by both a
    state vector and a learning rule.

Boundary coding: choice 0 is the lower boundary (0), choice 1 is the upper
boundary (a). The density for the upper boundary is obtained from the lower-
boundary density by the standard reflection ``v -> -v``, ``w -> 1 - w``.

Reference: Navarro D.J., Fuss I.G. (2009). Fast and accurate calculations for
first-passage times in Wiener diffusion models. J. Math. Psychol. 53:222-230.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Fixed term counts. Generous for the typical normalized-time range; the small-
# and large-time expansions are each accurate to well below NUTS noise here.
_SMALL_K = jnp.arange(-4, 5)          # k = -4..4
_LARGE_K = jnp.arange(1, 9)           # k = 1..8
_TAU_SWITCH = 0.5                      # use small-time below, large-time above
_TINY = 1e-12


def _log_ff_small(tau: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """log of the driftless normalized density, small-time expansion."""
    k = _SMALL_K
    arg = w[..., None] + 2.0 * k        # (..., K)
    expo = -(arg ** 2) / (2.0 * tau[..., None])
    m = jnp.max(expo, axis=-1, keepdims=True)
    s = jnp.sum(arg * jnp.exp(expo - m), axis=-1)
    log_pref = -0.5 * jnp.log(2.0 * jnp.pi * tau ** 3)
    return log_pref + m[..., 0] + jnp.log(jnp.clip(s, _TINY))


def _log_ff_large(tau: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """log of the driftless normalized density, large-time expansion."""
    k = _LARGE_K.astype(tau.dtype)
    expo = -(k ** 2) * (jnp.pi ** 2) * tau[..., None] / 2.0
    m = jnp.max(expo, axis=-1, keepdims=True)
    s = jnp.sum(k * jnp.exp(expo - m) * jnp.sin(k * jnp.pi * w[..., None]), axis=-1)
    return jnp.log(jnp.pi) + m[..., 0] + jnp.log(jnp.clip(s, _TINY))


def wfpt_logpdf(rt, choice, v, a, w, t0):
    """Wiener first-passage-time log-density for (rt, choice).

    All arguments broadcast. ``rt`` in seconds, ``choice`` in {0, 1}, ``v`` the
    drift, ``a`` the boundary separation, ``w`` the relative start in (0, 1),
    ``t0`` the non-decision time. Returns -inf where rt <= t0.
    """
    rt = jnp.asarray(rt, dtype=jnp.float32)
    choice = jnp.asarray(choice, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.float32)
    w = jnp.asarray(w, dtype=jnp.float32)
    t0 = jnp.asarray(t0, dtype=jnp.float32)

    # Reflect for the upper boundary so we always compute a lower-boundary hit.
    upper = choice > 0.5
    v_e = jnp.where(upper, -v, v)
    w_e = jnp.where(upper, 1.0 - w, w)

    t = rt - t0
    valid = t > 0.0
    t_safe = jnp.where(valid, t, 1.0)
    tau = t_safe / (a ** 2)

    log_ff = jnp.where(tau < _TAU_SWITCH,
                       _log_ff_small(tau, w_e),
                       _log_ff_large(tau, w_e))
    # Pull the drift back in and undo the a^2 time rescaling.
    log_p = (-v_e * a * w_e - 0.5 * (v_e ** 2) * t_safe - 2.0 * jnp.log(a) + log_ff)
    return jnp.where(valid, log_p, -jnp.inf)


# --------------------------------------------------------------------------- #
# Simulator (numpy): ground truth for validation and synthetic data.
# --------------------------------------------------------------------------- #

def simulate_ddm(v, a, w, t0, *, n, dt=1e-4, max_t=5.0, rng=None):
    """Euler-Maruyama first passage. Returns (rt, choice) arrays of length n."""
    rng = rng or np.random.default_rng(0)
    x = np.full(n, w * a)
    t = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    choice = np.full(n, -1)
    rt = np.full(n, np.nan)
    sq = np.sqrt(dt)
    steps = int(max_t / dt)
    for _ in range(steps):
        live = ~done
        if not live.any():
            break
        x[live] += v * dt + sq * rng.standard_normal(live.sum())
        t[live] += dt
        hi = live & (x >= a)
        lo = live & (x <= 0.0)
        choice[hi] = 1
        rt[hi] = t[hi] + t0
        choice[lo] = 0
        rt[lo] = t[lo] + t0
        done |= hi | lo
    return rt, choice


def simulate_subject(theta, phi, *, n_trials, seed=0, reversal_every=40,
                     p_good=0.8, p_bad=0.2):
    """One subject's trial sequence on a reversal bandit, with a learning rule.

    This is a value-learning task so the learning rule is identifiable: there is
    no strong perceptual evidence that would swamp the learned value. Two options
    (0 and 1); one is "good" (reward probability ``p_good``) and reverses every
    ``reversal_every`` trials. The subject tracks each option's value with a
    Rescorla-Wagner rule at rate ``alpha`` and chooses via the drift-diffusion
    process, with drift

        v_t = w_q * (Q1 - Q0)  +  beta * cue_t,

    where ``cue_t`` in {-1, +1} is a weak perceptual hint about the good option.
    ``theta`` = dict(a, t0, w, beta) is the state vector; ``phi`` = dict(alpha,
    w_q) is the learning rule (rate and value-to-drift coupling).
    """
    rng = np.random.default_rng(seed)
    a, t0, w, beta = theta["a"], theta["t0"], theta["w"], theta["beta"]
    alpha, w_q = phi["alpha"], phi["w_q"]

    Q = np.array([0.5, 0.5])
    rts, choices, rewards, cues, Qdiffs = [], [], [], [], []
    for i in range(n_trials):
        good = (i // reversal_every) % 2            # which option is good now
        cue = 1.0 if good == 1 else -1.0
        cue = cue if rng.random() < 0.7 else -cue   # cue is noisy (70% valid)
        v_t = w_q * (Q[1] - Q[0]) + beta * cue
        rt, ch = simulate_ddm(v_t, a, w, t0, n=1, rng=rng)
        ch = int(ch[0]); rt = float(rt[0])
        p = p_good if ch == good else p_bad
        reward = float(rng.random() < p)
        Qdiffs.append(Q[1] - Q[0])
        Q[ch] = Q[ch] + alpha * (reward - Q[ch])
        rts.append(rt); choices.append(ch); rewards.append(reward); cues.append(cue)
    return {
        "rt": np.array(rts), "choice": np.array(choices),
        "reward": np.array(rewards), "cue": np.array(cues),
        "Qdiff": np.array(Qdiffs),
    }
