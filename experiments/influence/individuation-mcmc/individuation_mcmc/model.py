"""Hierarchical Bayesian model: per-person state params and learning rule.

Each subject j has a state vector theta_j = (a_j, t0_j, w_j, beta_j) and a
learning rule phi_j = (alpha_j, w_q_j). Across trials a Rescorla-Wagner value
Q evolves from observed rewards at rate alpha_j and tilts the drift through
w_q_j, so choices and reaction times carry information about both who the
subject is (theta_j) and how they learn (phi_j). The likelihood is the analytic
Wiener first-passage density (ddm.wfpt_logpdf), which makes the whole model
differentiable and so fit by NUTS.

Two model forms share this code:
  * individual (default): subject parameters are drawn from group priors but are
    free to differ (non-centered hierarchy).
  * pooled: every subject shares one parameter vector. This is the group model
    the individual model must beat out-of-sample for individuation to mean
    anything.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .ddm import wfpt_logpdf


def _rw_value_diff(alpha, choice, reward):
    """Value difference (Q1 - Q0) available to the drift on each trial.

    Two-armed reversal bandit. ``alpha`` is (J,); ``choice`` and ``reward`` are
    (J, T), observed. Q for each arm starts at 0.5 and only the chosen arm is
    updated. Returns (J, T): the value difference Q1 - Q0 that trial t's drift
    sees (state before trial t's own update). Deterministic given alpha and the
    observed choices and rewards, which is what makes alpha identifiable.
    """
    def step(carry, inp):
        Q0, Q1 = carry
        c_t, r_t = inp                       # (J,), (J,)
        qdiff = Q1 - Q0
        chose1 = c_t
        chose0 = 1.0 - c_t
        Q1_next = Q1 + alpha * chose1 * (r_t - Q1)
        Q0_next = Q0 + alpha * chose0 * (r_t - Q0)
        return (Q0_next, Q1_next), qdiff
    choice_tT = jnp.swapaxes(choice, 0, 1)          # (T, J)
    reward_tT = jnp.swapaxes(reward, 0, 1)
    half = 0.5 * jnp.ones_like(alpha)
    _, qdiff_tT = jax.lax.scan(step, (half, half), (choice_tT, reward_tT))
    return jnp.swapaxes(qdiff_tT, 0, 1)             # (J, T)


def hierarchical_model(cue, choice, rt, reward, min_rt, *, pooled=False):
    """cue/choice/rt/reward: (J, T) arrays; min_rt: (J,)."""
    J = cue.shape[0]

    # Group-level means and spreads (weakly informative, on unconstrained scales).
    mu_log_a = numpyro.sample("mu_log_a", dist.Normal(jnp.log(1.2), 0.5))
    mu_w = numpyro.sample("mu_w", dist.Normal(0.0, 1.0))        # logit of w
    mu_beta = numpyro.sample("mu_beta", dist.Normal(1.0, 1.0))
    mu_logit_alpha = numpyro.sample("mu_logit_alpha", dist.Normal(0.0, 1.0))
    mu_wq = numpyro.sample("mu_wq", dist.Normal(0.0, 1.0))
    mu_t0 = numpyro.sample("mu_t0_frac", dist.Normal(0.0, 1.0))  # logit of t0 fraction

    if pooled:
        log_a = mu_log_a; w_raw = mu_w; beta = mu_beta
        logit_alpha = mu_logit_alpha; wq = mu_wq; t0_raw = mu_t0
        a = jnp.exp(log_a) * jnp.ones(J)
        w = jax.nn.sigmoid(w_raw) * jnp.ones(J)
        beta = beta * jnp.ones(J)
        alpha = jax.nn.sigmoid(logit_alpha) * jnp.ones(J)
        wq = wq * jnp.ones(J)
        t0 = jax.nn.sigmoid(t0_raw) * 0.95 * min_rt
    else:
        s_log_a = numpyro.sample("s_log_a", dist.HalfNormal(0.3))
        s_w = numpyro.sample("s_w", dist.HalfNormal(0.5))
        s_beta = numpyro.sample("s_beta", dist.HalfNormal(0.5))
        s_alpha = numpyro.sample("s_alpha", dist.HalfNormal(0.5))
        s_wq = numpyro.sample("s_wq", dist.HalfNormal(0.5))
        s_t0 = numpyro.sample("s_t0", dist.HalfNormal(0.5))
        with numpyro.plate("subj", J):
            log_a = mu_log_a + s_log_a * numpyro.sample("z_log_a", dist.Normal(0, 1))
            w_raw = mu_w + s_w * numpyro.sample("z_w", dist.Normal(0, 1))
            beta = mu_beta + s_beta * numpyro.sample("z_beta", dist.Normal(0, 1))
            logit_alpha = mu_logit_alpha + s_alpha * numpyro.sample("z_alpha", dist.Normal(0, 1))
            wq = mu_wq + s_wq * numpyro.sample("z_wq", dist.Normal(0, 1))
            t0_raw = mu_t0 + s_t0 * numpyro.sample("z_t0", dist.Normal(0, 1))
        a = jnp.exp(log_a)
        w = jax.nn.sigmoid(w_raw)
        alpha = jax.nn.sigmoid(logit_alpha)
        t0 = jax.nn.sigmoid(t0_raw) * 0.95 * min_rt

    numpyro.deterministic("a", a)
    numpyro.deterministic("w", w)
    numpyro.deterministic("beta", beta)
    numpyro.deterministic("alpha", alpha)
    numpyro.deterministic("w_q", wq)
    numpyro.deterministic("t0", t0)

    qdiff = _rw_value_diff(alpha, choice, reward)   # (J, T)
    drift = wq[:, None] * qdiff + beta[:, None] * cue

    logp = wfpt_logpdf(rt, choice, drift, a[:, None], w[:, None], t0[:, None])
    numpyro.factor("obs", jnp.sum(jnp.where(jnp.isfinite(logp), logp, -1e6)))


def pack_data(subjects, sl=None):
    """Stack simulate_subject dicts into (J, T) arrays for the model.

    ``sl`` optionally selects a trial slice (e.g. a train/test split); the
    minimum RT is taken over the selected trials.
    """
    import numpy as np
    sel = (lambda x: x[:, sl]) if sl is not None else (lambda x: x)
    rt = sel(np.stack([s["rt"] for s in subjects]))
    choice = sel(np.stack([s["choice"] for s in subjects]).astype(float))
    reward = sel(np.stack([s["reward"] for s in subjects]).astype(float))
    cue = sel(np.stack([s["cue"] for s in subjects]))
    min_rt = rt.min(axis=1)
    return dict(cue=jnp.asarray(cue), choice=jnp.asarray(choice),
                rt=jnp.asarray(rt), reward=jnp.asarray(reward),
                min_rt=jnp.asarray(min_rt))
