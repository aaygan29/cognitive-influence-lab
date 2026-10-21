"""
heads.py -- WARDEN's three heads.

H2 (Detector) reuses deceptkit's calibrated, abstaining fusion layer verbatim,
with WARDEN's channel set (ITPC removed, C10) and the caller's prevalence prior
for base-rate-honest PPV (C4: simulator-PPV only). H1 and H3 are new but built on
the same discipline.

H1 (Assay)  -- assumption-relative manipulability SENSITIVITY (not a bound, C2;
               not a trait, C1/C11). Reports a defensive-sufficiency verdict, not
               an exploitability score (C13). Brain-beats-behavior delta with a CI
               that may include zero.
H3 (Floor)  -- adaptive minimax countermeasure ORDERING (C5), coupled to detector
               fragility (C14), conjunction across channels (PRISM). Human-facing
               output is a "simulated countermeasure ordering," not a guarantee (C7).
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

# reuse deceptkit's fusion machinery
sys.path.insert(0, "/Users/aayushgandhi/.claude-science/orgs/98e6e2fb-66d2-4ec7-8443-8874d0a547b7/workspaces/b789047b-247d-46cc-971e-5449732b6a1f")
from deceptkit.fusion import MultivariateFusion, SPRTPolicy
from .simulate_influence import simulate_manipulability
from .assumptions import (Theta, enforce_scope, NEURAL_CHANNELS, BEHAVIORAL_CHANNELS,
                          AUTONOMIC_CHANNELS, ALL_CHANNELS)


# ---------------------------------------------------------------------------
# H2 -- live-engagement detector (deceptkit fusion, WARDEN channels)
# ---------------------------------------------------------------------------
def h2_detect(feats: pd.DataFrame, theta: Theta, n_splits=5, seed=0) -> dict:
    """Subject-grouped CV detection with abstention. Returns decisions + coverage
    + simulator-PPV at the caller's prevalence prior. ABSTAIN never clears (C8)."""
    enforce_scope(theta)                          # refuse per-individual output in adverse contexts
    from sklearn.model_selection import StratifiedGroupKFold
    y = feats["concealed"].to_numpy()
    groups = feats["subject_id"].to_numpy()
    channels = list(theta.channels)
    pol = SPRTPolicy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = []
    for tr, te in sgkf.split(np.zeros((len(feats), 1)), y, groups):
        m = MultivariateFusion(channels=channels, policy=pol)
        m.fit(feats.iloc[tr].reset_index(drop=True))
        parts.append(m.decide(feats.iloc[te].reset_index(drop=True)))
    out = pd.concat(parts, ignore_index=True)

    # coverage is the HEADLINE (C3): fraction of epochs we did NOT abstain on
    coverage = float((out["action"] != "ABSTAIN").mean())
    acted = out[out["action"] != "ABSTAIN"]
    # simulator accuracy among acted epochs (referenced to the simulator's labels, C4)
    if len(acted):
        called_informed = acted["action"] == "INFORMED"
        truth = acted["concealed"] == 1
        tp = int((called_informed & truth).sum()); fp = int((called_informed & ~truth).sum())
        sim_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    else:
        sim_precision = float("nan")

    # base-rate honest simulator-PPV at the caller's prevalence prior (C4)
    ppv = _ppv_at_prevalence(out, theta.prevalence_prior)
    return {"decisions": out, "coverage": coverage,
            "sim_precision_at_sim_prev": sim_precision,
            "simulator_ppv_at_prior": ppv,
            "prevalence_prior": theta.prevalence_prior,
            "note": "field sensitivity/specificity are UNMEASURABLE (C4); "
                    "PPV shown is a simulator PPV only."}


def _ppv_at_prevalence(dec: pd.DataFrame, prevalence: float) -> float:
    """Re-weight the simulator's TPR/FPR to an arbitrary base rate -> simulator-PPV."""
    informed = dec["action"] == "INFORMED"
    pos = dec["concealed"] == 1
    tpr = float((informed & pos).sum() / max((pos).sum(), 1))
    fpr = float((informed & ~pos).sum() / max((~pos).sum(), 1))
    num = tpr * prevalence
    den = tpr * prevalence + fpr * (1 - prevalence)
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# H1 -- manipulability SENSITIVITY assay (assumption-relative)
# ---------------------------------------------------------------------------
def h1_assay(mdf: pd.DataFrame, theta: Theta, n_splits=5, seed=0) -> dict:
    """Estimate expected realized sensitivity S(Theta) and the brain-beats-behavior
    delta with a bootstrap CI that may include zero (C1/C2/C12).

    Returns an assumption-relative SENSITIVITY and a defensive-sufficiency verdict,
    never a per-person exploitability score (C13) and never a stored per-person M (C11).
    """
    enforce_scope(theta)
    beh_cols = ["x_beh1", "x_beh2"]
    neu_cols = ["x_neu1", "x_neu2"]
    y = mdf["realized_shift"].to_numpy()
    groups = mdf["subject_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    def cv_r2(cols):
        preds = np.zeros_like(y)
        for tr, te in gkf.split(mdf, y, groups):
            lr = LinearRegression().fit(mdf.iloc[tr][cols], y[tr])
            preds[te] = lr.predict(mdf.iloc[te][cols])
        ss_res = np.sum((y - preds) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot, preds

    r2_beh, _ = cv_r2(beh_cols)
    r2_full, _ = cv_r2(beh_cols + neu_cols)
    delta = r2_full - r2_beh                       # brain-beats-behavior increment

    # bootstrap CI on the delta (subject-resampled) -- allowed to include zero
    rng = np.random.default_rng(seed)
    subs = np.unique(groups); deltas = []
    for _ in range(300):
        samp = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([np.where(groups == s)[0] for s in samp])
        sub = mdf.iloc[idx].reset_index(drop=True)
        ys = sub["realized_shift"].to_numpy(); gs = sub["subject_id"].to_numpy()
        try:
            gk = GroupKFold(n_splits=min(5, len(np.unique(gs))))
            def _r2(cols):
                pr = np.zeros_like(ys)
                for tr, te in gk.split(sub, ys, gs):
                    lr = LinearRegression().fit(sub.iloc[tr][cols], ys[tr])
                    pr[te] = lr.predict(sub.iloc[te][cols])
                return 1 - np.sum((ys - pr) ** 2) / np.sum((ys - ys.mean()) ** 2)
            deltas.append(_r2(beh_cols + neu_cols) - _r2(beh_cols))
        except Exception:
            continue
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan)

    # expected realized sensitivity S(Theta): mean |predicted shift| per unit influence
    lr_full = LinearRegression().fit(mdf[beh_cols + neu_cols], y)
    S = float(np.mean(np.abs(lr_full.predict(mdf[beh_cols + neu_cols]))) / max(theta.influence_scale, 1e-9))

    return {"S_sensitivity": S, "r2_behavior_only": r2_beh, "r2_full": r2_full,
            "brain_beats_behavior_delta": delta, "delta_ci95": (float(lo), float(hi)),
            "delta_ci_includes_zero": bool(lo <= 0 <= hi) if deltas else None,
            "operator_class": theta.operator_class,
            "note": "S is a SENSITIVITY under Theta, not a bound (C2) and not a trait "
                    "(C1). Low S under benign class is NOT clearance vs hostile class (C12)."}


# ---------------------------------------------------------------------------
# H1b -- bounded drug/arousal STATE sensitivity sweep (corpus Axis B)
# ---------------------------------------------------------------------------
def h1_state_sweep(theta: Theta, states=(-1.0, -0.5, 0.0, 0.5, 1.0),
                   n_subjects=800, neural_increment=0.22, seed=0) -> dict:
    """Report how far the manipulability sensitivity S(Theta) MOVES across a
    plausible drug/arousal state range (corpus Axis B: oxytocin/stress robustly
    moderate trust/conformity/compliance).

    This is the new capability the corpus licenses. It is deliberately a BOUNDED
    SWEEP, not a calibrated 'drugged M': the corpus effect sizes are too
    heterogeneous to point-estimate, so WARDEN emits the RANGE of S over states and
    the state-elasticity (dS per unit state), never a single drugged number. A wide
    range is itself the finding: state is an axis an adversary can move.
    """
    enforce_scope(theta)
    rows = []
    for s in states:
        mdf = simulate_manipulability(n_subjects=n_subjects, neural_increment=neural_increment,
                                      operator_class=theta.operator_class, state=s, seed=seed)
        res = h1_assay(mdf, theta, seed=seed)
        rows.append({"state": s, "S_sensitivity": res["S_sensitivity"],
                     "r2_full": res["r2_full"]})
    tab = pd.DataFrame(rows)
    S_lo, S_hi = float(tab["S_sensitivity"].min()), float(tab["S_sensitivity"].max())
    S_base = float(tab.loc[tab["state"] == 0.0, "S_sensitivity"].iloc[0])
    # state-elasticity: slope of S vs state (per unit s), a bounded moderator magnitude
    elasticity = float(np.polyfit(tab["state"], tab["S_sensitivity"], 1)[0])
    return {"sweep": tab, "S_at_baseline": S_base,
            "S_range_over_state": (S_lo, S_hi),
            "S_relative_swing": (S_hi - S_lo) / max(S_base, 1e-9),
            "state_elasticity": elasticity,
            "note": "BOUNDED sweep over a drug/arousal state axis (corpus Axis B). "
                    "Not a calibrated drug coefficient (effect sizes heterogeneous). "
                    "The RANGE is the finding: state is an adversary-movable axis. A "
                    "covertly induced state is an asymmetric-power case refused under C8."}


# ---------------------------------------------------------------------------
# H3 -- adaptive minimax countermeasure ordering (in-silico)
# ---------------------------------------------------------------------------
def h3_floor(feats_builder, theta: Theta, channels=None, seed=0) -> dict:
    """Compute the minimax saddle d* = argmin_d max_u S(d,u) over which channel to
    harden, with u free to REALLOCATE across open channels after seeing d (C5).

    feats_builder(blocked_channels, attack_channel) -> feats DataFrame lets H3 probe
    the detector's realized manipulability M when a defense blocks some channels and
    an adaptive adversary reallocates influence to attack_channel.

    Couples to detector fragility (C14): if signal is suppressed, d* is forced upward
    or ABSTAINS rather than solving on suppressed values. Conjunction across channels
    (PRISM): the floor must hold across ALL open channels, not on average.
    """
    channels = channels or list(theta.channels)

    def M(blocked, attack):
        """Realized manipulability: detector FAILURE rate (miss) when defense blocks
        `blocked` and adversary attacks `attack`. Higher M = worse for defender."""
        feats = feats_builder(blocked, attack)
        open_ch = [c for c in channels if c not in blocked]
        if not open_ch:
            return 0.0                             # everything blocked -> no attack surface
        pol = SPRTPolicy()
        m = MultivariateFusion(channels=open_ch, policy=pol).fit(feats)
        dec = m.decide(feats)
        pos = feats["concealed"] == 1
        # miss = attempt active but detector did not call INFORMED
        miss = ((dec["action"] != "INFORMED") & pos).mean()
        return float(miss)

    # candidate single-channel and pairwise defenses d
    candidate_defenses = [frozenset()] + [frozenset([c]) for c in channels] \
        + [frozenset([a, b]) for i, a in enumerate(channels) for b in channels[i+1:]]

    rows = []
    for d in candidate_defenses:
        open_ch = [c for c in channels if c not in d]
        # adaptive adversary: reallocate to the WORST open channel (max over u)
        worst = max((M(d, a) for a in open_ch), default=0.0)
        rows.append({"defense": tuple(sorted(d)), "n_blocked": len(d),
                     "max_manipulability_adaptive": worst})
    tab = pd.DataFrame(rows).sort_values("max_manipulability_adaptive").reset_index(drop=True)

    # minimax saddle: the defense minimizing worst-case manipulability
    d_star = tab.iloc[0]
    # non-adaptive comparison: single best channel evaluated WITHOUT reallocation
    single = tab[tab["n_blocked"] == 1]
    return {"ordering": tab, "d_star": d_star.to_dict(),
            "note": "d* is a SIMULATED countermeasure ordering, not a floor guarantee "
                    "(C7). Adaptive: adversary reallocates after seeing d (C5). "
                    "Conjunction: floor must hold across all open channels (PRISM)."}
