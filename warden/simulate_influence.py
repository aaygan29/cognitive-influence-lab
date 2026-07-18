"""
simulate_influence.py -- a mechanistically honest generator of influence-exposure
epochs, grounded in real effect sizes.

This is NOT an "influence generator." It encodes the structure of the
influence-LANDING detection problem and the manipulability-SENSITIVITY estimation
problem, at parameters fixed to the honest literature:

  * Neural-beats-behavior increment is MODEST. Falk et al. 2010 (MPFC r=0.49)
    report neural adds ~23% of variance beyond self-report. We set the neural
    channels to carry a modest, tunable increment over the behavioral floor, with
    the increment allowed to be near zero for some channels (validity_map H1.2).

  * Live influence-landing detection is WEAK. Unlike an averaged P300 CIT
    (d*~1.59), an influence attempt in a self-paced naturalistic stream has no
    probe onset and no clean baseline, so the single-epoch effect is small. We
    ground the per-epoch neural effect near the deceptkit single-trial scale
    (d~0.37) and dilute it further for the naturalistic setting.

  * Countermeasures collapse detection. A covert suppression (the Rosenfeld-style
    channel-flattening the foil gate cannot see) drives H2 toward chance
    (deceptkit 96% miss under covert CM).

  * Susceptibility has a CLINICAL TAIL, not a Gaussian bulk. Interrogative
    suggestibility is trait-stable and clinically patterned (corpus Axis E:
    forensic/ADHD/high-anxiety populations score higher; 10.1016/j.ijlp.2013.11.014,
    10.1017/s0033291708002882), and health-misinformation susceptibility is
    cross-situationally consistent within a person (corpus Axis A:
    10.1037/hea0000978). The between-subject reactivity distribution is therefore
    right-skewed with a heavy tail of genuinely-high-susceptibility subjects, so
    H3 worst-case-state reasoning sees the real vulnerable subpopulation.

  * Susceptibility is STATE-DEPENDENT (corpus Axis B, the new capability). A drug/
    arousal state robustly moderates trust/conformity/compliance: oxytocin raises
    conformity to competent partners and compliance with agents (10.1371/journal.
    pone.0153352, 10.1177/0018720816687205; review 10.1016/bs.pbr.2025.02.008),
    with documented boundary conditions (10.1037/pag0000545, 10.1016/j.yhbeh.2018.
    02.003). The effect sizes are heterogeneous, so state enters as a BOUNDED
    scalar s in [-1, +1] that multiplicatively scales the influence effect, never
    a calibrated point coefficient: s>0 = susceptibility-raising state (oxytocin/
    acute stress), s<0 = susceptibility-lowering. WARDEN reports how far the
    estimate MOVES across s, it does not emit a single "drugged M".

The generator emits a column contract compatible with deceptkit's fusion layer:
z_<channel>, quality, max_foil_z, foil_disp, concealed (= attempt_active),
used_countermeasure, subject_id, question_set_id (= epoch_id).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .assumptions import (NEURAL_CHANNELS, BEHAVIORAL_CHANNELS, AUTONOMIC_CHANNELS,
                          ALL_CHANNELS)

# Per-epoch (single-exposure) effect sizes, naturalistic influence-landing.
# Deliberately small: a naturalistic stream has no probe onset. Neural belief-
# processing markers carry a modest signal; behavior carries the honest floor.
# MEASURED groundings (real single-trial data, not literature guesses):
#   z_n400  = ERP CORE N400, CPz 300-500ms, 11 subj x 120 trials, congruent vs
#             incongruent: single-trial median d = 0.317 (averaged-ERP paired
#             d = 0.94, p = 0.011). DOI 10.1016/j.neuroimage.2020.117465.
#   z_choice controlled-regime anchor = Costello 2024 Study 2, single AI message
#             vs control d = 0.573 (full dialogue d = 0.659, ATE +12.5 belief
#             pts, p = 3e-19, n = 990). DOI 10.1126/science.adq1814. The 0.40
#             per-epoch value stays BELOW that measured causal effect because a
#             naturalistic stream lacks the controlled targeted rebuttal.
D_EPOCH = {
    "z_n400":   0.317,  # MEASURED ERP CORE N400 single-trial median d (was 0.34 est)
    "z_latepos":0.30,   # late positivity to incongruent claims
    "z_prederr":0.26,   # prediction-error signal
    "z_choice": 0.40,   # behavioral floor; controlled-regime anchor d=0.57 (Costello)
    "z_rt":     0.22,   # response-time / micro-hesitation drift
    "z_eda":    0.30,   # autonomic bridge (deceptkit-grounded)
    "z_pupil":  0.18,
}

# Shared orienting factor loadings (channels are genuinely correlated, as real
# physiology is -- this is the variance floor fusion must respect, PRISM/C9).
ORIENT_LOADING = {
    "z_n400": 0.25, "z_latepos": 0.30, "z_prederr": 0.20,
    "z_choice": 0.15, "z_rt": 0.35, "z_eda": 0.55, "z_pupil": 0.50,
}

# Bounded drug/arousal state sensitivity (corpus Axis B). At the extreme admissible
# state (|s| = 1) the influence effect is scaled by (1 + STATE_SENSITIVITY * s).
# 0.35 keeps the swing modest and bounded: a strong susceptibility-raising state
# lifts the per-epoch effect by ~35%, a lowering state cuts it by ~35%. This is a
# SENSITIVITY-SWEEP magnitude, not a calibrated oxytocin coefficient -- the corpus
# effect sizes are too heterogeneous to point-estimate (logged as a real limit).
STATE_SENSITIVITY = 0.35

# Right-skew of the between-subject reactivity tail (corpus Axis A + E). The
# lognormal sigma governs how heavy the high-susceptibility tail is; a small
# fraction of subjects (the clinical/forensic tail) react far above the median.
REACTIVITY_SIGMA = 0.55          # was 0.35 (Gaussian-ish); raised for the clinical tail
CLINICAL_TAIL_FRAC = 0.08        # ~8% draw from a shifted-up tail component
CLINICAL_TAIL_SHIFT = 0.9        # extra log-gain for the tail component


def simulate_influence(n_subjects=800, epochs_per_subject=6, prevalence=0.30,
                       countermeasure_rate=0.0, cm_skill=0.8, cm_type="probe_suppression",
                       naturalistic_dilution=0.65, poor_quality_rate=0.12,
                       state=0.0, seed=0) -> pd.DataFrame:
    """Generate influence-exposure epochs.

    prevalence          : fraction of epochs where an influence attempt is active
                          (H2 uses the caller's prevalence_prior separately for PPV).
    naturalistic_dilution: multiplies per-epoch d to model the missing probe onset
                          (1.0 = clean oddball paradigm; <1 = self-paced stream).
    countermeasure_rate  : fraction of attempt-active epochs with a covert CM.
    cm_type              : 'probe_suppression' (covert, gate-invisible),
                           'foil_augmentation' (crude, gate-visible),
                           'global_dampening'.
    state                : bounded drug/arousal state s in [-1, +1] (corpus Axis B).
                           Scales every per-epoch effect by (1 + STATE_SENSITIVITY*s).
                           s>0 raises susceptibility (oxytocin/acute stress), s<0
                           lowers it. Report a SWEEP over s, not a single value.
    """
    s_state = float(np.clip(state, -1.0, 1.0))
    state_gain = 1.0 + STATE_SENSITIVITY * s_state    # bounded multiplicative moderator
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        # per-subject reactivity: right-skewed lognormal with a clinical tail
        # (corpus Axis A + E). Most subjects near the median; a small forensic/
        # clinical fraction draws from a shifted-up tail component.
        tail = rng.random() < CLINICAL_TAIL_FRAC
        mu_log = CLINICAL_TAIL_SHIFT if tail else 0.0
        gain = np.exp(rng.normal(mu_log, REACTIVITY_SIGMA))   # lognormal, heavy right tail
        for e in range(epochs_per_subject):
            attempt = int(rng.random() < prevalence)
            used_cm = int(attempt and (rng.random() < countermeasure_rate))
            orient = rng.normal(0.0, 1.0)             # shared orienting fluctuation
            quality = 1.0 if rng.random() > poor_quality_rate else rng.uniform(0.2, 0.5)

            row = {"subject_id": s, "question_set_id": s * epochs_per_subject + e,
                   "concealed": attempt, "used_countermeasure": used_cm,
                   "cm_type": cm_type if used_cm else "none", "quality": quality}

            foil_zs = []
            for c in ALL_CHANNELS:
                d = D_EPOCH[c] * naturalistic_dilution * gain * state_gain
                mu = d * attempt
                # covert probe_suppression flattens the attempt signal (gate-blind)
                if used_cm and cm_type == "probe_suppression":
                    mu *= (1.0 - cm_skill)
                if used_cm and cm_type == "global_dampening":
                    mu *= (1.0 - 0.7 * cm_skill)
                z = mu + ORIENT_LOADING[c] * orient + rng.normal(0.0, 1.0)
                z *= quality                          # low quality shrinks signal
                row[c] = z
                # foil_augmentation boosts a comparison ("foil") channel -> gate sees it
                if used_cm and cm_type == "foil_augmentation":
                    foil_zs.append(abs(z) + rng.uniform(1.5, 3.0) * cm_skill)
                else:
                    foil_zs.append(abs(rng.normal(0.0, 1.0)))

            # countermeasure/quality diagnostics (drive the abstention gate, not evidence)
            row["max_foil_z"] = float(np.max(foil_zs))
            row["foil_disp"] = float(np.std(foil_zs))
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_manipulability(n_subjects=800, influence_scale=1.0,
                            neural_increment=0.22, operator_class="benign_persuasion",
                            state=0.0, seed=0) -> pd.DataFrame:
    """Generate per-subject data for the H1 sensitivity assay.

    Each subject has a latent susceptibility. The REALIZED behavioral shift under a
    modeled influence input u is the target. Behavioral baseline features and neural
    belief-processing features both carry susceptibility signal; the neural channels
    add a MODEST increment (neural_increment ~ Falk's +23% variance) over behavior.

    operator_class: 'benign_persuasion' is the class M is calibrated to (C12). A
    'hostile_tradecraft' operator is out-of-distribution -- its realized shift is
    generated from a DIFFERENT susceptibility mapping, so a model trained on benign
    fails to predict it (the P1.3 falsification).

    state: bounded drug/arousal state s in [-1, +1] (corpus Axis B). Scales the
    realized shift by (1 + STATE_SENSITIVITY*s). The H1 assay sweeps s to report
    how far S(Theta) moves across a plausible state range, never a single value.

    The latent susceptibility is right-skewed with a clinical tail (corpus Axis A+E):
    a small forensic/clinical fraction is far more susceptible than the Gaussian bulk.
    """
    s_state = float(np.clip(state, -1.0, 1.0))
    state_gain = 1.0 + STATE_SENSITIVITY * s_state
    rng = np.random.default_rng(seed)
    # right-skewed susceptibility: standard-normal bulk + a shifted clinical tail
    suscept = rng.normal(0.0, 1.0, n_subjects)
    tail_mask = rng.random(n_subjects) < CLINICAL_TAIL_FRAC
    suscept[tail_mask] += CLINICAL_TAIL_SHIFT + rng.normal(0, 0.4, tail_mask.sum())

    # behavioral baseline features (the honest floor): carry susceptibility + noise
    beh_slope = 0.55
    x_beh1 = beh_slope * suscept + rng.normal(0, 1.0, n_subjects)
    x_beh2 = 0.35 * suscept + rng.normal(0, 1.0, n_subjects)

    # neural belief-processing features: carry an INDEPENDENT slice of susceptibility
    # (the part behavior misses) -> this is the brain-beats-behavior increment.
    # u_extra is the outcome-relevant susceptibility dimension that behavior does
    # not see; the neural features load on it, so they add unique predictive value.
    neural_unique = np.sqrt(max(neural_increment, 0.0))
    u_extra = rng.normal(0, 1.0, n_subjects)                # extra susceptibility dim
    x_neu1 = 0.40 * suscept + 0.55 * u_extra + rng.normal(0, 0.8, n_subjects)
    x_neu2 = 0.30 * suscept + 0.40 * u_extra + rng.normal(0, 0.9, n_subjects)

    # realized shift under benign operator (state moderator scales the influence term)
    if operator_class == "benign_persuasion":
        shift = influence_scale * state_gain * (0.7 * suscept + neural_unique * u_extra) \
                + rng.normal(0, 0.8, n_subjects)
    else:
        # hostile tradecraft: exploits a DIFFERENT axis -> benign-trained model misses it
        hostile_axis = rng.normal(0, 1.0, n_subjects)
        shift = influence_scale * state_gain * (0.7 * hostile_axis + 0.3 * u_extra) \
                + rng.normal(0, 0.8, n_subjects)

    return pd.DataFrame({
        "subject_id": np.arange(n_subjects),
        "x_beh1": x_beh1, "x_beh2": x_beh2,
        "x_neu1": x_neu1, "x_neu2": x_neu2,
        "realized_shift": shift,
        "operator_class": operator_class,
        "state": s_state,
    })
