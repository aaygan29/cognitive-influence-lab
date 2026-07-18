"""
simulate.py (v0.3.0) — a mechanistically honest generator of Concealed
Information Test (CIT) data, grounded at the SINGLE-TRIAL level.

This is NOT a "lie generator." It encodes the structure of the *recognition*
detection problem. v0.3.0 rewrites the generative model in response to the
agent-council red team (see critiques.md / revision_log.md). The changes that
matter scientifically:

  C2/C14/C16 (averaged-vs-single-trial): the recognition effect is now defined
    at the SINGLE-TRIAL level and grounded to the ERP CORE P3 single-trial
    Cohen's d (median 0.37). Averaging n_trials probe presentations reduces the
    within-item trial noise as 1/sqrt(n), so the *averaged* effect size EMERGES
    from (single-trial d, n_trials, between-subject gain CV) rather than being
    typed in. `n_trials` is a first-class, sweepable parameter.

  C9 (channel correlation): an item-level orienting fluctuation is shared across
    autonomic channels via per-channel loadings, so channel saliences are
    genuinely correlated (as real physiology is). This is the variance floor
    that averaging cannot remove, and it is what makes fusion non-trivial.

  C4/C5/C11 (countermeasures): countermeasures are injected at the SIGNAL level
    (before within-set z / MAD) as a TAXONOMY with a graded skill parameter:
      - 'probe_suppression' : covertly flattens the PROBE (foils untouched) —
        the dangerous Rosenfeld-style CM the foil-based gate CANNOT see.
      - 'foil_augmentation' : boosts a few foils to mimic the probe — the crude
        CM the gate is designed to catch.
      - 'global_dampening'  : dampens ALL items (dissociation/counting).
    cm_skill in [0,1] scales efficacy (0 = untrained, 1 = expert).

  C12 (habituation): repeated exposure to probe items attenuates the recognition
    gain multiplicatively (exp(-lambda * exposure_count)). Most subjects have
    exposure 0; a configurable fraction are "returning/prepped" with decayed
    probe responses that mimic innocence.

  C13 (non-responders): the per-subject recognition gain is an explicit
    lognormal-like draw floored at 0; the fraction with sub-threshold gain is a
    reported quantity, not a hidden one.

Units: one row = one *item* (probe or foil), already AVERAGED over `n_trials`
presentations. Signal is in single-trial-noise SD units (sigma_trial := 1).
Items group into question sets (one probe + several foils). Decision unit is the
(subject, question_set).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Item-level physiological / behavioral channels. Higher = stronger orienting.
MODALITIES = ["p300", "eda", "pupil", "hr", "resp", "rt"]

# ---------------------------------------------------------------------------
# SINGLE-TRIAL recognition effect size (Cohen's d) per modality — the effect on
# ONE probe presentation, in single-trial-noise SD units.
#
# GROUNDING CHAIN (stated explicitly per C2/C14):
#   * p300: 0.37  — the ERP CORE P3 single-trial d (median across 15 subjects,
#     Pz, 300-600 ms; results/real_erp_stats.json). This is the ONE channel we
#     measured at the single-trial level.
#   * all other channels: derived from Meijer et al. (2014) meta-analytic
#     AVERAGED CIT d* by applying the SAME single-trial:averaged ratio measured
#     for P300, r = 0.37 / 1.89 = 0.196. So d_single[m] = d*_meijer[m] * r.
#     This preserves the meta-analytic ORDERING while grounding the absolute
#     scale to the single channel we have single-trial data for. The transfer
#     assumption (P300's averaging ratio applies to autonomic channels) is a
#     stated assumption, NOT a measurement — flagged in data_provenance.md.
# Meijer averaged d*: p300 1.89, eda 1.55, resp 1.11, hr 0.89; pupil 0.90 and
# rt 0.70 are non-meta-analytic placeholders (also flagged).
_MEIJER_AVG_D = {"p300": 1.89, "eda": 1.55, "pupil": 0.90, "hr": 0.89, "resp": 1.11, "rt": 0.70}
_SINGLE_TRIAL_RATIO = 0.37 / 1.89     # measured P300 single:averaged ratio
D_SINGLE = {m: _MEIJER_AVG_D[m] * _SINGLE_TRIAL_RATIO for m in MODALITIES}
# -> p300 0.370, eda 0.303, resp 0.217, pupil 0.176, hr 0.174, rt 0.137

# Item-level orienting variability (SD) that does NOT average out with more
# trials: each item has its own "how salient did this land" draw, partly shared
# across autonomic channels. This is the detection FLOOR — with infinite trials
# the probe/foil contrast is still limited by this.
#
# CALIBRATION (C2/C14 — internal consistency of the two ERP CORE numbers):
# ITEM_SD is pinned so that averaging n_trials=40 probe presentations reproduces
# the measured ERP CORE *averaged* between-subject P300 effect (d ~ 1.0), GIVEN
# the single-trial d of 0.37 and RECOG_GAIN_CV of 0.97. It is NOT tuned to hit an
# AUC target; it makes the single-trial and averaged groundings mutually
# consistent. Verified numerically in revision_log.md (emergent averaged P300
# d = 0.98 at ITEM_SD = 0.12).
ITEM_SD = 0.12
# Fraction of the item-level orienting variance that is SHARED across autonomic
# channels (drives inter-channel salience correlation; C9). p300 and rt load
# less on the shared autonomic factor.
_SHARED_LOAD = {"p300": 0.35, "eda": 0.75, "pupil": 0.6, "hr": 0.7, "resp": 0.7, "rt": 0.3}

# Between-subject variability of the RECOGNITION gain (C13). Grounded to the
# ERP CORE between-subject CV of the P300 effect (~0.97): some subjects are
# strong responders, a substantial fraction are near-zero non-responders.
RECOG_GAIN_CV = 0.97

# Habituation (C12): recognition gain decays multiplicatively with prior
# exposure to the probe items. exposure_count is drawn per subject.
HABITUATION_LAMBDA = 0.35

# Legacy polygraph channel calibration (set-level scalar). Driven by concealment
# but dominated by trait-anxiety + reactivity confounds so a legacy-ONLY
# detector lands at AUC ~0.70 — the honest independent CQT estimate (Iacono &
# Ben-Shakhar 2019), NOT the ~90% proponent claim.
LEGACY_D = 1.4
LEGACY_NOISE = 1.7

# Countermeasure taxonomy (C4/C5/C11).
CM_TYPES = ["probe_suppression", "foil_augmentation", "global_dampening"]


def simulate_cit(
    n_subjects: int = 400,
    sets_per_subject: int = 2,
    foils_per_set: int = 5,
    n_trials: int = 40,               # probe/foil presentations AVERAGED per item
    prevalence: float = 0.5,
    countermeasure_rate: float = 0.20,
    cm_skill: float = 0.7,            # 0=untrained .. 1=expert (graded efficacy)
    cm_mix: tuple = (0.34, 0.33, 0.33),  # P(probe_suppression, foil_aug, global_damp)
    returning_rate: float = 0.10,     # fraction of subjects with prior probe exposure
    anxiety_sd: float = 1.0,
    reactivity_sd: float = 1.2,
    poor_quality_rate: float = 0.12,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate an item-level CIT dataset (items already averaged over n_trials).

    Columns: subject_id, question_set_id, item_id, item_type ('probe'|'foil'),
    one per MODALITY (averaged item response, single-trial-noise units), quality,
    concealed (set-level truth), used_countermeasure, cm_type, cm_skill_eff,
    exposure_count, n_trials, trait_anxiety, recog_gain_eff.
    """
    rng = np.random.default_rng(seed)
    mod = MODALITIES
    p = len(mod)
    d_single = np.array([D_SINGLE[m] for m in mod])
    shared_load = np.array([_SHARED_LOAD[m] for m in mod])
    # split ITEM_SD into shared vs channel-specific parts (variance decomposition)
    item_shared_sd = ITEM_SD * shared_load
    item_priv_sd = ITEM_SD * np.sqrt(np.clip(1 - shared_load**2, 0, 1))
    trial_sd = 1.0                        # single-trial noise SD (standardized)
    avg_trial_sd = trial_sd / np.sqrt(n_trials)   # after averaging n_trials

    rows = []
    for s in range(n_subjects):
        subj = f"S{s:04d}"
        reactivity = rng.normal(0, reactivity_sd)          # nuisance offset (cancels in-set)
        gain = np.exp(rng.normal(0, 0.35))                 # nuisance scale (cancels)
        anxiety = abs(rng.normal(0, anxiety_sd))           # trait anxiety (confound)
        # per-subject recognition gain, grounded CV, floored at 0 (non-responder)
        recog_gain = max(0.0, rng.normal(1.0, RECOG_GAIN_CV))
        # habituation from prior exposure (most subjects 0)
        exposure = 0
        if rng.random() < returning_rate:
            exposure = int(rng.integers(1, 4))
        recog_gain_eff = recog_gain * np.exp(-HABITUATION_LAMBDA * exposure)
        quality = 1.0
        if rng.random() < poor_quality_rate:
            quality = rng.uniform(0.25, 0.6)
        q_noise = 1.0 + (1.0 - quality) * 2.5              # poor quality => more noise

        for qs in range(sets_per_subject):
            set_id = f"{subj}_Q{qs}"
            informed = rng.random() < prevalence
            cm = informed and (rng.random() < countermeasure_rate)
            cm_type = None
            cm_skill_eff = 0.0
            cm_targets = set()
            if cm:
                cm_type = CM_TYPES[int(rng.choice(len(CM_TYPES), p=list(cm_mix)))]
                # graded skill: draw around cm_skill, clipped to [0,1]
                cm_skill_eff = float(np.clip(rng.normal(cm_skill, 0.15), 0.0, 1.0))
                if cm_type == "foil_augmentation":
                    k = int(rng.integers(2, 4))
                    cm_targets = set(rng.choice(foils_per_set,
                                                size=min(k, foils_per_set),
                                                replace=False).tolist())

            # --- Legacy polygraph channel (set-level scalar) --------------
            legacy_signal = (LEGACY_D * (1.0 if informed else 0.0)
                             + 0.9 * anxiety
                             + 0.5 * reactivity
                             + rng.normal(0, LEGACY_NOISE))
            if cm:
                # physical/mental CMs also perturb the legacy box
                legacy_signal += rng.normal(0, 1.0) - 0.6 * cm_skill_eff * (LEGACY_D if informed else 0)

            n_items = foils_per_set + 1
            foil_idx = 0
            for it in range(n_items):
                is_probe = (it == 0)
                base = reactivity + anxiety * 0.8          # raises ALL items (cancels in-set)
                # item-level orienting: shared autonomic factor + channel-specific
                shared_item = rng.normal(0, 1)
                item_orient = item_shared_sd * shared_item + item_priv_sd * rng.normal(0, 1, p)
                # averaged within-item trial noise (shrinks with n_trials)
                avg_noise = rng.normal(0, avg_trial_sd * q_noise, p)
                vals = base + item_orient + avg_noise
                # recognition signal on the probe (single-trial d, per-subject gain)
                if is_probe and informed:
                    sig = d_single * recog_gain_eff
                    if cm and cm_type == "probe_suppression":
                        sig = sig * (1.0 - cm_skill_eff)   # covertly flatten probe
                    vals = vals + sig
                if not is_probe:
                    if cm and cm_type == "foil_augmentation" and (foil_idx in cm_targets):
                        # boost a foil toward probe magnitude (crude, gate-visible)
                        vals = vals + d_single * recog_gain_eff * cm_skill_eff * rng.uniform(0.8, 1.3)
                    foil_idx += 1
                if cm and cm_type == "global_dampening":
                    vals = vals * (1.0 - 0.6 * cm_skill_eff)  # dampen probe AND foils
                vals = vals * gain
                rows.append({
                    "subject_id": subj,
                    "question_set_id": set_id,
                    "item_id": f"{set_id}_i{it}",
                    "item_type": "probe" if is_probe else "foil",
                    **{m: float(vals[j]) for j, m in enumerate(mod)},
                    "legacy_poly": float(legacy_signal),
                    "quality": float(quality),
                    "concealed": int(informed),
                    "used_countermeasure": int(cm),
                    "cm_type": cm_type if cm else "none",
                    "cm_skill_eff": float(cm_skill_eff),
                    "exposure_count": int(exposure),
                    "n_trials": int(n_trials),
                    "trait_anxiety": float(anxiety),
                    "recog_gain_eff": float(recog_gain_eff),
                })
    return pd.DataFrame(rows)


def nonresponder_fraction(recog_gains: np.ndarray, thresh: float = 0.3) -> float:
    """Fraction of subjects whose recognition gain is below `thresh` (effectively
    non-responders). Reported per C13."""
    return float(np.mean(np.asarray(recog_gains) < thresh))


if __name__ == "__main__":
    df = simulate_cit(seed=0)
    print(df.shape)
    print(df.groupby("item_type")[MODALITIES].mean().round(3))
    print("cm types:", df[df.used_countermeasure == 1]["cm_type"].value_counts().to_dict())
