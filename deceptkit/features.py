"""
features.py — turn item-level responses into per-(subject, question_set) features.

The decision unit is a question set. For each set we compare the PROBE item
against the distribution of the FOIL items *within that same set and subject*.
This within-subject, within-set contrast is what removes the two nuisances that
sink absolute-physiology methods:

  - between-subject reactivity/gain: cancels because probe and foils share it.
  - trait anxiety: cancels because it raises probe and foils equally.

For each modality we compute a robust standardized "probe salience":
    z_m = (probe_m - median(foils_m)) / (MAD(foils_m) + eps)
i.e. how many robust SDs the probe stands above its own foils. Under the null
(naive subject) E[z_m] ~ 0; for an informed subject E[z_m] > 0.

We also emit countermeasure / quality diagnostics used by the decision layer.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .simulate import MODALITIES

EPS = 1e-6
_MAD_C = 1.4826  # scale factor so MAD estimates SD for a normal


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return _MAD_C * np.median(np.abs(x - med)) + EPS


def build_features(df_items: pd.DataFrame, modalities=MODALITIES) -> pd.DataFrame:
    """Aggregate item-level rows into one row per (subject, question_set).

    Feature columns produced:
      z_<mod>        : robust standardized probe salience per modality
      foil_disp      : mean cross-modal dispersion among foils (countermeasure cue:
                       a clean CIT has quiet foils; a countermeasure user has one or
                       more foils spiking like the probe -> high dispersion)
      max_foil_z     : the largest foil salience relative to the other foils
                       (direct countermeasure cue)
      quality        : session signal quality (passed through)
      n_foils        : number of foils in the set
    Plus carried labels: concealed, used_countermeasure.
    """
    # Subject-level pooled scale (MAD) per modality, from ALL of a subject's foil
    # items across their question sets. A per-set MAD from ~5 foils is a noisy
    # scale estimate; pooling stabilizes it. Centering stays per-set (below), so
    # set-specific offsets are still removed — only the scale is shared.
    subj_scale = {}
    foil_all = df_items[df_items.item_type == "foil"]
    for subj, gs in foil_all.groupby("subject_id", sort=False):
        subj_scale[subj] = {m: _mad(gs[m].to_numpy()) for m in modalities}

    out = []
    grp = df_items.groupby("question_set_id", sort=False)
    for set_id, g in grp:
        probe = g[g.item_type == "probe"]
        foils = g[g.item_type == "foil"]
        if len(probe) != 1 or len(foils) < 2:
            continue
        probe = probe.iloc[0]
        subj = g["subject_id"].iloc[0]
        feat = {"question_set_id": set_id, "subject_id": subj}
        # per-modality robust salience of the probe vs its own foils
        foil_z_matrix = []
        for m in modalities:
            fv = foils[m].to_numpy()
            med = np.median(fv)                       # per-set center
            scale = subj_scale[subj][m]               # subject-pooled scale
            feat[f"z_{m}"] = float((probe[m] - med) / scale)
            foil_z = (fv - med) / scale
            foil_z_matrix.append(foil_z)
        foil_z_matrix = np.vstack(foil_z_matrix)  # (n_mod, n_foils)
        # countermeasure cues: a foil that itself spikes across modalities
        per_foil_strength = foil_z_matrix.mean(axis=0)      # avg over modalities
        feat["max_foil_z"] = float(np.max(per_foil_strength))
        feat["foil_disp"] = float(np.std(per_foil_strength))
        feat["quality"] = float(g["quality"].iloc[0])
        feat["n_foils"] = int(len(foils))
        # legacy polygraph channel (set-level scalar), carried through for the
        # Bayesian fusion layer. Not a within-set contrast — it is the legacy
        # machine's own output for this set.
        if "legacy_poly" in g.columns:
            feat["legacy_poly"] = float(g["legacy_poly"].iloc[0])
        # labels + stratification diagnostics (not seen at inference)
        feat["concealed"] = int(g["concealed"].iloc[0])
        feat["used_countermeasure"] = int(g["used_countermeasure"].iloc[0])
        for extra in ("cm_type", "cm_skill_eff", "exposure_count", "recog_gain_eff", "n_trials"):
            if extra in g.columns:
                v = g[extra].iloc[0]
                feat[extra] = v
        out.append(feat)
    return pd.DataFrame(out)


def feature_columns(modalities=MODALITIES):
    """The columns the model is allowed to see at inference time."""
    return [f"z_{m}" for m in modalities] + ["max_foil_z", "foil_disp", "quality"]


if __name__ == "__main__":
    from .simulate import simulate_cit
    d = simulate_cit(seed=1)
    f = build_features(d)
    print(f.shape)
    print(f.groupby("concealed")[[c for c in f.columns if c.startswith("z_")]].mean().round(3))
