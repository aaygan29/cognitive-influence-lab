"""CSI = vmPFC − dlPFC from preprocessed task fMRI (nilearn).

CSI is a *neural* contrast (see PREREGISTRATION.md §1). This module computes it from
**fMRIPrep-style, template-space (MNI)** inputs. NARPS on OpenNeuro ships only **raw native-space
BOLD** (~688 MB/run, no derivatives), so the per-subject GLM must be run where preprocessing
exists (e.g. fMRIPrep on the user's compute) — this module does **not** preprocess or fabricate
data, and does not assume which contrast defines "activation" (a preregistered choice).

ROI defaults (Harvard-Oxford cortical atlas) — confirm before use:
  vmPFC ≈ {Frontal Medial Cortex, Frontal Orbital Cortex};  dlPFC ≈ {Middle Frontal Gyrus}.
"""
from __future__ import annotations

import numpy as np

VMPFC_LABELS = ("Frontal Medial Cortex", "Frontal Orbital Cortex")
DLPFC_LABELS = ("Middle Frontal Gyrus",)


def roi_mean(stat_img, mask_img) -> float:
    """Mean of ``stat_img`` over the nonzero voxels of ``mask_img`` (same space required)."""
    from nilearn.masking import apply_mask

    return float(np.nanmean(apply_mask(stat_img, mask_img)))


def compute_csi(vmpfc_activation: float, dlpfc_activation: float) -> float:
    """CSI = vmPFC − dlPFC activation."""
    return float(vmpfc_activation - dlpfc_activation)


def subject_csi(effect_map, vmpfc_mask, dlpfc_mask) -> float:
    """CSI for one subject's effect map: mean(vmPFC) − mean(dlPFC)."""
    return compute_csi(roi_mean(effect_map, vmpfc_mask), roi_mean(effect_map, dlpfc_mask))


def harvard_oxford_masks(threshold: int = 25):
    """Binary (vmPFC, dlPFC) masks from the Harvard-Oxford cortical atlas (nilearn fetch).

    Network/data-dependent; returns two Nifti images. ROI label sets are the module defaults.
    """
    from nilearn import datasets, image

    atlas = datasets.fetch_atlas_harvard_oxford(f"cort-maxprob-thr{threshold}-2mm")
    labels = list(atlas.labels)
    data = image.get_data(atlas.maps)

    def mask_for(names):
        idx = [labels.index(n) for n in names if n in labels]
        return image.new_img_like(atlas.maps, np.isin(data, idx).astype("int8"))

    return mask_for(VMPFC_LABELS), mask_for(DLPFC_LABELS)


def first_level_effect_map(bold_img, events, confounds, t_r: float, contrast: str):
    """nilearn FirstLevelModel → effect-size map for ``contrast``.

    ``contrast`` MUST be supplied (e.g. "gain" for the value-parametric effect). This module does
    not assume which contrast defines CSI — that is a preregistered choice. Requires
    fMRIPrep-preprocessed (MNI-space) BOLD + confounds.
    """
    if not contrast:
        raise ValueError("contrast must be specified (preregistered); none assumed")
    from nilearn.glm.first_level import FirstLevelModel

    flm = FirstLevelModel(t_r=t_r, standardize=False, minimize_memory=True)
    flm = flm.fit(bold_img, events=events, confounds=confounds)
    return flm.compute_contrast(contrast, output_type="effect_size")
