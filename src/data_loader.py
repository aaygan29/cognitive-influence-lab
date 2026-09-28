"""
Algonauts 2023 data loader.
Handles per-subject ROI fMRI betas and stimulus metadata.
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATA_ROOT = Path(__file__).parent.parent / "data"
FMRI_ROOT = DATA_ROOT / "fmri"
STIMULI_ROOT = DATA_ROOT / "stimuli"
FEATURES_ROOT = DATA_ROOT / "features"

ROIS = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",
    "EBA", "FBA-1", "FBA-2", "mTL-bodies",
    "OFA", "FFA-1", "FFA-2", "mTL-faces",
    "OPA", "PPA", "RSC",
    "OWFA", "VWFA-1", "VWFA-2", "mTL-words",
    "early", "midventral", "midlateral", "midparietal",
    "ventral", "lateral", "parietal",
]

# High-level ROI groupings for functional analysis
FUNCTIONAL_GROUPS = {
    "early_visual": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "body": ["EBA", "FBA-1", "FBA-2"],
    "face": ["OFA", "FFA-1", "FFA-2"],
    "place": ["OPA", "PPA", "RSC"],
    "word": ["OWFA", "VWFA-1", "VWFA-2"],
    "broad": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}


def subject_dir(subj_id: int) -> Path:
    return FMRI_ROOT / f"subj{subj_id:02d}"


def load_fmri(subj_id: int, hemisphere: str = "lh", split: str = "training") -> np.ndarray:
    """Load whole-brain fMRI betas for a subject.
    Returns array of shape (n_images, n_voxels).
    """
    path = subject_dir(subj_id) / f"{split}_split" / "training_fmri" / f"{hemisphere}_training_fmri.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing fMRI data: {path}\nRun scripts/download_data.py first.")
    return np.load(path)


def load_roi_mask(subj_id: int, roi: str, hemisphere: str = "lh") -> np.ndarray:
    """Load boolean mask for a named ROI."""
    mask_dir = subject_dir(subj_id) / "roi_masks"
    path = mask_dir / f"mapping_{roi}.npy"
    if not path.exists():
        path = mask_dir / f"{hemisphere}_{roi}.npy"
    if not path.exists():
        raise FileNotFoundError(f"ROI mask not found for {roi}: {mask_dir}")
    return np.load(path).astype(bool)


def load_roi_fmri(subj_id: int, roi: str, hemisphere: str = "lh") -> np.ndarray:
    """Load fMRI betas masked to a specific ROI. Returns (n_images, n_roi_voxels)."""
    betas = load_fmri(subj_id, hemisphere=hemisphere, split="training")
    mask = load_roi_mask(subj_id, roi, hemisphere)
    return betas[:, mask]


def load_all_rois(subj_id: int, rois: List[str], hemisphere: str = "lh") -> Dict[str, np.ndarray]:
    """Load multiple ROIs at once. Returns {roi_name: (n_images, n_voxels)}."""
    betas = load_fmri(subj_id, hemisphere=hemisphere)
    result = {}
    for roi in rois:
        try:
            mask = load_roi_mask(subj_id, roi, hemisphere)
            result[roi] = betas[:, mask]
        except FileNotFoundError:
            pass
    return result


def load_features(feature_type: str = "clip") -> np.ndarray:
    """Load pre-extracted DNN features. Returns (n_images, feature_dim)."""
    path = FEATURES_ROOT / f"{feature_type}_features.npy"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}\nRun scripts/extract_features.py first.")
    return np.load(path)


def load_image_ids() -> np.ndarray:
    """Load COCO image IDs for the training set (same order as fMRI)."""
    path = DATA_ROOT / "stimuli" / "image_ids.npy"
    if path.exists():
        return np.load(path)
    return None


def get_available_subjects() -> List[int]:
    """Return list of subjects that have fMRI data downloaded."""
    subjects = []
    for i in range(1, 9):
        d = subject_dir(i)
        if d.exists() and any(d.rglob("*.npy")):
            subjects.append(i)
    return subjects


def get_shared_images(subj_ids: List[int], n: Optional[int] = None) -> np.ndarray:
    """Return indices of images shared across all specified subjects (all training images are shared in Algonauts)."""
    available = get_available_subjects()
    shared = [s for s in subj_ids if s in available]
    if not shared:
        raise ValueError("No subjects available. Run download first.")
    # In Algonauts 2023, all subjects saw the same training images
    betas = load_fmri(shared[0], hemisphere="lh")
    n_total = betas.shape[0]
    indices = np.arange(n_total)
    if n is not None:
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=min(n, n_total), replace=False)
        indices = np.sort(indices)
    return indices
