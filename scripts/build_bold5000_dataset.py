"""
Build the BOLD5000 dataset into our experiment format.
Loads ROI betas for all 4 subjects on shared COCO images,
aligns with CLIP features, saves as numpy arrays ready for
encoding model training.

BOLD5000 ROIs available:
  EarlyVis (V1-V3), LOC, OPA, PPA, RSC — both hemispheres
"""
import sys, scipy.io, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"
BOLD_ROOT = DATA_ROOT / "bold5000"
FMRI_ROOT = BOLD_ROOT / "fmri" / "BOLD5000_GLMsingle_ROI_betas" / "mat"
STIMULI_ROOT = BOLD_ROOT / "stimuli"
FEATURES_ROOT = DATA_ROOT / "features"
OUT_ROOT = DATA_ROOT / "processed"
OUT_ROOT.mkdir(exist_ok=True)

SUBJECTS = ["CSI1", "CSI2", "CSI3", "CSI4"]

# ROI name mappings (some subjects use slightly different names)
ROI_VARIANTS = {
    "EarlyVis": ["EarlyVis"],
    "OPA":      ["OPA"],
    "PPA":      ["PPA"],
    "RSC":      ["RSC", "RRSC"],
    "LOC":      ["LOC", "LO"],   # CSI1/CSI4=LOC, CSI2/CSI3=LO
}

HEMISPHERES = ["LH", "RH"]


def find_mat_file(subj: str, roi: str, hemi: str):
    """Find the mat file for a given subject/ROI/hemisphere."""
    variants = ROI_VARIANTS.get(roi, [roi])
    for variant in variants:
        pattern = f"{subj}_GLMbetas-*_{hemi}{variant}.mat"
        matches = list(FMRI_ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_roi_betas(subj: str, roi: str, hemi: str):
    """Load (n_trials, n_voxels) beta array for a subject/ROI/hemi."""
    path = find_mat_file(subj, roi, hemi)
    if path is None:
        return None
    mat = scipy.io.loadmat(str(path))
    # Key is like 'ROIbetas_CSI1_LHEarlyVis'
    data_key = [k for k in mat.keys() if not k.startswith("_")]
    if not data_key:
        return None
    return mat[data_key[0]].astype(np.float32)


def build_dataset():
    """Build aligned (features, betas) arrays for all subjects."""

    # Load shared image list and beta indices
    shared_images = list(np.load(STIMULI_ROOT / "shared_coco_images.npy", allow_pickle=True))
    shared_images = [str(s) for s in shared_images]
    n_images = len(shared_images)
    print(f"Shared COCO images: {n_images}")

    # Load per-subject beta indices
    subject_indices = {}
    for subj in SUBJECTS:
        idx_path = STIMULI_ROOT / f"{subj}_shared_coco_indices.npy"
        if idx_path.exists():
            subject_indices[subj] = np.load(idx_path)
            print(f"  {subj}: {len(subject_indices[subj])} beta indices loaded")

    available_subjects = list(subject_indices.keys())

    # Load CLIP features (should be extracted already)
    features_path = FEATURES_ROOT / "bold5000_clip_features.npy"
    if not features_path.exists():
        print(f"Features not found at {features_path}")
        print("Run: python3 scripts/extract_bold5000_features.py first")
        return None, None

    features = np.load(features_path)
    assert len(features) == n_images, \
        f"Feature count mismatch: {len(features)} features vs {n_images} images"
    print(f"CLIP features: {features.shape}")

    # Build per-subject ROI betas aligned to shared_images order
    all_roi_betas = {}
    for subj in available_subjects:
        beta_idx = subject_indices[subj]
        roi_betas = {}

        for roi in ROI_VARIANTS.keys():
            # Combine LH + RH by concatenation
            hemi_arrays = []
            for hemi in HEMISPHERES:
                betas_full = load_roi_betas(subj, roi, hemi)
                if betas_full is None:
                    continue
                betas_shared = betas_full[beta_idx]  # select shared image trials
                hemi_arrays.append(betas_shared)

            if hemi_arrays:
                combined = np.concatenate(hemi_arrays, axis=1)
                roi_betas[roi] = combined
                print(f"  {subj}/{roi}: {combined.shape}")

        all_roi_betas[subj] = roi_betas

    # Save processed data
    print(f"\nSaving processed data to {OUT_ROOT}/...")
    np.save(OUT_ROOT / "clip_features.npy", features)
    np.save(OUT_ROOT / "image_names.npy", np.array(shared_images))

    for subj, roi_betas in all_roi_betas.items():
        subj_dir = OUT_ROOT / subj
        subj_dir.mkdir(exist_ok=True)
        for roi, betas in roi_betas.items():
            np.save(subj_dir / f"{roi}_betas.npy", betas)

    # Save metadata
    import json
    meta = {
        "n_images": n_images,
        "n_subjects": len(available_subjects),
        "subjects": available_subjects,
        "rois": list(ROI_VARIANTS.keys()),
        "feature_dim": int(features.shape[1]),
        "dataset": "BOLD5000_Release2",
        "stimuli": "COCO_train2014 (shared subset)",
    }
    with open(OUT_ROOT / "metadata.json", "w") as f:
        import json; json.dump(meta, f, indent=2)

    print("\nDataset built:")
    print(f"  Subjects: {available_subjects}")
    print(f"  ROIs: {list(ROI_VARIANTS.keys())}")
    print(f"  Images: {n_images}")
    print(f"  Features: {features.shape}")
    return features, all_roi_betas


if __name__ == "__main__":
    build_dataset()
