"""
Train subject-specific digital brains (encoding models).
One ridge regression model per ROI per subject.
"""
import sys
import numpy as np
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import (
    get_available_subjects, load_fmri, load_roi_mask, load_features,
    FMRI_ROOT, DATA_ROOT,
)
from src.encoding_model import DigitalBrain, train_all_digital_brains


def load_roi_betas_for_subject(subj_id: int, rois: list) -> dict:
    """Load all ROI betas for a subject. Returns {roi: (n_images, n_voxels)}."""
    betas_lh = load_fmri(subj_id, hemisphere="lh")
    roi_betas = {}
    roi_dir = FMRI_ROOT / f"subj{subj_id:02d}" / "roi_masks"

    for roi in rois:
        mask_path = roi_dir / f"mapping_{roi}.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(bool)
        if mask.sum() == 0:
            continue
        roi_betas[roi] = betas_lh[:, mask]

    return roi_betas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--rois", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="Retrain even if cached")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    # Determine subjects
    available = get_available_subjects()
    if not available:
        print("No subject data found. Run download_data.py first.")
        sys.exit(1)
    subjects = args.subjects or available
    subjects = [s for s in subjects if s in available]
    print(f"Training digital brains for subjects: {subjects}")

    # Load features
    features = load_features(feature_type="clip")
    print(f"Features loaded: {features.shape}")
    n_total = features.shape[0]

    # Train/test split (consistent across subjects — same stimulus indices)
    rng = np.random.default_rng(42)
    n_test = int(n_total * args.test_fraction)
    test_idx = rng.choice(n_total, size=n_test, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), test_idx)

    # Save split indices for evaluation
    splits_dir = DATA_ROOT / "splits"
    splits_dir.mkdir(exist_ok=True)
    np.save(splits_dir / "train_idx.npy", train_idx)
    np.save(splits_dir / "test_idx.npy", test_idx)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    # Load all ROI betas
    all_roi_betas = {}
    for subj_id in subjects:
        print(f"\nLoading fMRI data for subject {subj_id:02d}...")
        # Determine available ROIs
        roi_dir = FMRI_ROOT / f"subj{subj_id:02d}" / "roi_masks"
        available_rois = [p.stem.replace("mapping_", "") for p in roi_dir.glob("mapping_*.npy")]
        rois = args.rois or available_rois
        roi_betas = load_roi_betas_for_subject(subj_id, rois)

        # Subset to training images
        all_roi_betas[subj_id] = {roi: betas[train_idx] for roi, betas in roi_betas.items()}
        print(f"  Loaded {len(roi_betas)} ROIs for subject {subj_id:02d}")

    features_train = features[train_idx]

    # Train
    digital_brains = train_all_digital_brains(
        subject_ids=subjects,
        features_train=features_train,
        all_roi_betas=all_roi_betas,
        force_retrain=args.force,
    )

    print(f"\nTrained {len(digital_brains)} digital brains:")
    for sid, brain in digital_brains.items():
        print(f"  Subject {sid:02d}: {brain.fitted_rois()}")

    # Save training summary
    summary = {
        "subjects": subjects,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "rois_per_subject": {str(sid): brain.fitted_rois() for sid, brain in digital_brains.items()},
    }
    with open(DATA_ROOT / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {DATA_ROOT / 'training_summary.json'}")


if __name__ == "__main__":
    main()
