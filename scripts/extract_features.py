"""
Extract CLIP ViT-L/14 features from stimuli.
Saves to data/features/clip_features.npy
"""
import sys
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_extractor import extract_and_save, load_stimuli_array, STIMULI_ROOT, FEATURES_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    # Check if synthetic features already exist
    synthetic_features = STIMULI_ROOT / "synthetic_features.npy"
    if args.synthetic or synthetic_features.exists():
        if synthetic_features.exists():
            features = np.load(synthetic_features)
            FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
            dest = FEATURES_ROOT / "clip_features.npy"
            import shutil
            shutil.copy(synthetic_features, dest)
            print(f"Copied synthetic features {features.shape} → {dest}")
            return features
        else:
            print("Synthetic features not found. Run download_data.py --synthetic first.")
            return

    # Real stimuli
    stimuli = load_stimuli_array()
    if stimuli is None:
        # Try image directory
        img_dir = STIMULI_ROOT / "training_images"
        if img_dir.exists():
            from pathlib import Path
            img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if img_paths:
                print(f"Found {len(img_paths)} image files")
                features = extract_and_save(img_paths, batch_size=args.batch_size, device=args.device)
                return features

        print("No stimuli found. Options:")
        print("  1. Download real data: python scripts/download_data.py")
        print("  2. Use synthetic data: python scripts/download_data.py --synthetic")
        print("     Then: python scripts/extract_features.py --synthetic")
        return

    print(f"Loaded stimuli: {stimuli.shape}")
    features = extract_and_save(stimuli, batch_size=args.batch_size, device=args.device)
    print(f"Features: {features.shape}")
    return features


if __name__ == "__main__":
    main()
