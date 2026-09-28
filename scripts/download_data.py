"""
Download Algonauts 2023 data (NSD subset) from OSF.

Downloads:
  - Per-subject fMRI beta weights (ROI-masked, training split)
  - Corresponding stimulus images
  - ROI mask arrays

OSF project: https://osf.io/2rvsj/
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
FMRI_ROOT = DATA_ROOT / "fmri"
STIMULI_ROOT = DATA_ROOT / "stimuli"

# OSF file structure for Algonauts 2023
OSF_PROJECT = "2rvsj"
OSF_BASE = f"https://files.osf.io/v1/resources/{OSF_PROJECT}/providers/osfstorage"

# Direct download links for subject-level zip files
# These are the per-subject training data packages
SUBJECT_URLS = {
    1: "https://osf.io/2rvsj/download?subj=subj01",
    2: "https://osf.io/2rvsj/download?subj=subj02",
    3: "https://osf.io/2rvsj/download?subj=subj03",
    4: "https://osf.io/2rvsj/download?subj=subj04",
}


def download_file(url: str, dest: Path, desc: str = None):
    """Download with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return

    desc = desc or dest.name
    print(f"  Downloading {desc}...")

    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                block_size = 8192
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  URL: {url}")
        if dest.exists():
            dest.unlink()
        return False
    return True


def download_via_osf_cli(subj_ids: list):
    """Use osfclient if available for more reliable downloads."""
    try:
        import osfclient  # noqa
        has_osf = True
    except ImportError:
        has_osf = False

    if not has_osf:
        print("osfclient not available, using direct download")
        return False

    for subj_id in subj_ids:
        subj_str = f"subj{subj_id:02d}"
        dest_dir = FMRI_ROOT / subj_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "osfclient", "clone",
            "-p", OSF_PROJECT,
            str(dest_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  OSF clone failed: {result.stderr[:200]}")
            return False
    return True


def download_algonauts_manual(subj_ids: list):
    """
    Manual download instructions and fallback downloader.
    The Algonauts 2023 data requires visiting the challenge page.
    """
    print("\n" + "="*60)
    print("ALGONAUTS 2023 DATA DOWNLOAD")
    print("="*60)
    print("\nThe Algonauts 2023 data is hosted at:")
    print("  https://codalab.lisn.upsaclay.fr/competitions/9304")
    print("  Mirror: http://algonauts.csail.mit.edu/2023/challenge.html")
    print("\nDirect OSF data: https://osf.io/2rvsj/")
    print("\nManual download steps:")
    print("  1. Visit https://osf.io/2rvsj/")
    print("  2. Download per-subject zip files to:")
    for subj_id in subj_ids:
        print(f"     {FMRI_ROOT}/subj{subj_id:02d}/")
    print("\nAlternatively, use the Algonauts toolkit:")
    print("  pip install algonauts2023")
    print()

    # Try direct OSF download via osfclient
    print("Attempting automated download via osfclient...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "osfclient", "ls", OSF_PROJECT],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("OSF project accessible. Files available:")
            print(result.stdout[:500])
        else:
            print(f"Could not list OSF files: {result.stderr[:200]}")
    except Exception as e:
        print(f"OSF access failed: {e}")


def create_synthetic_data(subj_ids: list, n_images: int = 500, seed: int = 42):
    """
    Generate synthetic fMRI data for pipeline testing when real data unavailable.

    Uses a known linear model y = X @ W + noise, where W is subject-specific.
    This lets us validate the pipeline with ground-truth answers before running
    on real data.

    Real evaluation should always use actual NSD/Algonauts data.
    """
    rng = np.random.default_rng(seed)
    print(f"\nGenerating synthetic data for {len(subj_ids)} subjects, {n_images} images...")
    print("(For pipeline validation — replace with real Algonauts data for publication)")

    # Shared stimulus feature space (simulate CLIP features)
    feature_dim = 768
    features = rng.standard_normal((n_images, feature_dim)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    # ROI definitions: (name, n_voxels)
    roi_defs = [
        ("V1v", 800), ("V1d", 600), ("V2v", 500), ("V2d", 400),
        ("V3v", 400), ("V3d", 300), ("hV4", 250),
        ("EBA", 300), ("FBA-1", 200), ("FBA-2", 150),
        ("OFA", 200), ("FFA-1", 250), ("FFA-2", 200),
        ("OPA", 250), ("PPA", 350), ("RSC", 300),
        ("OWFA", 150), ("VWFA-1", 200), ("VWFA-2", 180),
        ("early", 1200), ("midventral", 900), ("midlateral", 800),
        ("midparietal", 700), ("ventral", 950), ("lateral", 850), ("parietal", 750),
    ]

    n_total_voxels = sum(n for _, n in roi_defs)

    # Save shared features
    STIMULI_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(STIMULI_ROOT / "synthetic_features.npy", features)
    print(f"  Saved synthetic features: {features.shape}")

    # Save ROI definitions
    roi_names = [name for name, _ in roi_defs]
    roi_sizes = {name: size for name, size in roi_defs}
    np.save(DATA_ROOT / "roi_names.npy", roi_names)

    for subj_id in subj_ids:
        subj_dir = FMRI_ROOT / f"subj{subj_id:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        roi_dir = subj_dir / "roi_masks"
        roi_dir.mkdir(parents=True, exist_ok=True)
        fmri_dir = subj_dir / "training_split" / "training_fmri"
        fmri_dir.mkdir(parents=True, exist_ok=True)

        # Subject-specific weight matrix (captures individual differences)
        # Different subjects have different response profiles
        noise_level = 0.3
        subject_bias = rng.standard_normal((feature_dim,)) * 0.1  # idiosyncratic tuning

        voxel_offset = 0
        lh_betas = np.zeros((n_images, n_total_voxels), dtype=np.float32)

        for roi_name, n_voxels in roi_defs:
            # Subject-specific mapping: features → voxels
            # Each ROI has a different feature-to-voxel mapping
            W = rng.standard_normal((feature_dim, n_voxels)).astype(np.float32)
            W /= np.linalg.norm(W, axis=0, keepdims=True)

            # Subject-specific scaling (individual differences in ROI magnitude)
            subj_scale = rng.uniform(0.7, 1.3)

            # Compute betas: signal + noise
            signal = (features + subject_bias) @ W * subj_scale
            noise = rng.standard_normal((n_images, n_voxels)).astype(np.float32) * noise_level
            betas = signal + noise

            lh_betas[:, voxel_offset:voxel_offset + n_voxels] = betas

            # Save ROI mask (which voxels belong to this ROI)
            mask = np.zeros(n_total_voxels, dtype=bool)
            mask[voxel_offset:voxel_offset + n_voxels] = True
            np.save(roi_dir / f"mapping_{roi_name}.npy", mask)

            voxel_offset += n_voxels

        np.save(fmri_dir / "lh_training_fmri.npy", lh_betas)
        print(f"  Subject {subj_id:02d}: betas {lh_betas.shape}, {len(roi_defs)} ROIs saved")

    print(f"\nSynthetic data saved to {FMRI_ROOT}")
    print("Run pipeline with --synthetic flag")
    return features


def main():
    parser = argparse.ArgumentParser(description="Download or generate digital brain experiment data")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for pipeline testing (no download needed)")
    parser.add_argument("--n-images", type=int, default=500,
                        help="Number of images for synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_data(args.subjects, n_images=args.n_images)
    else:
        download_algonauts_manual(args.subjects)
        print("\nIf download fails, run with --synthetic to test the pipeline:")
        print(f"  python scripts/download_data.py --synthetic --subjects {' '.join(map(str, args.subjects))}")


if __name__ == "__main__":
    main()
