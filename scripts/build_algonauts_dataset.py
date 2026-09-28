"""
Build the Algonauts 2023 dataset into experiment format.

SCALING NOTE (N=4 → N=8):
  This pipeline natively targets all 8 NSD subjects (subj01–subj08). The current
  pilot uses 4 subjects (subj05–subj08) due to local disk constraints during
  development — not a methodological limitation. To reproduce with all 8 subjects:

    1. Download all 8 subject zip files from the Algonauts 2023 challenge:
       https://algonautsproject.com/2023
    2. Extract to the same directory structure used here
    3. Change SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)] (already set to all 8)
    4. Re-run: python scripts/build_algonauts_dataset.py
               python scripts/run_algonauts_experiment.py

  With N=8: fingerprint chance = 12.5%, p(100% accuracy by chance) ≈ 5×10⁻⁸.
  No code changes required — the pipeline scales automatically.

Algonauts data structure (per subject):
  subj0X/
    training_split/
      training_fmri/
        lh_training_fmri.npy   (n_images, ~19004 vertices)
        rh_training_fmri.npy   (n_images, ~20544 vertices)
      training_images/
        train-XXXX_nsd-YYYYY.png
    test_split/
      test_images/
        test-XXXX_nsd-YYYYY.png
    roi_masks/
      lh.prf-visualrois_challenge_space.npy
      lh.floc-faces_challenge_space.npy
      ...
      mapping_prf-visualrois.npy  (dict: int → roi_name)
      mapping_floc-faces.npy
      ...

ROI classes:
  prf-visualrois  → V1v, V1d, V2v, V2d, V3v, V3d, hV4
  floc-bodies     → EBA, FBA-1, FBA-2, mTL-bodies
  floc-faces      → OFA, FFA-1, FFA-2, mTL-faces, aTL-faces
  floc-places     → OPA, PPA, RSC
  floc-words      → OWFA, VWFA-1, VWFA-2, mfs-words, mTL-words
  streams         → early, midventral, midlateral, midparietal,
                    ventral, lateral, parietal
"""
import sys, json, pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
ALG_ROOT  = Path("/path/to/home/Desktop/Delete Later")
FEAT_DIR  = DATA_ROOT / "features"
PROC_DIR  = DATA_ROOT / "algonauts_processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]

# ROI classes and which ones to use for the experiment
ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-faces":     ["OFA", "FFA-1", "FFA-2"],
    "floc-bodies":    ["EBA", "FBA-1", "FBA-2"],
    "floc-places":    ["OPA", "PPA", "RSC"],
    "floc-words":     ["OWFA", "VWFA-1", "VWFA-2"],
    "streams":        ["early", "midventral", "midlateral", "midparietal",
                       "ventral", "lateral", "parietal"],
}
TARGET_ROIS = [roi for rois in ROI_CLASSES.values() for roi in rois]


def load_roi_mask(subj_dir: Path, roi_class: str, roi_name: str,
                  hemi: str = "lh") -> Optional[np.ndarray]:
    """
    Load boolean mask of challenge-space vertices for a named ROI.
    Returns (n_vertices,) bool array or None if ROI not found.
    """
    mask_dir = subj_dir / "roi_masks"

    # Load the ROI class array (int labels per vertex)
    class_file = mask_dir / f"{hemi}.{roi_class}_challenge_space.npy"
    if not class_file.exists():
        return None
    roi_array = np.load(class_file)

    # Load the mapping dict: int → roi_name
    mapping_file = mask_dir / f"mapping_{roi_class}.npy"
    if not mapping_file.exists():
        return None
    mapping = np.load(mapping_file, allow_pickle=True).item()
    # mapping is {int: roi_name_string}
    target_int = None
    for k, v in mapping.items():
        if v == roi_name:
            target_int = k
            break
    if target_int is None:
        return None

    return (roi_array == target_int)


def load_subject_roi_betas(
    subj_id: str,
    rois: List[str] = None,
    hemispheres: Tuple[str, ...] = ("lh", "rh"),
) -> Dict[str, np.ndarray]:
    """
    Load ROI-masked betas for one subject.
    Returns {roi_name: (n_images, n_voxels)} concatenated across hemispheres.
    """
    subj_dir = ALG_ROOT / subj_id
    rois = rois or TARGET_ROIS

    # Load whole-brain betas
    betas_lh = np.load(subj_dir / "training_split" / "training_fmri" / "lh_training_fmri.npy")
    betas_rh = np.load(subj_dir / "training_split" / "training_fmri" / "rh_training_fmri.npy")
    print(f"  {subj_id}: LH {betas_lh.shape}  RH {betas_rh.shape}")

    roi_betas = {}
    for roi_class, roi_names in ROI_CLASSES.items():
        for roi_name in roi_names:
            if roi_name not in rois:
                continue
            hemi_arrays = []
            for hemi, betas in [("lh", betas_lh), ("rh", betas_rh)]:
                mask = load_roi_mask(subj_dir, roi_class, roi_name, hemi)
                if mask is not None and mask.sum() > 0:
                    hemi_arrays.append(betas[:, mask])
            if hemi_arrays:
                combined = np.concatenate(hemi_arrays, axis=1)
                roi_betas[roi_name] = combined

    found = sorted(roi_betas.keys())
    print(f"  {subj_id}: {len(found)} ROIs loaded — {found}")
    return roi_betas


def get_image_order(subj_id: str) -> List[str]:
    """Return training image filenames in the order they correspond to fMRI rows."""
    img_dir = ALG_ROOT / subj_id / "training_split" / "training_images"
    # Files named train-XXXX_nsd-YYYYY.png — sort by the train index
    files = sorted(img_dir.glob("train-*.png"),
                   key=lambda p: int(p.stem.split("-")[1].split("_")[0]))
    return [f.name for f in files]


def extract_clip_features(image_names: List[str], subj_id: str,
                           batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Extract CLIP ViT-L/14 features for a subject's training images."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    feat_path = FEAT_DIR / f"algonauts_{subj_id}_clip_features.npy"
    if feat_path.exists():
        print(f"  Features already exist: {feat_path.name}")
        return np.load(feat_path)

    img_dir = ALG_ROOT / subj_id / "training_split" / "training_images"
    print(f"  Loading CLIP ViT-L/14 on {device}...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval()

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size),
                      desc=f"  {subj_id} features"):
            batch = [Image.open(img_dir / n).convert("RGB")
                     for n in image_names[i:i+batch_size]]
            inp = processor(images=batch, return_tensors="pt").to(device)
            out = model.vision_model(pixel_values=inp["pixel_values"])
            all_feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    np.save(feat_path, features)
    print(f"  Saved: {feat_path.name}  {features.shape}")
    return features


def build_shared_image_features(device: str = "cpu") -> Tuple[np.ndarray, List[str]]:
    """
    Build a single shared feature matrix for the 1,000 NSD images
    seen by all 8 subjects. Used for cross-subject RSA and fingerprinting.
    Returns (features, image_names) sorted by NSD ID.
    """
    shared_feat_path = FEAT_DIR / "algonauts_shared1000_clip_features.npy"
    shared_names_path = FEAT_DIR / "algonauts_shared1000_names.npy"

    if shared_feat_path.exists() and shared_names_path.exists():
        return (np.load(shared_feat_path),
                list(np.load(shared_names_path, allow_pickle=True)))

    # Find the 1000 shared images: appear in training set of ALL subjects
    # NSD image ID is in filename: train-XXXX_nsd-YYYYY.png
    subj_nsd_ids = {}
    for subj_id in SUBJECTS:
        subj_dir = ALG_ROOT / subj_id
        if not (subj_dir / "training_split" / "training_images").exists():
            continue
        img_dir = subj_dir / "training_split" / "training_images"
        nsd_to_fname = {}
        for p in img_dir.glob("train-*.png"):
            parts = p.stem.split("_nsd-")
            if len(parts) == 2:
                nsd_to_fname[int(parts[1])] = p.name
        subj_nsd_ids[subj_id] = nsd_to_fname

    if not subj_nsd_ids:
        raise RuntimeError("No subject training images found. Check ALG_ROOT path.")

    shared_nsd_ids = sorted(set.intersection(*[set(d.keys()) for d in subj_nsd_ids.values()]))
    print(f"Shared NSD images (all subjects): {len(shared_nsd_ids)}")

    # Use subj01's copies for feature extraction
    ref_subj = SUBJECTS[0]
    ref_dir = ALG_ROOT / ref_subj / "training_split" / "training_images"
    image_names = [subj_nsd_ids[ref_subj][nsd_id] for nsd_id in shared_nsd_ids]

    import torch
    from transformers import CLIPModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval()

    all_feats = []
    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(image_names), batch_size), desc="Shared image features"):
            batch = [Image.open(ref_dir / n).convert("RGB") for n in image_names[i:i+batch_size]]
            inp = processor(images=batch, return_tensors="pt").to(device)
            out = model.vision_model(pixel_values=inp["pixel_values"])
            all_feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    np.save(shared_feat_path, features)
    np.save(shared_names_path, np.array(image_names))
    print(f"Saved shared features: {features.shape}")
    return features, image_names


def build_dataset(device: str = "cpu", force: bool = False):
    """Build complete processed dataset for all available subjects."""
    print("="*60)
    print("Building Algonauts 2023 dataset")
    print("="*60)

    available = [s for s in SUBJECTS
                 if (ALG_ROOT / s / "training_split" / "training_fmri"
                     / "lh_training_fmri.npy").exists()]
    if not available:
        print(f"No subject data found at:\n  {ALG_ROOT}")
        print("Waiting for downloads to complete.")
        return None, None

    print(f"Available subjects: {available}")

    # Per-subject features + betas
    all_roi_betas = {}
    all_features  = {}
    all_img_names = {}

    for subj_id in available:
        print(f"\n--- {subj_id} ---")
        subj_proc = PROC_DIR / subj_id
        subj_proc.mkdir(exist_ok=True)

        # Image order
        img_names = get_image_order(subj_id)
        all_img_names[subj_id] = img_names
        np.save(subj_proc / "image_names.npy", np.array(img_names))
        print(f"  Training images: {len(img_names)}")

        # CLIP features
        feats = extract_clip_features(img_names, subj_id, device=device)
        all_features[subj_id] = feats

        # ROI betas
        betas_cache = subj_proc / "roi_betas_cache.pkl"
        if betas_cache.exists() and not force:
            with open(betas_cache, "rb") as f:
                roi_betas = pickle.load(f)
            print(f"  Loaded cached ROI betas: {sorted(roi_betas.keys())}")
        else:
            roi_betas = load_subject_roi_betas(subj_id)
            with open(betas_cache, "wb") as f:
                pickle.dump(roi_betas, f)

        # Save individual ROI files
        for roi, betas in roi_betas.items():
            np.save(subj_proc / f"{roi}_betas.npy", betas)

        all_roi_betas[subj_id] = roi_betas

    # Shared 1000-image features (for cross-subject evaluation)
    print("\n--- Shared images (cross-subject RSA/fingerprinting) ---")
    shared_feats, shared_names = build_shared_image_features(device=device)

    # Build shared-image beta index arrays per subject
    for subj_id in available:
        img_names = all_img_names[subj_id]
        img_to_idx = {name: i for i, name in enumerate(img_names)}
        shared_indices = []
        valid_shared = []
        for name in shared_names:
            if name in img_to_idx:
                shared_indices.append(img_to_idx[name])
                valid_shared.append(name)
        np.save(PROC_DIR / subj_id / "shared_image_indices.npy", np.array(shared_indices))
        print(f"  {subj_id}: {len(shared_indices)} shared images mapped")

    # Save metadata
    meta = {
        "subjects": available,
        "rois": TARGET_ROIS,
        "n_shared": len(shared_names),
        "dataset": "Algonauts2023_NSD",
    }
    with open(PROC_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset ready: {len(available)} subjects, {len(TARGET_ROIS)} ROIs")
    return all_features, all_roi_betas


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_dataset(device=args.device, force=args.force)
