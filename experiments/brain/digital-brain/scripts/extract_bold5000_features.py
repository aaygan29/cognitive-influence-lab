"""
Extract CLIP ViT-L/14 features for the 1188 shared COCO images from BOLD5000.
Saves to data/features/bold5000_clip_features.npy
Images must be downloaded first (download_bold5000_coco.py).
"""
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

IMG_DIR   = Path(__file__).parent.parent / "data" / "bold5000" / "stimuli" / "coco_images"
META_DIR  = Path(__file__).parent.parent / "data" / "bold5000" / "stimuli"
FEAT_DIR  = Path(__file__).parent.parent / "data" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH  = FEAT_DIR / "bold5000_clip_features.npy"


def extract(batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    if OUT_PATH.exists():
        print(f"Features already exist: {OUT_PATH}")
        return np.load(OUT_PATH)

    # Load image list (must be same order as betas)
    image_names = list(np.load(META_DIR / "shared_coco_images.npy", allow_pickle=True))
    image_names = [str(s) for s in image_names]
    print(f"Images to extract: {len(image_names)}")

    # Verify all images exist
    missing = [n for n in image_names if not (IMG_DIR / n).exists()]
    if missing:
        print(f"Missing {len(missing)} images. Run download script first.")
        print(f"  First missing: {missing[:3]}")
        return None

    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading CLIP ViT-L/14 on {device}...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    model.eval()

    all_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size), desc="Extracting features"):
            batch_names = image_names[i:i+batch_size]
            images = [Image.open(IMG_DIR / n).convert("RGB") for n in batch_names]
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(feats)

    features = np.concatenate(all_features, axis=0)
    np.save(OUT_PATH, features)
    print(f"Saved: {features.shape} → {OUT_PATH}")
    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extract(batch_size=args.batch_size, device=args.device)
