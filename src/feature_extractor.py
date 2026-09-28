"""
Extract DNN features from stimuli using CLIP ViT-L/14.
Features serve as input to the encoding models.
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


FEATURES_ROOT = Path(__file__).parent.parent / "data" / "features"
STIMULI_ROOT = Path(__file__).parent.parent / "data" / "stimuli"

MODEL_ID = "openai/clip-vit-large-patch14"


class CLIPFeatureExtractor:
    """Extracts frozen CLIP ViT-L/14 image features."""

    def __init__(self, device: str = "cpu", layer: str = "vision_last"):
        self.device = device
        self.layer = layer
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            print(f"Loading {MODEL_ID}...")
            self._processor = CLIPProcessor.from_pretrained(MODEL_ID)
            self._model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
            self._model.eval()

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        self._load()
        inputs = self._processor(images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self._model.vision_model(**{k: v for k, v in inputs.items() if k != "pixel_values"},
                                           pixel_values=inputs["pixel_values"])
        # CLS token from last hidden state
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return features

    def extract_from_array(self, image_array: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Extract features from (N, H, W, C) uint8 array."""
        self._load()
        n = len(image_array)
        all_features = []
        for i in tqdm(range(0, n, batch_size), desc="Extracting CLIP features"):
            batch = image_array[i:i+batch_size]
            imgs = [Image.fromarray(img) for img in batch]
            feats = self.extract_batch(imgs)
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)

    def extract_from_paths(self, paths: List[Path], batch_size: int = 64) -> np.ndarray:
        self._load()
        all_features = []
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting CLIP features"):
            batch_paths = paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            feats = self.extract_batch(imgs)
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)


def extract_and_save(
    image_source: Union[np.ndarray, List[Path]],
    output_name: str = "clip",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Extract features and save to disk. Returns features array."""
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_ROOT / f"{output_name}_features.npy"

    if out_path.exists():
        print(f"Features already exist at {out_path}, loading...")
        return np.load(out_path)

    extractor = CLIPFeatureExtractor(device=device)
    if isinstance(image_source, np.ndarray):
        features = extractor.extract_from_array(image_source, batch_size=batch_size)
    else:
        features = extractor.extract_from_paths(image_source, batch_size=batch_size)

    np.save(out_path, features)
    print(f"Saved features: {features.shape} → {out_path}")
    return features


def load_stimuli_array() -> Optional[np.ndarray]:
    """Load the Algonauts training images as a numpy array if available."""
    path = STIMULI_ROOT / "training_images.npy"
    if path.exists():
        print(f"Loading stimuli from {path}...")
        return np.load(path)
    return None
