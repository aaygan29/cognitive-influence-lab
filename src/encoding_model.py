"""
Subject-specific encoding models: DNN features → fMRI voxel responses.
Each subject gets its own fitted model per ROI.
This is the "digital brain" — the function f_subj(stimulus) → predicted_brain_response.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm import tqdm


MODELS_ROOT = Path(__file__).parent.parent / "results" / "models"


class DigitalBrain:
    """
    A subject-specific encoding model that maps stimuli to predicted brain responses.
    This IS the digital brain — a personalized function approximating the subject's
    visual cortex.
    """

    def __init__(self, subj_id: int, alphas=None):
        self.subj_id = subj_id
        self.alphas = alphas or [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        self.roi_models: Dict[str, Pipeline] = {}
        self.roi_voxel_counts: Dict[str, int] = {}
        self._is_fitted = False

    def fit(self, features: np.ndarray, roi_betas: Dict[str, np.ndarray], verbose: bool = True) -> "DigitalBrain":
        """
        Train per-ROI encoding models.
        features: (n_train, feature_dim)
        roi_betas: {roi_name: (n_train, n_voxels)}
        """
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"\nTraining digital brain for subject {self.subj_id:02d}...")

        for roi, betas in (tqdm(roi_betas.items(), desc=f"subj{self.subj_id:02d}") if verbose else roi_betas.items()):
            n_voxels = betas.shape[1]
            self.roi_voxel_counts[roi] = n_voxels

            # PCA to reduce voxel dimensionality (speeds up fitting, improves generalization)
            n_components = min(100, n_voxels, features.shape[0] - 1)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=42)),
                ("ridge", RidgeCV(alphas=self.alphas, cv=3, scoring="r2")),
            ])

            # Fit on voxel PCs (predict PC scores, not raw voxels)
            from sklearn.decomposition import PCA as _PCA
            voxel_pca = _PCA(n_components=n_components, random_state=42)
            betas_reduced = voxel_pca.fit_transform(betas)

            pipeline.fit(features, betas_reduced)
            self.roi_models[roi] = (pipeline, voxel_pca)

        self._is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict brain responses to new stimuli. Returns {roi: (n_images, n_voxels_pcs)}."""
        if not self._is_fitted:
            raise RuntimeError("Digital brain not fitted. Call fit() first.")
        predictions = {}
        for roi, (pipeline, voxel_pca) in self.roi_models.items():
            pred_pcs = pipeline.predict(features)
            # Reconstruct approximate voxel space
            pred_voxels = voxel_pca.inverse_transform(pred_pcs)
            predictions[roi] = pred_voxels
        return predictions

    def predict_roi(self, features: np.ndarray, roi: str) -> np.ndarray:
        if roi not in self.roi_models:
            raise KeyError(f"ROI '{roi}' not in this digital brain. Available: {list(self.roi_models)}")
        pipeline, voxel_pca = self.roi_models[roi]
        return voxel_pca.inverse_transform(pipeline.predict(features))

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = MODELS_ROOT / f"digital_brain_subj{self.subj_id:02d}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved: {path}")

    @staticmethod
    def load(subj_id: int, path: Optional[Path] = None) -> "DigitalBrain":
        if path is None:
            path = MODELS_ROOT / f"digital_brain_subj{subj_id:02d}.pkl"
        with open(path, "rb") as f:
            brain = pickle.load(f)
        return brain

    @staticmethod
    def load_if_exists(subj_id: int) -> Optional["DigitalBrain"]:
        path = MODELS_ROOT / f"digital_brain_subj{subj_id:02d}.pkl"
        if path.exists():
            return DigitalBrain.load(subj_id, path)
        return None

    def fitted_rois(self) -> List[str]:
        return list(self.roi_models.keys())

    def __repr__(self):
        status = "fitted" if self._is_fitted else "unfitted"
        rois = list(self.roi_models.keys()) if self._is_fitted else []
        return f"DigitalBrain(subj={self.subj_id:02d}, status={status}, rois={rois})"


def train_all_digital_brains(
    subject_ids: List[int],
    features_train: np.ndarray,
    all_roi_betas: Dict[int, Dict[str, np.ndarray]],
    force_retrain: bool = False,
) -> Dict[int, DigitalBrain]:
    """
    Train digital brains for all subjects.
    Returns {subj_id: DigitalBrain}.
    """
    digital_brains = {}
    for subj_id in subject_ids:
        existing = DigitalBrain.load_if_exists(subj_id)
        if existing is not None and not force_retrain:
            print(f"Loaded cached digital brain for subject {subj_id:02d}")
            digital_brains[subj_id] = existing
            continue

        brain = DigitalBrain(subj_id=subj_id)
        brain.fit(features_train, all_roi_betas[subj_id])
        brain.save()
        digital_brains[subj_id] = brain

    return digital_brains
