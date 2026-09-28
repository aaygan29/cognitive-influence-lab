"""
Geometry-aware encoding model for digital brain construction.

Standard ridge regression optimizes:
    L = ||f(X) - Y||²   (amplitude only)

This model optimizes:
    L = α * MSE(f(X), Y) + β * RDM_divergence(f(X), Y)

where RDM_divergence measures the distance between the pairwise geometry
of predicted vs actual responses — forcing the digital brain to preserve
the subject's internal representational structure, not just response magnitudes.

This directly addresses the RSA paradox: a model optimized only for amplitude
will fingerprint correctly (amplitude is subject-specific) but fail RSA
(geometry is not constrained to match).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Optional
import pickle
from pathlib import Path


# ── Geometry losses ──────────────────────────────────────────────────────────

def soft_rdm(activations: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable RDM via cosine dissimilarity.
    activations: (N, D)
    Returns: (N, N) dissimilarity matrix
    """
    normed = nn.functional.normalize(activations, dim=1)
    sim = normed @ normed.T                     # (N, N) cosine similarity
    return (1.0 - sim) / temperature            # dissimilarity ∈ [0, 2]


def rdm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Frobenius distance between predicted and target RDMs.
    Penalizes any mismatch in pairwise representational geometry.
    """
    rdm_pred   = soft_rdm(pred)
    rdm_target = soft_rdm(target)
    # Use upper triangle only (symmetric matrix, diagonal is zero)
    n = pred.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    v_pred   = rdm_pred[idx[0], idx[1]]
    v_target = rdm_target[idx[0], idx[1]]
    return nn.functional.mse_loss(v_pred, v_target)


def differentiable_rank(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Differentiable soft rank via pairwise comparisons (Blondel et al. 2020).
    For each element x_i, soft rank ≈ sum_j sigmoid((x_i - x_j) / tau).
    tau controls sharpness: smaller → closer to hard rank, but harder gradients.
    """
    # x: (N,) → pairwise differences (N, N)
    diff = x.unsqueeze(1) - x.unsqueeze(0)           # (N, N)
    soft_indicator = torch.sigmoid(diff / tau)         # ≈ 1 if x_i > x_j
    return soft_indicator.sum(dim=1)                   # (N,) soft ranks


def rank_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Differentiable approximation of 1 - Spearman_rho(RDM_pred, RDM_target).
    Uses pairwise sigmoid comparisons for soft ranking (fully differentiable).
    """
    n = pred.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    rdm_pred   = soft_rdm(pred)[idx[0], idx[1]]
    rdm_target = soft_rdm(target.detach())[idx[0], idx[1]]

    rank_p = differentiable_rank(rdm_pred,   tau=tau)
    rank_t = differentiable_rank(rdm_target, tau=tau)

    # Pearson correlation on soft ranks ≈ Spearman
    rank_p = rank_p - rank_p.mean()
    rank_t = rank_t - rank_t.mean()
    denom = (rank_p.norm() * rank_t.norm()).clamp(min=1e-8)
    rho = (rank_p * rank_t).sum() / denom
    return 1.0 - rho


class GeometryLoss(nn.Module):
    """Combined amplitude + geometry loss."""
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha   # MSE weight (amplitude)
        self.beta  = beta    # RDM Frobenius weight (geometry magnitude)
        self.gamma = gamma   # Rank loss weight (geometry order)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse  = nn.functional.mse_loss(pred, target)
        geom = rdm_loss(pred, target)
        rank = rank_loss(pred, target)
        total = self.alpha * mse + self.beta * geom + self.gamma * rank
        return {
            "total": total,
            "mse": mse.detach(),
            "geom": geom.detach(),
            "rank": rank.detach(),
        }


# ── Encoder network ──────────────────────────────────────────────────────────

class NeuralEncoder(nn.Module):
    """
    Small MLP: CLIP features → voxel PC scores.
    Deeper than ridge regression, enabling non-linear subject-specific tuning.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Geometry-aware digital brain ─────────────────────────────────────────────

class GeometryAwareDigitalBrain:
    """
    Digital brain trained with a dual objective:
    1. Predict voxel response amplitudes (MSE)
    2. Preserve representational geometry (RDM divergence)

    This directly addresses the RSA paradox observed in ridge regression:
    amplitude-only optimization fingerprints correctly but fails RSA self-advantage.
    """

    def __init__(
        self,
        subj_id: str,
        alpha: float = 1.0,     # MSE weight
        beta: float = 0.5,      # RDM geometry weight
        gamma: float = 0.3,     # Rank correlation weight
        hidden_dim: int = 512,
        n_voxel_pcs: int = 100,
        n_epochs: int = 200,
        lr: float = 3e-4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.subj_id    = subj_id
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.hidden_dim = hidden_dim
        self.n_voxel_pcs = n_voxel_pcs
        self.n_epochs   = n_epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = device

        self.roi_encoders: Dict[str, tuple] = {}  # roi → (encoder, scaler, vpca)
        self.train_history: Dict[str, list] = {}

    def fit(self, features: np.ndarray, roi_betas: Dict[str, np.ndarray],
            verbose: bool = True) -> "GeometryAwareDigitalBrain":

        if verbose:
            print(f"\n  Geometry-aware training: {self.subj_id}")
            print(f"  Loss weights: MSE={self.alpha} | RDM={self.beta} | Rank={self.gamma}")

        for roi, betas in roi_betas.items():
            n_vox = betas.shape[1]
            n_pcs = min(self.n_voxel_pcs, n_vox, features.shape[0] - 1)

            # Reduce voxel dimensionality
            vpca = PCA(n_components=n_pcs, random_state=42)
            betas_pcs = vpca.fit_transform(betas).astype(np.float32)

            # Normalise inputs
            scaler = StandardScaler()
            X = scaler.fit_transform(features).astype(np.float32)

            X_t  = torch.from_numpy(X).to(self.device)
            Y_t  = torch.from_numpy(betas_pcs).to(self.device)

            # Build encoder
            encoder = NeuralEncoder(
                input_dim=X.shape[1],
                output_dim=n_pcs,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

            criterion = GeometryLoss(self.alpha, self.beta, self.gamma)
            optimizer = optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

            dataset = TensorDataset(X_t, Y_t)
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            history = {"total": [], "mse": [], "geom": [], "rank": []}
            encoder.train()
            for epoch in range(self.n_epochs):
                epoch_losses = {k: 0.0 for k in history}
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = encoder(xb)
                    losses = criterion(pred, yb)
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    optimizer.step()
                    for k in epoch_losses:
                        epoch_losses[k] += losses[k].item()
                scheduler.step()
                for k in history:
                    history[k].append(epoch_losses[k] / len(loader))

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"    {roi} epoch {epoch+1}/{self.n_epochs}: "
                          f"total={history['total'][-1]:.4f}  "
                          f"mse={history['mse'][-1]:.4f}  "
                          f"geom={history['geom'][-1]:.4f}  "
                          f"rank={history['rank'][-1]:.4f}")

            self.roi_encoders[roi] = (encoder, scaler, vpca)
            self.train_history[roi] = history

        return self

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        preds = {}
        for roi, (encoder, scaler, vpca) in self.roi_encoders.items():
            encoder.eval()
            X = scaler.transform(features).astype(np.float32)
            X_t = torch.from_numpy(X).to(self.device)
            pred_pcs = encoder(X_t).cpu().numpy()
            preds[roi] = vpca.inverse_transform(pred_pcs)
        return preds

    def predict_roi(self, features: np.ndarray, roi: str) -> np.ndarray:
        encoder, scaler, vpca = self.roi_encoders[roi]
        encoder.eval()
        with torch.no_grad():
            X = torch.from_numpy(scaler.transform(features).astype(np.float32)).to(self.device)
            return vpca.inverse_transform(encoder(X).cpu().numpy())

    def fitted_rois(self):
        return list(self.roi_encoders.keys())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Move encoders to CPU before pickling
        for roi in self.roi_encoders:
            enc, sc, pca = self.roi_encoders[roi]
            enc.cpu()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved: {path.name}")

    @staticmethod
    def load(path: Path) -> "GeometryAwareDigitalBrain":
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_if_exists(subj: str, model_dir: Path) -> Optional["GeometryAwareDigitalBrain"]:
        p = model_dir / f"geo_brain_{subj}.pkl"
        return GeometryAwareDigitalBrain.load(p) if p.exists() else None


# ── RSA delta diagnostic ──────────────────────────────────────────────────────

def rsa_self_advantage(
    brain: "GeometryAwareDigitalBrain",
    features_rsa: np.ndarray,
    real_rsa_self: Dict[str, np.ndarray],
    real_rsa_others: list,
) -> Dict[str, Dict]:
    """
    Compute RSA self-advantage for each ROI.
    Returns {roi: {rsa_self, rsa_other_mean, advantage}}.
    This is the key metric for the RSA paradox — positive advantage means
    the digital brain's geometry matches its own subject better than others.
    """
    from scipy.spatial.distance import pdist, squareform
    preds = brain.predict(features_rsa)
    n = features_rsa.shape[0]
    idx = np.triu_indices(n, k=1)

    def rdm_vec(act):
        return squareform(pdist(act, metric="correlation"))[idx]

    results = {}
    for roi, pred in preds.items():
        if roi not in real_rsa_self:
            continue
        rdm_p = rdm_vec(pred)
        rdm_s = rdm_vec(real_rsa_self[roi])
        r_self = float(spearmanr(rdm_p, rdm_s)[0])

        r_others = []
        for other_rsa in real_rsa_others:
            if roi in other_rsa:
                r_others.append(float(spearmanr(rdm_p, rdm_vec(other_rsa[roi]))[0]))

        results[roi] = {
            "rsa_self":  r_self,
            "rsa_other": float(np.mean(r_others)) if r_others else float("nan"),
            "advantage": r_self - float(np.mean(r_others)) if r_others else float("nan"),
        }
    return results
