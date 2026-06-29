"""Frozen brain-encoder interface.

An encoder maps a content feature vector to a profile of region-of-interest (ROI) activations
that the belief-formation signature B(s) reads from. The reference encoder is a transparent,
deterministic heuristic (it is honest about being a heuristic, not a validated fMRI model). Real
encoders (TRIBE v2, Mary, Qualia) plug in behind the same interface; their loaders are stubs here
because their weights are not present in this environment.

Content features (interpretable persuasion channels, each in [0, 1]):
  emotional_intensity, evidence, identity_appeal, framing, concession, social_proof

ROI activations returned (each a float, higher = more drive):
  MPFC  - engagement / persuasion-uptake correlate (Falk et al. 2010)
  NAcc  - reward / value, aggregate-choice forecasting correlate (Genevsky 2017; Knutson 2018)
  DMN   - default-mode self-referential belief-protection correlate (Kaplan 2016; Kossowska 2026)
  dmPFC - resistance / belief-maintenance correlate (Kaplan 2016)

No em dashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Surface content features that a content-only model can read off the text.
CONTENT_FEATURES = [
    "emotional_intensity",
    "evidence",
    "identity_appeal",
    "framing",
    "concession",
    "social_proof",
]

# A brain encoder trained on human responses also reads a response-relevant latent that surface
# content features miss (it encodes how a population's brain responds, not just lexical surface).
# The encoder sees this extra channel; the content-only baseline does not. This is the modeled
# source of any incremental value B(s) could have, and exactly the thing the analysis tests for.
ENCODER_FEATURES = CONTENT_FEATURES + ["response_latent"]

ROIS = ["MPFC", "NAcc", "DMN", "dmPFC"]


@dataclass(frozen=True)
class EncoderOutput:
    """ROI activations predicted from content, plus the encoder name for provenance."""
    activations: dict
    encoder_name: str


class BrainEncoder:
    """Interface. A real encoder overrides _predict_rois."""

    name = "abstract"

    def predict(self, features: dict) -> EncoderOutput:
        x = np.array([float(features.get(k, 0.0)) for k in ENCODER_FEATURES], dtype=float)
        x = np.clip(x, 0.0, 1.0)
        rois = self._predict_rois(x)
        return EncoderOutput(activations={r: float(v) for r, v in zip(ROIS, rois)}, encoder_name=self.name)

    def _predict_rois(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReferenceEncoder(BrainEncoder):
    """Deterministic, transparent heuristic mapping content channels to ROI drive.

    The weight matrix encodes only the directions the literature supports, not magnitudes:
      - emotional_intensity and social_proof drive NAcc value and MPFC engagement.
      - evidence drives MPFC engagement and lowers DMN/dmPFC resistance.
      - identity_appeal raises DMN/dmPFC resistance (identity-protective; Kaplan 2016).
      - framing raises MPFC engagement.
      - concession lowers resistance (reduces counterarguing).
    This is a heuristic. It is NOT a validated fMRI model. Swap in a learned encoder for any
    neural claim; the reference encoder exists so the pipeline runs and is auditable end to end.
    """

    name = "reference"

    # rows = ROIS [MPFC, NAcc, DMN, dmPFC]; cols = ENCODER_FEATURES (6 surface + response_latent)
    W = np.array([
        # emo   evid   ident  fram   conc   social  resp_latent
        [0.55, 0.65, 0.05, 0.50, 0.20, 0.40, 0.60],   # MPFC engagement
        [0.70, 0.10, 0.15, 0.20, 0.05, 0.60, 0.55],   # NAcc value
        [0.10, -0.45, 0.70, -0.10, -0.40, 0.15, -0.30],  # DMN resistance
        [0.05, -0.50, 0.65, -0.05, -0.45, 0.10, -0.25],  # dmPFC resistance
    ], dtype=float)

    def _predict_rois(self, x: np.ndarray) -> np.ndarray:
        raw = self.W @ x
        # squash to a stable, comparable range; keep sign structure
        return np.tanh(raw)


class PerturbedEncoder(ReferenceEncoder):
    """A reference encoder with a fixed, seeded perturbation of its weights.

    Used ONLY to exercise the encoder-swap robustness check (H-encoder) end to end on the
    committed synthetic stand-in, standing in for genuinely independent encoders (TRIBE / Mary /
    Qualia) until their weights are wired. A real swap replaces this with a real second encoder.
    """

    def __init__(self, seed: int, scale: float = 0.15, name: str | None = None):
        rng = np.random.default_rng(seed)
        self._W = self.W + rng.normal(0.0, scale, size=self.W.shape)
        self.name = name or f"perturbed_{seed}"

    def _predict_rois(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self._W @ x)


class TribeV2Encoder(BrainEncoder):
    """Stub for the frozen TRIBE v2 encoder. Wire weights to enable real runs."""

    name = "tribe_v2"

    def _predict_rois(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TRIBE v2 weights are not present in this environment. Place the open weights per "
            "DATA_MANIFEST.json and implement the content->predicted-BOLD->ROI projection here, "
            "then register this encoder in get_encoder()."
        )


def get_encoder(name: str) -> BrainEncoder:
    if name == "reference":
        return ReferenceEncoder()
    if name == "tribe_v2":
        return TribeV2Encoder()
    if name.startswith("perturbed_"):
        seed = int(name.split("_", 1)[1])
        return PerturbedEncoder(seed=seed, name=name)
    raise ValueError(f"unknown encoder: {name}")
