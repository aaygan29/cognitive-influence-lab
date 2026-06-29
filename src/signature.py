"""The neural belief-formation signature B(s).

B(s) = E + V - R, where
  E = MPFC engagement  (Falk et al. 2010: MPFC predicts persuasion-induced behavior change)
  V = NAcc value       (Genevsky et al. 2017; Knutson et al. 2018: NAcc forecasts aggregate choice)
  R = DMN + dmPFC      (Kaplan et al. 2016; Kossowska et al. 2026: belief-protective resistance)

The three ROI scores are read from a frozen encoder's predicted activation profile. In a real run
the ROI masks come from Neurosynth meta-analytic maps (Yarkoni et al. 2011); here, where the encoder
already returns named ROI activations, the masks reduce to a fixed, documented linear read-out so the
pipeline is auditable. B(s) is a hypothesis to be validated (see analysis.py), not an assumed truth.

No em dashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .encoders import BrainEncoder, EncoderOutput


@dataclass(frozen=True)
class BeliefSignature:
    E: float       # engagement (MPFC)
    V: float       # value (NAcc)
    R: float       # resistance (DMN + dmPFC)
    B: float       # E + V - R
    encoder_name: str


def signature_from_activations(out: EncoderOutput) -> BeliefSignature:
    a = out.activations
    E = float(a["MPFC"])
    V = float(a["NAcc"])
    R = float(a["DMN"] + a["dmPFC"]) / 2.0
    B = E + V - R
    return BeliefSignature(E=E, V=V, R=R, B=B, encoder_name=out.encoder_name)


def compute_signatures(features_rows, encoder: BrainEncoder) -> np.ndarray:
    """Return an array of B(s) for a list/iterable of feature dicts under one encoder."""
    vals = []
    for feats in features_rows:
        out = encoder.predict(feats)
        vals.append(signature_from_activations(out).B)
    return np.asarray(vals, dtype=float)


def compute_signature_table(features_rows, encoder: BrainEncoder):
    """Return E, V, R, B columns as a dict of arrays (for the fusion model indicators)."""
    E, V, R, B = [], [], [], []
    for feats in features_rows:
        sig = signature_from_activations(encoder.predict(feats))
        E.append(sig.E); V.append(sig.V); R.append(sig.R); B.append(sig.B)
    return {
        "E": np.asarray(E),
        "V": np.asarray(V),
        "R": np.asarray(R),
        "B": np.asarray(B),
    }
