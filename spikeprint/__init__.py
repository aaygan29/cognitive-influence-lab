"""spikeprint — behaviorally-validated measurement of choice distortion from content.

This package makes one kind of claim, and it is behavioral: does a content-level metric (CSI)
predict observed choice distortion? It does not detect "manipulation" and does not read any
individual's neural state. See PREREGISTRATION.md (section 1).
"""
from .validate import (
    Finding,
    UnvalidatedClaimError,
    benjamini_hochberg,
    bootstrap_ci,
    decide_family,
    gate,
)

__all__ = [
    "Finding",
    "UnvalidatedClaimError",
    "benjamini_hochberg",
    "bootstrap_ci",
    "decide_family",
    "gate",
]
__version__ = "0.0.1"
