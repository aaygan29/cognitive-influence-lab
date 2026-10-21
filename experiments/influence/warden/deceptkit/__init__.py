"""
deceptkit — a calibrated, CIT-first multimodal concealed-information detector.

Not a "lie detector." It estimates, per question set, the probability that a
subject recognizes a concealed detail, with an honest confidence interval and an
explicit ABSTAIN action for low-quality or countermeasure-compromised cases.

Pipeline:
    simulate_cit / your data  ->  build_features  ->  DeceptionModel.fit/.decide
    ->  evaluate.*

See README for the scientific rationale and the mapping from this reference
implementation to a real sensor deployment.
"""
from .simulate import (simulate_cit, MODALITIES, D_SINGLE, CM_TYPES,
                        nonresponder_fraction)
from .features import build_features, feature_columns
from .model import DeceptionModel, DecisionPolicy, fit_predict_crossval
from .fusion import (BayesianFusion, MultivariateFusion, SPRTPolicy,
                     fit_predict_crossval_fusion, EVIDENCE_CHANNELS)
from . import evaluate

__all__ = [
    "simulate_cit", "MODALITIES", "D_SINGLE", "CM_TYPES", "nonresponder_fraction",
    "build_features", "feature_columns",
    "DeceptionModel", "DecisionPolicy", "fit_predict_crossval",
    "BayesianFusion", "MultivariateFusion", "SPRTPolicy", "fit_predict_crossval_fusion",
    "EVIDENCE_CHANNELS",
    "evaluate",
]
__version__ = "0.3.0"
