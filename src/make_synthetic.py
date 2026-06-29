"""Synthetic stand-in with a KNOWN ground truth.

This is a pipeline-validation harness, not a result. It generates persuasive items with interpretable
content features, a true latent belief that is driven by a content component and (optionally) a neural
component, and several noisy indicators of that latent belief. With the neural weight on (positive
control) the pipeline should detect the neural signal; with it off (negative control) the pipeline
should report null. The real run replaces this with public outcome corpora (see docs/DATA.md).

The neural drive is exactly the reference-encoder B(s), so a positive control is a setting where a
genuine neural quantity drives outcomes. Because B(s) is a nonlinear (tanh) combination of the content
channels, a linear content-only model cannot fully reproduce it, which is what leaves room for B(s) to
add incremental value when the neural weight is on, and no room when it is off.

No em dashes.
"""

from __future__ import annotations

import numpy as np

from .encoders import CONTENT_FEATURES, ReferenceEncoder
from .signature import signature_from_activations


def _standardize(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def generate(
    n: int = 1200,
    neural_weight: float = 0.9,
    content_weight: float = 1.0,
    noise: float = 0.7,
    seed: int = 7,
) -> dict:
    """Return a dict of arrays: content features, true neural drive N, latent L, and indicators.

    Parameters
    ----------
    neural_weight : beta. 0.0 = negative control (no neural signal in outcomes).
    content_weight : alpha. Weight on the content-only component.
    noise : standard deviation of latent noise.
    """
    rng = np.random.default_rng(seed)
    enc = ReferenceEncoder()

    # surface content features in [0, 1] (the baseline sees these)
    F = rng.beta(2.0, 2.0, size=(n, len(CONTENT_FEATURES)))
    # response-relevant latent the brain encoder reads but surface features miss (baseline cannot see)
    g = rng.beta(2.0, 2.0, size=n)
    rows = [
        {**{k: F[i, j] for j, k in enumerate(CONTENT_FEATURES)}, "response_latent": float(g[i])}
        for i in range(n)
    ]

    # the brain-encoder signature reads surface features AND the response latent
    N = np.array([signature_from_activations(enc.predict(r)).B for r in rows])

    # a content-only linear functional the baseline can capture (surface features only)
    w_content = rng.normal(0.0, 1.0, size=len(CONTENT_FEATURES))
    content_lin = F @ w_content

    # the neural component of belief is the response latent g, which is invisible to the baseline.
    # B(s) adds value only insofar as it carries g. neural_weight=0 removes the neural signal.
    zc = _standardize(content_lin)
    zn = _standardize(g)
    L = content_weight * zc + neural_weight * zn + rng.normal(0.0, noise, size=n)

    # indicators of the latent belief (each noisy), all from public-style measurements
    p_behavior = 1.0 / (1.0 + np.exp(-L))                 # behavior: argument delta / donation
    behavior = (rng.uniform(size=n) < p_behavior).astype(int)
    self_report = L + rng.normal(0.0, 0.9, size=n)         # attitude scale
    certainty = np.abs(L) + rng.normal(0.0, 0.9, size=n)   # attitude strength
    mist = 0.7 * L + rng.normal(0.0, 1.1, size=n)          # discernment-style indicator

    return {
        "features": F,
        "feature_names": CONTENT_FEATURES,
        "response_latent": g,
        "rows": rows,
        "true_neural_drive": N,
        "latent_true": L,
        "behavior": behavior,
        "self_report": self_report,
        "certainty": certainty,
        "mist": mist,
        "meta": {
            "n": n,
            "neural_weight": neural_weight,
            "content_weight": content_weight,
            "noise": noise,
            "seed": seed,
        },
    }
