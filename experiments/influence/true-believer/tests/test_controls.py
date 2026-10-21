"""Positive-and-negative-control tests for the instrument.

These lock the discipline: the pipeline must DETECT a neural signal when it is present, report NULL
when it is absent (content-only world), and report CHANCE when the signature is scrambled. If any of
these flips, the instrument is not trustworthy and the tests fail loudly.

No em dashes.
"""

import numpy as np

from src.make_synthetic import generate
from src.encoders import get_encoder, ReferenceEncoder, PerturbedEncoder, CONTENT_FEATURES, ENCODER_FEATURES
from src.signature import compute_signatures, signature_from_activations
from src.fusion import estimate_latent_factor, h_load
from src.analysis import h_incremental, h_bound, conformal_coverage


def _indicators(data):
    return {
        "behavior": data["behavior"],
        "self_report": data["self_report"],
        "certainty": data["certainty"],
        "mist": data["mist"],
    }


def test_positive_control_detects_signal():
    data = generate(n=1400, neural_weight=0.9, seed=7)
    latent = estimate_latent_factor(_indicators(data))
    b = compute_signatures(data["rows"], get_encoder("reference"))
    load = h_load(b, latent)
    incr = h_incremental(data["rows"], data["behavior"])["finding"]
    assert load.passed, f"H-load should pass when signal present (r={load.value})"
    assert incr["passed"], f"H-incremental should pass when signal present (dAUC={incr['value']})"
    assert incr["value"] > 0


def test_content_only_world_reports_null_increment():
    data = generate(n=1400, neural_weight=0.0, seed=7)
    incr = h_incremental(data["rows"], data["behavior"])["finding"]
    assert not incr["passed"], "H-incremental must be null when there is no neural signal"


def test_scrambled_signature_goes_to_chance():
    data = generate(n=1400, neural_weight=0.9, seed=7)
    latent = estimate_latent_factor(_indicators(data))
    b = compute_signatures(data["rows"], get_encoder("reference"))
    rng = np.random.default_rng(123)
    b_scram = b[rng.permutation(len(b))]
    load = h_load(b_scram, latent)
    bound = h_bound(data["rows"], data["behavior"], signature=b_scram)["finding"]
    incr = h_incremental(data["rows"], data["behavior"], signature=b_scram)["finding"]
    assert not load.passed, "scrambled B must not load"
    assert not incr["passed"], "scrambled B must add nothing"
    assert abs(bound["value"] - 0.5) < 0.05, "scrambled B AUC must be near chance"


def test_encoder_feature_contract():
    assert ENCODER_FEATURES[-1] == "response_latent"
    assert len(CONTENT_FEATURES) == 6
    # baseline cannot see the response latent; encoder can
    enc = ReferenceEncoder()
    out_with = enc.predict({**{k: 0.5 for k in CONTENT_FEATURES}, "response_latent": 1.0})
    out_without = enc.predict({**{k: 0.5 for k in CONTENT_FEATURES}, "response_latent": 0.0})
    b_with = signature_from_activations(out_with).B
    b_without = signature_from_activations(out_without).B
    assert b_with != b_without, "encoder must respond to the response latent"


def test_perturbed_encoder_is_correlated_not_identical():
    data = generate(n=500, neural_weight=0.9, seed=3)
    b_ref = compute_signatures(data["rows"], ReferenceEncoder())
    b_pert = compute_signatures(data["rows"], PerturbedEncoder(seed=11))
    r = np.corrcoef(b_ref, b_pert)[0, 1]
    assert 0.5 < r < 0.999, f"perturbed encoder should be correlated but distinct (r={r})"
