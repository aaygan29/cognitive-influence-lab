"""Unit tests for the spiking/rate decision models. Skipped where torch/snntorch are absent (CI)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("snntorch")

from spikeprint.decision_models import (
    RateMLP,
    SpikingMLP,
    predict_proba,
    train_model,
)


def test_spiking_forward_returns_logit_and_nonneg_spikes():
    import torch

    logit, spikes = SpikingMLP(hidden=8, t_steps=5)(torch.zeros(4, 2))
    assert logit.shape == (4,)
    assert float(spikes) >= 0.0


def test_operation_counts():
    assert RateMLP(16).macs_per_decision() == 2 * 16 + 16
    assert SpikingMLP(16, 20).input_macs_per_decision() == 2 * 16 * 20


def test_train_and_predict_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2)).astype("float32")
    y = (x[:, 0] > 0).astype("float32")
    model = train_model(SpikingMLP(8, 5), x, y, epochs=3, spiking=True)
    p, spikes_per_dec = predict_proba(model, x, spiking=True)
    assert p.shape == (50,)
    assert spikes_per_dec is not None and spikes_per_dec >= 0.0
