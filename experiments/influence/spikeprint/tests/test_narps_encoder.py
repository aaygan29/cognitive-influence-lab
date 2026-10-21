"""Documents the root cause of the H1a null: the neurosignal reference encoder is magnitude-blind
for gambles (it tokenizes with ``[a-zA-Z']+`` and discards digits) yet responds to affective
wording. Skipped where neurosignal is not importable (e.g. CI without the separate package).
"""
import pytest

pytest.importorskip("neurosignal")

from neurosignal import analyze

from spikeprint.datasets import render_gamble_text


def _valence(text: str) -> float:
    return analyze(text=text).metrics[0].score


def test_encoder_is_magnitude_blind_for_gambles():
    # identical wording, very different amounts -> identical score (this is what makes H1a null)
    assert _valence(render_gamble_text(40, 5)) == _valence(render_gamble_text(5, 20))


def test_encoder_responds_to_affective_wording():
    pos = _valence("a wonderful exciting amazing reward you will love")
    neg = _valence("a terrible painful frightening loss that will hurt")
    assert pos != neg
