"""Tests for the in-silico belief model. Skipped if corpora absent.

No em dashes.
"""

import os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
CMV = os.path.join(ROOT, "data", "raw", "winning-args-corpus", "utterances.jsonl")
P4G = os.path.join(ROOT, "data", "raw", "persuasionforgood-corpus", "utterances.jsonl")
HAS = os.path.exists(CMV) and os.path.exists(P4G)


def test_brain_state_is_approach_minus_avoidance():
    from src.insilico_brain import brain_state
    from src.realdata import FACTOR_NAMES
    # a content vector high on evidence (approach) should score higher than one high on identity threat
    idx = {f: i for i, f in enumerate(FACTOR_NAMES)}
    hi_appr = np.zeros((1, len(FACTOR_NAMES))); hi_appr[0, idx["evidence"]] = 2.0
    hi_avoid = np.zeros((1, len(FACTOR_NAMES))); hi_avoid[0, idx["identity_threat"]] = 2.0
    assert brain_state(hi_appr)[0] > brain_state(hi_avoid)[0]


@pytest.mark.skipif(not HAS, reason="both corpora required")
def test_evidence_and_identity_threat_are_robust_factors():
    from src.insilico_brain import factor_variance
    rob = factor_variance(n_boot=200)["cross_corpus_robust"]
    assert rob["evidence"]["robust"], "evidence should robustly raise belief across corpora"
    assert rob["identity_threat"]["robust"], "identity threat should robustly lower belief"
    # signs match the brain prior
    assert rob["evidence"]["cmv_coef"] > 0 and rob["evidence"]["p4g_coef"] > 0
    assert rob["identity_threat"]["cmv_coef"] < 0 and rob["identity_threat"]["p4g_coef"] < 0


@pytest.mark.skipif(not HAS, reason="both corpora required")
def test_cross_domain_above_chance_and_optimization_lifts():
    from src.insilico_brain import cross_domain, optimize_counterfactual
    cd = cross_domain()
    assert cd["CMV->P4G"]["auc_brain_state"] > 0.5
    opt = optimize_counterfactual("P4G")
    assert opt["mean_lift"] > 0, "altering robust factors should raise the belief score"
