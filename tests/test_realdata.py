"""Real-data pipeline tests. Skipped if the ChangeMyView corpus is not present (it is not committed).

No em dashes.
"""

import os
import numpy as np
import pytest

from src.realdata import extract_features, signature, zscore

ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(ROOT, "data", "raw", "winning-args-corpus")
HAS_CORPUS = os.path.exists(os.path.join(CORPUS, "utterances.jsonl"))


def test_feature_extractor_is_deterministic_and_typed():
    f1 = extract_features("Because the evidence clearly shows this. Consider http://x.com and 3 studies.")
    f2 = extract_features("Because the evidence clearly shows this. Consider http://x.com and 3 studies.")
    assert f1 == f2
    assert set(["content", "E", "V", "R"]).issubset(f1.keys())
    assert f1["E"] > 0  # evidence + frame present


def test_signature_is_zscored_combination():
    E = np.array([1.0, 2.0, 3.0]); V = np.array([0.0, 1.0, 2.0]); R = np.array([2.0, 1.0, 0.0])
    B = signature(E, V, R)
    assert abs(B.mean()) < 1e-9  # z-scored components sum to mean ~0
    assert B[2] > B[0]  # high E,V low R => high B


@pytest.mark.skipif(not HAS_CORPUS, reason="ChangeMyView corpus not downloaded")
def test_real_within_pair_engagement_is_significant():
    from src.realdata import assemble
    from src.run_real import within_pair_channels
    data = assemble(CORPUS)
    ch = within_pair_channels(data)
    # the engagement channel should win within matched pairs, well above chance
    assert ch["E"]["win_rate"] > 0.55
    assert ch["E"]["wilcoxon_p"] < 1e-10


@pytest.mark.skipif(not HAS_CORPUS, reason="ChangeMyView corpus not downloaded")
def test_real_permutation_null_is_near_half():
    from src.realdata import assemble
    from src.run_real import permutation_null
    data = assemble(CORPUS)
    pn = permutation_null(data, n_perm=60)
    assert abs(pn["null_win_rate_mean"] - 0.5) < 0.03
