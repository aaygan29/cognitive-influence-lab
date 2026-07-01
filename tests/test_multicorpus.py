"""Cross-corpus test. Skipped if either corpus is absent (neither is committed).

No em dashes.
"""

import os
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
CMV = os.path.join(ROOT, "data", "raw", "winning-args-corpus", "utterances.jsonl")
P4G = os.path.join(ROOT, "data", "raw", "persuasionforgood-corpus", "utterances.jsonl")
HAS_BOTH = os.path.exists(CMV) and os.path.exists(P4G)


@pytest.mark.skipif(not HAS_BOTH, reason="both corpora required")
def test_p4g_loads_and_is_balanced():
    from src.multicorpus import load_p4g
    d = load_p4g()
    assert 800 < len(d["y"]) < 1100
    frac = d["y"].mean()
    assert 0.35 < frac < 0.65  # roughly balanced donate/not


@pytest.mark.skipif(not HAS_BOTH, reason="both corpora required")
def test_engagement_points_same_direction_in_both():
    from src.multicorpus import main
    out = main()
    e_cmv = out["corpus1"]["coefs"]["E"]["sign"]
    e_p4g = out["corpus2"]["coefs"]["E"]["sign"]
    assert e_cmv == e_p4g == 1, "engagement should point positive in both corpora"
    # pooled engagement estimate should be significant (CI excludes 0)
    meta = out["cross_corpus_verdict"]["E"]["meta_ci95"]
    assert meta[0] > 0, "pooled engagement estimate should exclude zero"
