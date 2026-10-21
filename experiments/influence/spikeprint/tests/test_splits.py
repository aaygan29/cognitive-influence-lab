import numpy as np
import pytest

from spikeprint.splits import assert_no_group_leakage, grouped_split, split_hash


def _make_groups():
    """~150 rows over 50 unique problem ids, each id repeated 1-4 times (aggregate-data shape)."""
    rng = np.random.default_rng(0)
    gids = []
    for g in range(50):
        gids.extend([f"p{g}"] * int(rng.integers(1, 5)))
    return gids


def test_grouped_split_has_no_leakage_and_covers_all_rows():
    g = _make_groups()
    s = grouped_split(g, seed=0)
    assert_no_group_leakage(g, s)  # must not raise
    assert sum(len(v) for v in s.values()) == len(g)


def test_grouped_split_is_deterministic():
    g = _make_groups()
    assert split_hash(grouped_split(g, seed=0)) == split_hash(grouped_split(g, seed=0))


def test_grouped_split_changes_with_seed():
    g = _make_groups()
    assert split_hash(grouped_split(g, seed=0)) != split_hash(grouped_split(g, seed=1))


def test_leakage_detector_flags_overlap():
    g = ["a", "a", "b", "b"]
    leaky = {
        "train": np.array([0, 2]),
        "val": np.array([], dtype=int),
        "test": np.array([1, 3]),  # 'a' and 'b' now straddle train/test
    }
    with pytest.raises(ValueError):
        assert_no_group_leakage(g, leaky)
