"""Fast smoke tests for the persuasion-control scenario.

These run small batches so they stay quick; they check the shape of the result,
not the headline numbers (those come from the full run).
"""

from __future__ import annotations

import numpy as np

from persuasion_control import scenario as S


def test_approve_rate_increases_with_dose():
    """A larger persuasion dose should not decrease the approve rate."""
    low = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.004, seed=1, n_trials=60)
    high = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.024, seed=1, n_trials=60)
    assert high.approve_rate >= low.approve_rate


def test_zero_dose_is_near_coinflip_on_borderline():
    """With no steering and borderline evidence the decision should be roughly a
    coin flip, not pinned to one option."""
    b = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.0, seed=2, n_trials=120)
    assert 0.2 < b.approve_rate < 0.8


def test_realized_dose_monotone_in_amplitude():
    lo = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.004, seed=3, n_trials=10)
    hi = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.016, seed=3, n_trials=10)
    assert hi.realized_dose_na_s > lo.realized_dose_na_s


def test_choice_logic_matches_final_state():
    b = S.simulate_decision(S.BORDERLINE_COHERENCE, 0.012, seed=4, n_trials=16)
    z = b.traj.S1 - b.traj.S2
    expected = float(np.mean(z[-1, :] > 0.0))
    assert abs(b.approve_rate - expected) < 1e-12
