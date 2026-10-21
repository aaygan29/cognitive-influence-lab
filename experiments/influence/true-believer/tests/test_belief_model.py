"""Tests for the formal belief-dynamics model: the four falsifiable predictions must hold.

No em dashes.
"""

import numpy as np

from src.belief_model import (
    beta_crit, equilibria, stiffness,
    pred_hysteresis, pred_sequential_vs_single, pred_robustness_scaling, susceptibility_curve,
)


def test_bistability_above_threshold():
    # alpha > 0 gives two stable wells at beta = 0; alpha <= 0 gives one
    assert len(equilibria(1.0, 0.0)) == 2
    assert len(equilibria(-0.5, 0.0)) == 1


def test_hysteresis_loop_has_area():
    bu, up, bd, down = pred_hysteresis()
    area = abs(np.trapezoid(up, bu) - np.trapezoid(down[::-1], bd[::-1]))
    assert area > 0.5, f"hysteresis loop should have nonzero area, got {area}"


def test_sequential_beats_single():
    seq, single = pred_sequential_vs_single()
    assert seq > single + 0.5, f"sequential ({seq}) should beat single large step ({single})"
    assert seq > 0, "sequential should cross the barrier toward the target"


def test_robustness_scales_three_halves():
    a, bc, p = pred_robustness_scaling()
    assert abs(p - 1.5) < 0.05, f"beta_crit exponent should be ~1.5, got {p}"


def test_susceptibility_diverges_at_fold():
    sc = susceptibility_curve(alpha=1.0)
    assert sc[-1, 1] > 5 * sc[0, 1], "susceptibility must diverge as beta approaches the fold"


def test_beta_crit_matches_fold_set():
    # at beta_crit the lower well and the unstable root merge (double root): stiffness -> 0
    alpha = 1.0
    bc = beta_crit(alpha)
    eqs = equilibria(alpha, bc * 0.999)
    b_low = eqs[0]
    assert stiffness(b_low, alpha) < 0.1, "stiffness should approach zero at the fold"
