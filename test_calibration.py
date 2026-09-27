"""Tests for the level-fixed calibration layer: finite-sample validity and the no-call branch."""

from __future__ import annotations

import random
import unittest

from calibration import CalibratedDetector


class TestCalibration(unittest.TestCase):
    def test_rank_pvalue_monotone(self):
        det = CalibratedDetector(alpha=0.05).fit([float(i) for i in range(100)])
        self.assertLessEqual(det.p_value(1000.0), det.p_value(50.0))
        self.assertLessEqual(det.p_value(50.0), det.p_value(-1.0))

    def test_max_candidate_small_p(self):
        det = CalibratedDetector(alpha=0.05).fit([float(i) for i in range(99)])
        # candidate strictly above all 99 null draws -> p = 1/(99+1) = 0.01
        self.assertAlmostEqual(det.p_value(1000.0), 1.0 / 100.0)

    def test_no_call_branch_below_ceiling(self):
        det = CalibratedDetector(alpha=0.005).fit([float(i) for i in range(10)])  # ceiling 1/11
        r = det.decide(1000.0)
        self.assertEqual(r.decision, "no-call")
        self.assertGreater(det.min_achievable_alpha, det.alpha)

    def test_call_when_reachable(self):
        det = CalibratedDetector(alpha=0.05).fit([float(i) for i in range(99)])
        self.assertEqual(det.decide(1000.0).decision, "call")

    def test_empty_calibration_is_no_call(self):
        det = CalibratedDetector(alpha=0.05).fit([])
        r = det.decide(5.0)
        self.assertEqual(r.p_value, 1.0)
        self.assertEqual(r.decision, "no-call")

    def test_finite_sample_level_holds(self):
        """Under the null (candidate exchangeable with calibration), P(p <= alpha) <= alpha."""
        rng = random.Random(20260930)
        alpha = 0.1
        m, trials = 200, 3000
        false_calls = 0
        for _ in range(trials):
            draws = [rng.gauss(0, 1) for _ in range(m + 1)]
            cand, null = draws[0], draws[1:]
            det = CalibratedDetector(alpha=alpha).fit(null)
            if det.decide(cand).decision == "call":
                false_calls += 1
        rate = false_calls / trials
        # Valid layer: realized false-call rate must not exceed alpha beyond MC noise.
        self.assertLessEqual(rate, alpha + 0.03, f"realized {rate:.3f} > alpha {alpha}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
