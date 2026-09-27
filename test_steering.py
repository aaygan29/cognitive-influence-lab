"""Tests for the steering arm's pure core: effect statistic, calibration, paired verdict.

The GPU path (hooks, forward passes) is not exercised here; the effect statistic, the
random-direction calibration, and the causal-vs-artifact verdict are all pure and are what the
scientific claim rests on.
"""

from __future__ import annotations

import random
import unittest

from steering import (
    steering_effect,
    calibrate_effect,
    paired_steering_verdict,
)


class TestEffect(unittest.TestCase):
    def test_positive_shift(self):
        base = [0.0, 0.0, 0.0]
        steered = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(steering_effect(base, steered), 2.0)

    def test_no_shift(self):
        self.assertEqual(steering_effect([1, 2, 3], [1, 2, 3]), 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            steering_effect([1, 2], [1])


class TestPairedVerdict(unittest.TestCase):
    def _null(self):
        rng = random.Random(7)
        return [rng.gauss(0, 1) for _ in range(300)]

    def test_causal_handle(self):
        null = self._null()
        jlens_cal = calibrate_effect(6.0, null)   # far in the upper tail -> call
        tuned_cal = calibrate_effect(0.1, null)   # near the null -> no-call
        v = paired_steering_verdict(6.0, 0.1, jlens_cal, tuned_cal)
        self.assertTrue(v.causal_handle, v)
        self.assertIn("CAUSAL HANDLE", v.verdict)

    def test_linearity_artifact(self):
        null = self._null()
        # both directions move behaviour, tuned as much as J-Lens
        jlens_cal = calibrate_effect(6.0, null)
        tuned_cal = calibrate_effect(6.5, null)
        v = paired_steering_verdict(6.0, 6.5, jlens_cal, tuned_cal)
        self.assertFalse(v.causal_handle, v)
        self.assertIn("ARTIFACT", v.verdict)

    def test_no_causal_effect(self):
        null = self._null()
        jlens_cal = calibrate_effect(0.2, null)   # inside the null
        tuned_cal = calibrate_effect(0.1, null)
        v = paired_steering_verdict(0.2, 0.1, jlens_cal, tuned_cal)
        self.assertFalse(v.causal_handle, v)
        self.assertIn("no causal effect", v.verdict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
