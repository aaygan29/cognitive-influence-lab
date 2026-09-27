"""Tests for the calibrated white-box detector and the tuned-lens linearity control.

All CPU: the detector consumes precomputed per-item lens scores, so no torch/model is needed.
"""

from __future__ import annotations

import random
import unittest

from whitebox_detector import (
    separation_statistic,
    run_arm,
    run_paired_detector,
)


def _labels(n_in: int, n_benign: int) -> list[bool]:
    return [True] * n_in + [False] * n_benign


class TestSeparationStatistic(unittest.TestCase):
    def test_perfect_separation(self):
        scores = [10, 9, 8, 1, 0, -1]
        labels = _labels(3, 3)
        self.assertEqual(separation_statistic(scores, labels), 1.0)

    def test_no_separation(self):
        scores = [1, 1, 1, 1]
        labels = _labels(2, 2)
        self.assertEqual(separation_statistic(scores, labels), 0.5)

    def test_reversed_separation(self):
        scores = [0, 0, 0, 9, 9, 9]  # in-condition scores LOWER
        labels = _labels(3, 3)
        self.assertEqual(separation_statistic(scores, labels), 0.0)

    def test_empty_group(self):
        self.assertEqual(separation_statistic([1, 2, 3], [False, False, False]), 0.5)


class TestArm(unittest.TestCase):
    def test_strong_signal_calls(self):
        # in-condition clearly higher on the loyal concept -> detection
        scores = [5, 6, 7, 8, 0, 0.2, -0.1, 0.1]
        labels = _labels(4, 4)
        rep = run_arm("jlens", scores, labels, alpha=0.05, n_perms=500)
        self.assertEqual(rep.calibration.decision, "call")
        self.assertGreater(rep.statistic, 0.9)

    def test_null_signal_no_call(self):
        rng = random.Random(1)
        scores = [rng.gauss(0, 1) for _ in range(16)]
        labels = _labels(8, 8)
        rng.shuffle(labels)
        rep = run_arm("jlens", scores, labels, alpha=0.05, n_perms=1000)
        self.assertEqual(rep.calibration.decision, "no-call")

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            run_arm("jlens", [1, 2, 3], [True, False], n_perms=10)


class TestLinearityControl(unittest.TestCase):
    def test_survives_control(self):
        # J-Lens separates strongly; the tuned lens does not (shuffled / flat).
        jlens = [5, 6, 7, 8, 0, 0.1, -0.2, 0.15]
        tuned = [0.1, -0.1, 0.2, 0.0, 0.05, -0.05, 0.1, -0.1]
        labels = _labels(4, 4)
        out = run_paired_detector(jlens, tuned, labels, alpha=0.05, n_perms=1000)
        self.assertEqual(out["jlens"]["cal_decision"], "call")
        self.assertTrue(out["linearity"]["survives_control"], out["linearity"])
        self.assertIn("SURVIVES", out["linearity"]["verdict"])

    def test_linearity_artifact(self):
        # Both lenses separate equally well: the detection is a linearity artifact.
        shared = [5, 6, 7, 8, 0, 0.1, -0.2, 0.15]
        labels = _labels(4, 4)
        out = run_paired_detector(shared, list(shared), labels, alpha=0.05, n_perms=1000)
        self.assertEqual(out["jlens"]["cal_decision"], "call")
        self.assertFalse(out["linearity"]["survives_control"], out["linearity"])
        self.assertIn("ARTIFACT", out["linearity"]["verdict"])

    def test_no_jlens_detection(self):
        rng = random.Random(2)
        jlens = [rng.gauss(0, 1) for _ in range(16)]
        tuned = [rng.gauss(0, 1) for _ in range(16)]
        labels = _labels(8, 8)
        rng.shuffle(labels)
        out = run_paired_detector(jlens, tuned, labels, alpha=0.05, n_perms=800)
        self.assertEqual(out["jlens"]["cal_decision"], "no-call")
        self.assertFalse(out["linearity"]["survives_control"])
        self.assertIn("no J-Lens detection", out["linearity"]["verdict"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
