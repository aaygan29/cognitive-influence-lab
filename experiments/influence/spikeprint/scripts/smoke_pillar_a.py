#!/usr/bin/env python3
"""SYNTHETIC smoke test of the Pillar-A pipeline — NOT a scientific result.

Generates fake data where a latent "CSI" score drives choices by construction, then runs the
exact analysis the real study will run (AUC + cluster-bootstrap CI + permutation p, decided
under the registered Finding contract). Purpose: prove the machinery produces a correctly-formed,
validated Finding end-to-end *before any real data is touched*. The numbers say nothing about the
world — they only show the pipeline works.
"""
from __future__ import annotations

import numpy as np

from spikeprint.analysis import incremental_validity, predictive_validity


def main() -> None:
    rng = np.random.default_rng(0)
    n, n_problems = 600, 120
    groups = rng.integers(0, n_problems, n)             # repeated "problems" = leakage-safe unit
    csi = rng.normal(size=n)                            # pretend CSI score on each stimulus
    choice = (rng.random(n) < 1 / (1 + np.exp(-1.3 * csi))).astype(int)  # pretend accept/reject
    baseline = rng.normal(size=n)                       # an uninformative text baseline

    print("=== SYNTHETIC SMOKE — NOT a scientific result (pipeline check only) ===\n")
    h1 = predictive_validity(
        "H1 predictive (synthetic)", "SYNTHETIC", csi, choice, groups=groups, n_boot=2000, n_perm=2000
    )
    print(h1.render())
    h2 = incremental_validity(
        "H2 incremental (synthetic)", "SYNTHETIC", csi, baseline, choice, groups=groups, n_boot=2000
    )
    print(h2.render())
    print("\nInterpretation: the pipeline emits well-formed, decided Findings. On REAL NARPS data")
    print("these same calls produce the actual value-framing (H1a) result — pass or fail per the")
    print("kill criteria. (Persuasive-text H1b is a separate construct on CMV / Persuasion-for-Good.)")


if __name__ == "__main__":
    main()
