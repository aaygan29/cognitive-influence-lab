#!/usr/bin/env python3
"""Minimum detectable effect size (MDES) for a correlation, via the Fisher-z approximation.

For a two-sided test at level ``alpha`` and target ``power``, the smallest |rho| detectable at
sample size ``n`` is r = tanh((z_{1-alpha/2} + z_power) / sqrt(n - 3)). Pure stdlib; deterministic.
Used to record a numeric MDES in PREREGISTRATION.md (sec. 3) before OSF registration.
"""
from __future__ import annotations

import math
from statistics import NormalDist


def mdes_correlation(
    n: int, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True
) -> float:
    if n <= 4:
        raise ValueError("n must be > 4")
    za = NormalDist().inv_cdf(1 - alpha / 2 if two_sided else 1 - alpha)
    zb = NormalDist().inv_cdf(power)
    return math.tanh((za + zb) / math.sqrt(n - 3))


if __name__ == "__main__":
    for n in (250, 1000, 5000, 13000):
        print(f"n={n:>6}  MDES |rho| (alpha=.05, power=.80, two-sided) = {mdes_correlation(n):.4f}")
