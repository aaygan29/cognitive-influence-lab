"""Latent belief fusion (multitrait-multimethod) and the H-load validity test.

Belief is a latent factor estimated from several validated indicators (behavior, self-report,
certainty, a discernment-style measure). The load-bearing validity test for the neural signature is
whether B(s) loads on that shared latent factor. Loading validates B(s); loading on nothing falsifies
it. This is the rigorous form of combining disparate evidence into one assessment.

No em dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy import stats


@dataclass
class Finding:
    """Every reported number carries its evidence. Mirrors the program's validation contract."""
    name: str
    value: float
    ci95: tuple
    n: int
    passed: bool
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        d["ci95"] = [float(self.ci95[0]), float(self.ci95[1])]
        d["value"] = float(self.value)
        return d


def _standardize_cols(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def estimate_latent_factor(indicators: dict) -> np.ndarray:
    """One-factor model over the standardized indicator columns; returns latent factor scores.

    indicators: dict of name -> 1D array (behavior, self_report, certainty, mist).
    The behavior indicator is binary; standardizing it is acceptable for a single common-factor score.
    """
    names = list(indicators.keys())
    M = np.column_stack([np.asarray(indicators[k], dtype=float) for k in names])
    Mz = _standardize_cols(M)
    fa = FactorAnalysis(n_components=1, random_state=0)
    scores = fa.fit_transform(Mz)[:, 0]
    # orient the factor so it correlates positively with behavior (interpretability)
    if "behavior" in indicators:
        if np.corrcoef(scores, np.asarray(indicators["behavior"], dtype=float))[0, 1] < 0:
            scores = -scores
    return scores


def _boot_corr_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(a)
    r0 = float(np.corrcoef(a, b)[0, 1])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.corrcoef(a[idx], b[idx])[0, 1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r0, (lo, hi)


def h_load(B: np.ndarray, latent_scores: np.ndarray, min_r: float = 0.15, seed: int = 0) -> Finding:
    """Does B(s) load on the latent belief factor?

    Kill criterion: pass requires the bootstrap 95% CI of corr(B, latent) to exclude 0 AND the point
    correlation to exceed min_r (a non-trivial loading, not just significance).
    """
    B = np.asarray(B, dtype=float)
    r, (lo, hi) = _boot_corr_ci(B, latent_scores, seed=seed)
    passed = bool((lo > 0 and hi > 0 and r >= min_r) or (lo < 0 and hi < 0 and -r >= min_r))
    return Finding(
        name="H_load_corr_B_latent",
        value=r,
        ci95=(lo, hi),
        n=len(B),
        passed=passed,
        note=f"min_r={min_r}; loads on shared belief factor" if passed else f"min_r={min_r}; does not load",
    )
