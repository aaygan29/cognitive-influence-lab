"""Validation contract for spikeprint.

Every reported quantity is a :class:`Finding`: a value bound to the public dataset it was
measured on, a baseline, an effect size, a 95% CI, n, an optional p-value, and a pass/fail
decision. A Finding that has not cleared its test is ``passed=False`` and must be rendered
``UNVALIDATED``. No claim is emitted as a bare scalar.

Two decision levels, matching ``PREREGISTRATION.md`` (sections 3 and 6):

* :meth:`Finding.decide` — single-hypothesis primitive: passes iff the 95% CI excludes the null
  (requires a CI and an n).
* :func:`decide_family` — the **registered** family-level rule: a Finding passes iff its CI
  excludes the null AND it survives Benjamini-Hochberg FDR across the whole H1-H4 family
  (requires a p-value on every Finding). This is the rule CI / report generation must use so
  that "validated" means exactly what was preregistered.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class UnvalidatedClaimError(RuntimeError):
    """Raised when an unvalidated Finding is presented as a result."""


@dataclass(frozen=True)
class Finding:
    """A single measured quantity bound to its evidence.

    A Finding is the only currency spikeprint reports. It cannot become validated without a
    confidence interval and a sample size, so silence/ambiguity defaults to UNVALIDATED rather
    than to a claim.
    """

    name: str
    value: float
    dataset: str
    metric: str = "effect"
    baseline: Optional[float] = None
    effect_size: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    p_value: Optional[float] = None
    null: float = 0.0  # the chance value for this metric (e.g. 0.0 for a difference, 0.5 for AUC)
    passed: bool = False
    notes: str = ""

    def decide(self, null: Optional[float] = None) -> "Finding":
        """Single-hypothesis primitive: pass iff the 95% CI excludes the metric's null.

        Uses this Finding's own ``null`` (e.g. 0.5 for AUC, 0.0 for a difference) unless an
        override is passed. Requires both a CI and an n. NOTE: this does NOT apply the
        family-wise FDR correction; for confirmatory reporting use :func:`decide_family`.
        """
        nv = self.null if null is None else null
        ok = (
            self.ci95 is not None
            and self.n is not None
            and (self.ci95[0] > nv or self.ci95[1] < nv)
        )
        return replace(self, passed=bool(ok))

    def render(self) -> str:
        tag = "VALIDATED" if self.passed else "UNVALIDATED"
        ci = f"[{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]" if self.ci95 is not None else "[n/a]"
        base = "" if self.baseline is None else f" baseline={self.baseline:.3f}"
        p = "" if self.p_value is None else f" p={self.p_value:.4g}"
        return (
            f"[{tag}] {self.name}: {self.value:.3f} ({self.metric}) on {self.dataset}"
            f" 95%CI={ci} n={self.n}{p}{base}"
        )


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR. Return a boolean array (original order) of rejected hypotheses.

    Controls the false discovery rate at ``alpha`` (Benjamini & Hochberg, 1995).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = p[order] <= thresh
    rejected = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.max(np.nonzero(below)[0]))
        sorted_reject = np.zeros(m, dtype=bool)
        sorted_reject[: kmax + 1] = True
        rejected[order] = sorted_reject
    return rejected


def decide_family(
    findings: Sequence[Finding], alpha: float = 0.05, null: Optional[float] = None
) -> List[Finding]:
    """Registered family-level decision.

    A Finding passes iff (a) its 95% CI excludes its metric's null AND (b) it is rejected under
    Benjamini-Hochberg FDR across the whole family at ``alpha``. Each Finding's own ``null`` is
    used (e.g. 0.5 for AUC, 0.0 for a difference), so metrics with different nulls can share a
    family; pass ``null`` only to override all of them. Requires a p-value on every Finding.
    """
    items = list(findings)
    if not items:
        return []
    if any(f.p_value is None for f in items):
        raise ValueError(
            "decide_family requires a p_value on every Finding (registered BH-FDR rule)."
        )
    rejected = benjamini_hochberg([f.p_value for f in items], alpha=alpha)
    out: List[Finding] = []
    for f, rej in zip(items, rejected):
        nv = f.null if null is None else null
        ci_ok = (
            f.ci95 is not None and f.n is not None and (f.ci95[0] > nv or f.ci95[1] < nv)
        )
        out.append(replace(f, passed=bool(rej and ci_ok)))
    return out


def gate(findings: Sequence[Finding]) -> None:
    """Raise if any Finding is unvalidated. Use in CI / report generation to block overclaiming."""
    unvalidated = [f.name for f in findings if not f.passed]
    if unvalidated:
        raise UnvalidatedClaimError(f"unvalidated findings present: {unvalidated}")


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap (1-alpha) CI for ``statistic``. Deterministic given ``seed``.

    Percentile method by default; BCa is a planned option for skewed effect-size sampling
    distributions (see PREREGISTRATION.md sec. 6). Reported CIs state the method used.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("data must be a non-empty 1-D sequence")
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = statistic(arr[rng.integers(0, n, n)])
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
