"""Pillar A analysis: does a score predict a binary choice above chance, and above a baseline?

Pure NumPy. The core estimand is the AUC (= P(score ranks a "1" above a "0"); equivalent to the
Mann-Whitney statistic), with a cluster (group) bootstrap CI and a label-permutation p-value.
These feed the registered :class:`~spikeprint.validate.Finding` contract.

This module computes statistics only; it does NOT fetch data and makes no scientific claim on
its own. Confirmatory runs add the registered mixed-effects model (statsmodels) on top; the
AUC + cluster bootstrap here is the leakage-safe, assumption-light core (see PREREGISTRATION.md
sec. 3, 6). Degenerate (single-class) resamples are skipped AND counted; the skip fraction is
surfaced on every Finding and flagged above the registered threshold (1%).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .validate import Finding

SKIP_FLAG_THRESHOLD = 0.01  # registered: flag a Finding if >1% of resamples were degenerate


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared — like scipy.stats.rankdata, stdlib-only."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size, dtype=float)
    i = 0
    n = a.size
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the Mann-Whitney U statistic (tie-aware). 0.5 = chance."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc requires both classes present")
    r = _rankdata(s)
    u = r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _group_codes(groups: Optional[Sequence], n: int) -> Tuple[np.ndarray, int]:
    if groups is None:
        return np.arange(n), n
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return inv, int(inv.max()) + 1


def cluster_bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Optional[Sequence] = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    return_skips: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """Percentile CI for AUC, resampling whole groups with replacement (cluster bootstrap).

    Resampling at the group level (not the row level) respects within-group correlation — the
    same discipline as the leakage-safe splits. Degenerate (single-class) resamples are skipped
    and counted; with ``return_skips`` the skip fraction is also returned.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    codes, n_groups = _group_codes(groups, s.size)
    rows = [np.nonzero(codes == k)[0] for k in range(n_groups)]
    rng = np.random.default_rng(seed)
    boots = []
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([rows[k] for k in chosen])
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        boots.append(auc(s[idx], yy))
    if not boots:
        raise ValueError("all bootstrap resamples were degenerate (single-class)")
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    if return_skips:
        return float(lo), float(hi), skipped / n_boot
    return float(lo), float(hi)


def permutation_pvalue(
    scores: Sequence[float],
    labels: Sequence[int],
    observed: Optional[float] = None,
    n_perm: int = 10_000,
    seed: int = 0,
    return_skips: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Two-sided permutation p-value for AUC != 0.5 (shuffle labels). Deterministic given seed.

    Uses the Phipson & Smyth (2010) estimator over the permutations that were actually evaluated;
    degenerate (single-class) permutations are skipped and counted.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    obs = auc(s, y) if observed is None else observed
    eff = abs(obs - 0.5)
    rng = np.random.default_rng(seed)
    count = 0
    used = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        if perm.min() == perm.max():
            continue
        used += 1
        if abs(auc(s, perm) - 0.5) >= eff:
            count += 1
    p = (count + 1) / (used + 1)
    if return_skips:
        return p, (n_perm - used) / n_perm
    return p


def _skip_note(*skips: float) -> str:
    hi = max(skips)
    flag = "HIGH_SKIP(>1%) — CI/p may be biased; " if hi > SKIP_FLAG_THRESHOLD else ""
    return f"{flag}resample_skip_frac={hi:.4f}"


def predictive_validity(
    name: str,
    dataset: str,
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Optional[Sequence] = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Finding:
    """H1: does ``scores`` predict the binary ``labels`` above chance? Returns a decided Finding.

    Metric is AUC with null = 0.5 (carried on the Finding so the family rule decides it correctly).
    """
    a = auc(scores, labels)
    lo, hi, bs_skip = cluster_bootstrap_ci(
        scores, labels, groups, n_boot, alpha, seed, return_skips=True
    )
    p, perm_skip = permutation_pvalue(scores, labels, a, n_perm, seed, return_skips=True)
    f = Finding(
        name=name,
        value=a,
        dataset=dataset,
        metric="AUC",
        baseline=0.5,
        effect_size=a - 0.5,
        ci95=(lo, hi),
        n=int(np.asarray(labels).size),
        p_value=p,
        null=0.5,
        notes=_skip_note(bs_skip, perm_skip),
    )
    return f.decide()


def incremental_validity(
    name: str,
    dataset: str,
    csi_scores: Sequence[float],
    baseline_scores: Sequence[float],
    labels: Sequence[int],
    groups: Optional[Sequence] = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Finding:
    """H2: does CSI beat a baseline score? Effect = AUC(CSI) - AUC(baseline), paired bootstrap."""
    csi = np.asarray(csi_scores, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    y = np.asarray(labels)
    d = auc(csi, y) - auc(base, y)
    codes, n_groups = _group_codes(groups, csi.size)
    rows = [np.nonzero(codes == k)[0] for k in range(n_groups)]
    rng = np.random.default_rng(seed)
    boots = []
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([rows[k] for k in chosen])
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        boots.append(auc(csi[idx], yy) - auc(base[idx], yy))
    if not boots:
        raise ValueError("all bootstrap resamples were degenerate (single-class)")
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(min(max(p, 1.0 / (boots.size + 1)), 1.0))
    f = Finding(
        name=name,
        value=d,
        dataset=dataset,
        metric="dAUC(CSI-baseline)",
        baseline=0.0,
        effect_size=d,
        ci95=(float(lo), float(hi)),
        n=int(y.size),
        p_value=p,
        null=0.0,
        notes=_skip_note(skipped / n_boot) + "; H2 is one-sided (pass = lower 95% CI > 0)",
    )
    # H2 is directional: CSI is incrementally valid only if it STRICTLY beats the baseline.
    return replace(f, passed=bool(f.ci95[0] > 0.0))
