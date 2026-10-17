"""Honest evaluation: ROC/AUC, bootstrap CIs, lead time, dose-response.

Pure numpy, no learned classifier. AUC here is simply the probability that a
randomly chosen manipulated trial scores higher than a randomly chosen honest
trial (the Mann-Whitney interpretation), which is exactly what a threshold-free
detector should be judged on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score(pos) > score(neg)), ties counted as half."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-based computation, O(n log n).
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # Average ranks for ties.
    _assign_tie_ranks(allv, ranks)
    r_pos = ranks[: pos.size].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    i = 0
    n = len(sv)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def bootstrap_auc_ci(
    pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Return (auc, lo, hi) with a percentile bootstrap CI."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    rng = np.random.default_rng(seed)
    point = roc_auc(pos, neg)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pb = pos[rng.integers(0, pos.size, pos.size)]
        nb = neg[rng.integers(0, neg.size, neg.size)]
        boots[b] = roc_auc(pb, nb)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def permutation_test_auc(
    pos: np.ndarray, neg: np.ndarray, n_perm: int = 2000, seed: int | None = 0
) -> tuple[float, float]:
    """Test AUC against chance by permuting labels. Returns (auc, p_one_sided).

    Under the null the two samples come from the same distribution, so shuffling
    the pooled scores and re-splitting gives the null AUC distribution. The
    one-sided p-value is the fraction of permuted AUCs at least as large as the
    observed one (add-one smoothing so p is never exactly zero).
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    obs = roc_auc(pos, neg)
    pooled = np.concatenate([pos, neg])
    m = pos.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if roc_auc(pooled[:m], pooled[m:]) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, float(p)


def delong_paired_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> dict:
    """Paired DeLong test comparing two detectors' AUCs on the same trials.

    ``labels`` is 1 for the positive class (manipulated) and 0 for the negative.
    Returns both AUCs, their difference, a two-sided z and p. This is the correct
    test for two scores evaluated on one shared test set, where the AUCs are
    correlated and an unpaired comparison would be wrong.
    """
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    neg = labels == 0
    m, n = int(pos.sum()), int(neg.sum())

    def components(scores):
        x = np.asarray(scores, float)[pos]   # positives
        y = np.asarray(scores, float)[neg]   # negatives
        # Placement values via midrank comparison (ties = 0.5).
        gt = (x[:, None] > y[None, :]).astype(float) + 0.5 * (x[:, None] == y[None, :])
        v10 = gt.mean(axis=1)                 # length m
        v01 = gt.mean(axis=0)                 # length n
        auc = gt.mean()
        return auc, v10, v01

    auc_a, v10a, v01a = components(scores_a)
    auc_b, v10b, v01b = components(scores_b)

    s10 = np.cov(np.stack([v10a, v10b]), bias=False)
    s01 = np.cov(np.stack([v01a, v01b]), bias=False)
    cov = s10 / m + s01 / n                   # 2x2 covariance of (auc_a, auc_b)

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        z, p = 0.0, 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "diff": float(auc_a - auc_b), "z": float(z), "p": float(p),
            "n_pos": m, "n_neg": n}


def stratified_auc(
    scores: np.ndarray, confound: np.ndarray, labels: np.ndarray, n_bins: int = 5
) -> dict:
    """AUC of ``scores`` within strata of a ``confound``, to remove its influence.

    If a detector only works because it tracks a nuisance variable (here,
    commitment speed), then within narrow bins of that variable it should fall to
    chance. Computing AUC within confound bins and averaging (weighted by the
    number of discordant pairs available in each bin) tests whether the detector
    carries information beyond the confound. Returns per-bin AUCs and a
    pair-weighted pooled AUC.
    """
    scores = np.asarray(scores, float)
    confound = np.asarray(confound, float)
    labels = np.asarray(labels).astype(int)
    edges = np.quantile(confound, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    per_bin, num, den = [], 0.0, 0.0
    for i in range(n_bins):
        m = (confound >= edges[i]) & (confound < edges[i + 1])
        pos = scores[m & (labels == 1)]
        neg = scores[m & (labels == 0)]
        if pos.size and neg.size:
            a = roc_auc(pos, neg)
            w = pos.size * neg.size
            per_bin.append({"bin": i, "auc": a, "n_pos": int(pos.size),
                            "n_neg": int(neg.size)})
            num += a * w
            den += w
    pooled = num / den if den > 0 else float("nan")
    return {"pooled_stratified_auc": float(pooled), "per_bin": per_bin}


@dataclass
class LeadTimeSummary:
    detection_rate: float      # fraction of manipulated trials that alarmed
    median_lead_s: float       # median (commit - alarm), alarmed trials only
    iqr_lead_s: tuple[float, float]
    pre_commit_rate: float     # fraction alarmed strictly before commitment


def summarize_lead_times(alarms: list[dict]) -> LeadTimeSummary:
    alarmed = [a for a in alarms if a["alarmed"]]
    det_rate = len(alarmed) / len(alarms) if alarms else float("nan")
    leads = np.array([a["lead_time"] for a in alarmed if np.isfinite(a["lead_time"])])
    if leads.size:
        med = float(np.median(leads))
        q1, q3 = np.quantile(leads, [0.25, 0.75])
        pre = float(np.mean(leads > 0))
    else:
        med, q1, q3, pre = float("nan"), float("nan"), float("nan"), float("nan")
    return LeadTimeSummary(det_rate, med, (float(q1), float(q3)), pre)


def stealth_floor(
    doses: np.ndarray, aucs: np.ndarray, threshold: float = 0.75
) -> float:
    """Smallest dose at which AUC first reaches ``threshold`` (linear interp).

    Below this dose the manipulation is, by this detector, statistically
    indistinguishable from an honest hard decision. Returns ``nan`` if the
    threshold is never crossed within the sampled dose range.
    """
    doses = np.asarray(doses, float)
    aucs = np.asarray(aucs, float)
    order = np.argsort(doses)
    doses, aucs = doses[order], aucs[order]
    for i in range(1, len(doses)):
        if aucs[i - 1] < threshold <= aucs[i]:
            frac = (threshold - aucs[i - 1]) / (aucs[i] - aucs[i - 1])
            return float(doses[i - 1] + frac * (doses[i] - doses[i - 1]))
    if len(aucs) and aucs[-1] >= threshold and aucs[0] >= threshold:
        return float(doses[0])
    return float("nan")
