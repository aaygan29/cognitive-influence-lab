"""Deterministic, group-keyed train/val/test splits with a frozen split hash.

Splits are assigned by GROUP id (e.g., a choices13k problem id), so the same group can never
appear in two partitions. This prevents the outcome leakage a naive row split causes on
aggregate data, where one gamble/problem recurs across many rows (see PREREGISTRATION.md
sec. 5). The split is hashed; analyses verify the hash before any outcome is loaded.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Sequence, Tuple

import numpy as np


def grouped_split(
    group_ids: Sequence,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Assign each row to train/val/test by its group id. Same group id -> same partition.

    Deterministic given ``seed``: unique group ids are sorted, then permuted with a seeded RNG,
    then sliced by ``fracs``. Returns a dict ``{"train"|"val"|"test": index_array}``.
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1.0")
    gid = [str(g) for g in group_ids]
    uniq = np.array(sorted(set(gid)))
    rng = np.random.default_rng(seed)
    uniq = uniq[rng.permutation(uniq.size)]
    n = uniq.size
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    assign = {}
    for i, g in enumerate(uniq):
        assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    idx: Dict[str, list] = {"train": [], "val": [], "test": []}
    for j, g in enumerate(gid):
        idx[assign[g]].append(j)
    return {k: np.asarray(v, dtype=int) for k, v in idx.items()}


def split_hash(splits: Dict[str, np.ndarray]) -> str:
    """Stable SHA-256 of a split (committed before outcomes are loaded; verified at analysis)."""
    payload = {k: sorted(int(i) for i in v) for k, v in splits.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_no_group_leakage(group_ids: Sequence, splits: Dict[str, np.ndarray]) -> None:
    """Raise if any group id appears in more than one partition."""
    g = np.array([str(x) for x in group_ids])
    sets = {k: set(g[v].tolist()) for k, v in splits.items()}
    overlap = (
        (sets["train"] & sets["test"])
        | (sets["train"] & sets["val"])
        | (sets["val"] & sets["test"])
    )
    if overlap:
        raise ValueError(f"group leakage across splits (e.g. {sorted(overlap)[:5]})")
