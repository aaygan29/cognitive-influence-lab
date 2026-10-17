"""Commitment geometry of the population code.

An independent corroborating channel. In the reduced circuit, before a choice
is resolved the two options are represented along nearly separate directions;
once the choice resolves, the chosen option's representation expands and the
population settles onto a common action-aligned axis. This module measures both
effects from population activity so that a run of the simulator can confirm the
substrate behaves as intended, and so the monitor can use commitment timing as
a feature.

The two reported quantities:

  * ``subspace_angle`` - principal angle (degrees) between the pop-1-chosen and
    pop-2-chosen population directions. Large before commitment.
  * ``expansion_ratio`` - variance of the chosen-option population states over
    variance of the unchosen-option states, late in the trial. Greater than one
    when the chosen representation expands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import Trajectory
from .population import PopulationReadout


@dataclass
class CommitmentGeometry:
    subspace_angle_deg: float
    expansion_ratio: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def commitment_geometry(
    traj: Trajectory,
    readout: PopulationReadout,
    *,
    pre_frac: float = 0.4,
    post_frac: float = 0.85,
    rate_bound: float = 15.0,
) -> CommitmentGeometry:
    """Measure option-axis separation and chosen-subspace expansion.

    ``pre_frac`` and ``post_frac`` locate the pre- and post-commitment time
    points as fractions of the trial. Trials are grouped by their realized
    choice; the analysis needs both choices to be present.
    """
    rates = readout.project(traj)                # (T, n_trials, n_units)
    T = rates.shape[0]
    pre = int(pre_frac * T)
    post = int(post_frac * T)

    choices = traj.choices(rate_bound)
    g1 = choices == 1
    g2 = choices == -1
    if g1.sum() < 3 or g2.sum() < 3:
        return CommitmentGeometry(float("nan"), float("nan"))

    # Population direction that separates choices, pre vs post commitment.
    pre_axis = _unit(rates[pre, g1].mean(0) - rates[pre, g2].mean(0))
    post_axis = _unit(rates[post, g1].mean(0) - rates[post, g2].mean(0))
    cos = float(np.clip(abs(np.dot(pre_axis, post_axis)), 0.0, 1.0))
    angle = np.degrees(np.arccos(cos))

    # Chosen vs unchosen representational variance, late in the trial.
    chosen_var = rates[post, g1].var(0).sum() + rates[post, g2].var(0).sum()
    # Unchosen: swap the grouping to read the losing representation's spread.
    unchosen_var = rates[pre, g1].var(0).sum() + rates[pre, g2].var(0).sum()
    expansion = float(chosen_var / unchosen_var) if unchosen_var > 0 else float("nan")

    return CommitmentGeometry(angle, expansion)
