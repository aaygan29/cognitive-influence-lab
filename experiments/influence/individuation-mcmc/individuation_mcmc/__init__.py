"""Hierarchical Bayesian inference of per-person decision parameters and
learning rules, via the drift-diffusion reduction of the value circuit.

The posterior over a subject's state vector is an individuation tool (predicts
that person's choices with calibrated uncertainty), the posterior over their
learning rule is a model of how they learn, and both feed the steering channel
so a fitted subject has a person-specific capture dose.
"""

from .ddm import simulate_subject, simulate_ddm, wfpt_logpdf  # noqa: F401
from .model import hierarchical_model, pack_data  # noqa: F401
