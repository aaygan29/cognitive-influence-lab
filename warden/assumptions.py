"""
assumptions.py -- the Theta ledger.

WARDEN is a reasoning instrument over stated assumptions, not a measurement of a
mind (critique C1). Every output is relative to a Theta. This module makes Theta
an explicit, serializable object that travels with every result, so a reader can
always see what the numbers are relative to.

No em dashes anywhere in this package.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Sequence
import json


# The read channels. Neural belief-processing markers index PROCESSING of
# incongruence, explicitly not YIELDING (C10 -- entrainment / ITPC is removed as
# an evidence channel). Behavioral and autonomic channels are the honest floor.
NEURAL_CHANNELS     = ["z_n400", "z_latepos", "z_prederr"]   # N400, late positivity, prediction error
BEHAVIORAL_CHANNELS = ["z_choice", "z_rt"]                    # choice shift, response-time drift
AUTONOMIC_CHANNELS  = ["z_eda", "z_pupil"]                    # EDA, pupil (deceptkit bridge)
ALL_CHANNELS        = NEURAL_CHANNELS + BEHAVIORAL_CHANNELS + AUTONOMIC_CHANNELS

OPERATOR_CLASSES = ["benign_persuasion", "hostile_tradecraft"]


@dataclass
class Theta:
    """The full assumption set that produces a WARDEN output."""
    operator_class: str = "benign_persuasion"        # C12: the class M is calibrated to
    influence_scale: float = 1.0                      # scale of the admissible influence input u
    adversary_observes_output: bool = False           # C6: oracle threat model
    prevalence_prior: float = 0.01                    # H2 base rate (base-rate honesty)
    admissible_u_note: str = "unit-norm perturbations across open channels"
    norm: str = "l2"                                  # C2: aggregation choice, swept for stability
    context: str = "self_directed"                    # C8: gates adverse-context refusal
    channels: Sequence[str] = field(default_factory=lambda: list(ALL_CHANNELS))

    # Asymmetric-power contexts where per-individual H1/H2 output is refused (C8).
    ADVERSE_CONTEXTS = ("employment", "pre_employment", "custody", "immigration",
                        "security_clearance", "adversarial_negotiation", "interrogation")

    def is_adverse(self) -> bool:
        return self.context in self.ADVERSE_CONTEXTS

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ScopeRefusal(Exception):
    """Raised when WARDEN is asked for per-individual output in an adverse context (C8)."""
    pass


def enforce_scope(theta: Theta):
    """C8 enforced at the API boundary: no per-individual H1/H2 output in any
    asymmetric-power relationship. Abstain here is a refusal to run, not a caveat."""
    if theta.is_adverse():
        raise ScopeRefusal(
            f"context '{theta.context}' is an asymmetric-power setting; "
            "per-individual H1/H2 output is refused by construction (C8). "
            "Permitted: self_directed or irb_supervised research only."
        )
