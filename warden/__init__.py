"""
warden -- a calibrated cognitive-security reasoning instrument (v0.1 skeleton).

WARDEN reasons over STATED ASSUMPTIONS (a Theta), it does not measure a mind.
Three heads, each calibrated and abstaining, built on the deceptkit discipline:

    H1 Assay    -- assumption-relative manipulability SENSITIVITY (not a bound)
    H2 Detector -- live-engagement detection with mandatory abstention + simulator-PPV
    H3 Floor    -- adaptive minimax countermeasure ORDERING (in-silico)

This is a reference skeleton. It reuses deceptkit's fusion layer (SPRT policy,
isotonic calibration, StratifiedGroupKFold, abstention gates) for H2, and applies
the same honesty discipline to H1 and H3. See WARDEN_proposal_v2.md for the design
and revision_log_warden.md for how the red team hardened it.
"""
from .assumptions import Theta, enforce_scope, ScopeRefusal, ALL_CHANNELS
from .simulate_influence import simulate_influence, simulate_manipulability
from .heads import h1_assay, h2_detect, h3_floor, h1_state_sweep

__all__ = ["Theta", "enforce_scope", "ScopeRefusal", "ALL_CHANNELS",
           "simulate_influence", "simulate_manipulability",
           "h1_assay", "h2_detect", "h3_floor", "h1_state_sweep"]
__version__ = "0.2.0"
