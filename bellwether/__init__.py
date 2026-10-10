"""bellwether: an early-warning instrument for value-decision capture.

The package provides a mechanistic, fully in-silico testbed in which a
value-based decision is produced by a biophysically grounded attractor
circuit, an adversary can covertly steer that decision by exploiting the
circuit's sensitivity near its choice bifurcation, and a monitor attempts to
raise an alarm *before* the behavioral commitment.

Four pieces fit together:

1. ``dynamics``   - the reduced two-variable decision circuit (the substrate).
2. ``population`` - a random high-dimensional neural readout of that circuit.
3. ``ews`` / ``estimator`` / ``geometry`` - the observables a monitor can read
   from population activity: critical-slowing-down statistics, the recovered
   drift field of the decision variable, and the commitment geometry.
4. ``monitor`` / ``detect`` - the online alarm and its honest evaluation
   (ROC/AUC with bootstrap CIs, detection lead time, dose-response, and the
   stealth floor below which steering is statistically invisible).

Nothing here is fit to real neural data. The value of the artifact is that
every claim it makes is a falsifiable property of a transparent generative
model, so the same monitor can later be pointed at real recordings.
"""

__version__ = "0.1.0"

from . import dynamics, population, ews, estimator, geometry, monitor, detect

__all__ = [
    "dynamics",
    "population",
    "ews",
    "estimator",
    "geometry",
    "monitor",
    "detect",
    "__version__",
]
