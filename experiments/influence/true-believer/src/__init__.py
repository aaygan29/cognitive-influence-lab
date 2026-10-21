"""cultist: demonstrating and quantifying a neural-grounded belief-imparting risk.

The package exposes a small, auditable instrument:
  - encoders:   frozen brain-encoder interface (reference + pluggable real encoders)
  - signature:  B(s) = E + V - R, the neural belief-formation signature from content
  - baseline:   a content-only persuasion model (the thing B must beat)
  - fusion:     a latent-variable belief model fusing several validated indicators
  - analysis:   H-load / H-incremental / H-bound with pre-committed kill criteria

No em dashes. Public data only. No deployable persuasion optimizer.
"""

__version__ = "0.1.0"
