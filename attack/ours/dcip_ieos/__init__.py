"""DCIP-IEOS attack modules.

The package exposes a very small subset of the original research code.  It is
primarily intended for unit tests and does not pull in heavy third‑party
dependencies.  The :class:`VictimAdapter` wrapper is provided for convenience
when a caller wishes to query a real model, yet it gracefully degrades when the
required libraries are unavailable.
"""

from .victim_adapter import VictimAdapter  # noqa: F401

__all__ = ["VictimAdapter"]