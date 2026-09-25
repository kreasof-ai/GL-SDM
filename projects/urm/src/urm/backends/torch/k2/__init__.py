"""K2 family — Torch reference ops over a compact fixed-address state bundle.

One file per op: ``linear_delta`` (the linear-delta matrix-state recurrence) and
``dyadic_banks`` (the banked dyadic A4 schedule). The family re-exports the
combined provider tuple so the registry finds them.
"""

from .dyadic_banks import DyadicBankedStateTorchReferenceProvider
from .linear_delta import K2TorchReferenceProvider

PROVIDERS = (K2TorchReferenceProvider(), DyadicBankedStateTorchReferenceProvider())

__all__ = ["PROVIDERS", "K2TorchReferenceProvider", "DyadicBankedStateTorchReferenceProvider"]
