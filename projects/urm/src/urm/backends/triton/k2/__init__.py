"""K2 family — native Triton ops over a compact fixed-address state bundle.

One file per op: ``matrix_scan`` (the linear-delta matrix-state recurrence, the
plan-wired providers) and ``diagonal`` (the Mamba-2-class diagonal state-width
schedule, benchmark-facing). The family re-exports the combined provider tuple so
the registry's per-family import finds them.
"""

from .dyadic_banks import DyadicBankedStateNativeTritonProvider
from .matrix_scan import K2NativeDiagonalProvider, K2NativeMatrixProvider

PROVIDERS = (
    K2NativeDiagonalProvider(),
    K2NativeMatrixProvider(),
    DyadicBankedStateNativeTritonProvider(),
)

__all__ = ["PROVIDERS", "K2NativeDiagonalProvider", "K2NativeMatrixProvider", "DyadicBankedStateNativeTritonProvider"]
