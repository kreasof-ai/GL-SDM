"""K4 family — exact feedback substitution over the op's own emitted history.

A K4 op's per-token value depends on the op's OWN outputs at earlier positions
through a given triangular transition operator, resolved exactly by ordered
substitution (serial in t, full-history read). Distinct read-domain from K1
(external source streams), K2 (compact fixed-address state) and K3 (routed
slots). One file per op; the family re-exports the combined provider tuple.
"""

from .triangular_solve import TriangularSolveNativeTritonProvider

PROVIDERS = (TriangularSolveNativeTritonProvider(),)

__all__ = ["PROVIDERS", "TriangularSolveNativeTritonProvider"]
