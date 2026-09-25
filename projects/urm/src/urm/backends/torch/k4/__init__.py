"""K4 family — Torch reference ops: exact feedback substitution over the emitted stream."""

from .triangular_solve import TriangularSolveTorchReferenceProvider

PROVIDERS = (TriangularSolveTorchReferenceProvider(),)

__all__ = ["PROVIDERS", "TriangularSolveTorchReferenceProvider"]
