"""NumPy oracle for the k4 family — the independent high-precision equation."""

from .triangular_solve import K4NumpyProvider

PROVIDERS = (K4NumpyProvider(),)

__all__ = ["PROVIDERS", "K4NumpyProvider"]
