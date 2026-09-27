"""NumPy oracle for the k3 family — the independent high-precision equation."""

from .sparse_state import K3NumpyProvider

PROVIDERS = (K3NumpyProvider(),)

__all__ = ["PROVIDERS", "K3NumpyProvider"]
