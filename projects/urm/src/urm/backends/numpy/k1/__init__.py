"""NumPy oracle for the k1 family — the independent high-precision equation."""

from .softmax import K1NumpyProvider

PROVIDERS = (K1NumpyProvider(),)

__all__ = ["PROVIDERS", "K1NumpyProvider"]
