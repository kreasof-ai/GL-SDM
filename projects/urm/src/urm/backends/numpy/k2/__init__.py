"""NumPy oracle for the k2 family — the independent high-precision equation."""

from .dyadic_banks import K2DyadicBankedStateNumpyProvider
from .linear_delta import K2NumpyProvider

PROVIDERS = (K2NumpyProvider(), K2DyadicBankedStateNumpyProvider())

__all__ = ["PROVIDERS", "K2NumpyProvider", "K2DyadicBankedStateNumpyProvider"]
