"""Independent float64 NumPy reference equations for the K1/K2/K3 families.

These equations are correctness oracles, never performance backends. They are
implemented independently of the native Triton kernels and reject semantics
they do not represent.
"""

from __future__ import annotations

__all__ = [
    "k1_attention",
    "k2",
    "k2_operators",
    "k3",
]


def __getattr__(name: str):
    import importlib

    if name in set(__all__):
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
