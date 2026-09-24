"""Independent float64 NumPy reference backend for the K1/K2/K3 families.

These equations are correctness oracles, never performance backends. They are
implemented independently of the native Triton kernels and reject semantics
they do not represent.
"""

from __future__ import annotations

from .k1_routed import ReferenceResult, execute, merge_writes

__all__ = [
    "ReferenceResult",
    "execute",
    "merge_writes",
    "k1",
    "k1_attention",
    "k1_routed",
    "k2",
    "k2_operators",
    "k3",
    "graph",
    # Back-compat submodule aliases retained for the independent-parity tests.
    "matrix_state",
    "nonlinear_recurrence",
    "softmax_attention",
    "sparse_slot",
    "routed",
    "composition",
]


def __getattr__(name: str):
    import importlib

    # Canonical submodule names.
    canonical = {
        "k1",
        "k1_attention",
        "k1_routed",
        "k2",
        "k2_operators",
        "k3",
        "graph",
    }
    # Aliases mapping the pre-cutover oracle module names onto their new homes.
    aliases = {
        "matrix_state": "k2",
        "nonlinear_recurrence": "k2_operators",
        "softmax_attention": "k1_attention",
        "sparse_slot": "k3",
        "routed": "k1_routed",
        "composition": "graph",
    }
    if name in canonical:
        return importlib.import_module(f"{__name__}.{name}")
    if name in aliases:
        return importlib.import_module(f"{__name__}.{aliases[name]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
