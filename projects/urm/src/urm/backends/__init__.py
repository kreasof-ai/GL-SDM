"""Backend packages: the single Provider contract plus per-family providers.

Every backend — reference NumPy oracle, reference Torch, native Triton, or the
trusted library tier — is a :class:`~urm.backends.providers.Provider` living in
its family package (``providers/k1``, ``providers/k2``, ``providers/k3``). A
future backend implements that one surface; there is no per-tier ad-hoc entry
point. Imports are lazy where possible; the compiler owns candidate choice,
cost and schedule decisions, not this package.
"""

from __future__ import annotations

__all__ = [
    "TorchRoutedReductionBackend",
    "TritonRoutedReductionBackend",
    "TritonSparseRouteBackend",
    "TritonSparseStateMixerBackend",
]


def __getattr__(name: str):
    if name == "TorchRoutedReductionBackend":
        from .providers.k1.routed_torch import TorchRoutedReductionBackend

        return TorchRoutedReductionBackend
    if name == "TritonRoutedReductionBackend":
        from .providers.k1.routed_launcher import TritonRoutedReductionBackend

        return TritonRoutedReductionBackend
    if name == "TritonSparseRouteBackend":
        from .providers.k3.triton_route_launcher import TritonSparseRouteBackend

        return TritonSparseRouteBackend
    if name == "TritonSparseStateMixerBackend":
        from .providers.k3.triton_state_launcher import TritonSparseStateMixerBackend

        return TritonSparseStateMixerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
