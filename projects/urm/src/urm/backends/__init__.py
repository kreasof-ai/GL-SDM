"""Backend packages: pure capability contracts plus K1/K2/K3 and reference.

This package keeps imports lazy where possible. It exposes the native Triton
and reference backend implementations without owning candidate choice, cost, or
schedule decisions - those belong to the compiler stages.
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
        from .reference.torch.k1 import TorchRoutedReductionBackend

        return TorchRoutedReductionBackend
    if name == "TritonRoutedReductionBackend":
        from .triton.k1.routed_launcher import TritonRoutedReductionBackend

        return TritonRoutedReductionBackend
    if name == "TritonSparseRouteBackend":
        from .triton.k3.route_launcher import TritonSparseRouteBackend

        return TritonSparseRouteBackend
    if name == "TritonSparseStateMixerBackend":
        from .triton.k3.state_launcher import TritonSparseStateMixerBackend

        return TritonSparseStateMixerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
