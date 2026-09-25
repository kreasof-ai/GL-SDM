"""K3 family — native Triton ops over indexed mutable state with certified routes.

One file per op: ``sparse_state`` (the state read/update mixer) and
``route_generation`` (the route-selection op producing the certified routes the
state ops consume). The family re-exports the combined provider tuple so the
registry's per-family import finds them.
"""

from .route_generation import K3RouteNativeTritonProvider, TritonSparseRouteBackend
from .sparse_state import K3NativeTritonProvider, TritonSparseStateMixerBackend

PROVIDERS = (K3NativeTritonProvider(), K3RouteNativeTritonProvider())

__all__ = [
    "PROVIDERS",
    "K3NativeTritonProvider",
    "K3RouteNativeTritonProvider",
    "TritonSparseStateMixerBackend",
    "TritonSparseRouteBackend",
]
