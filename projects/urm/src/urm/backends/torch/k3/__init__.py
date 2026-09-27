"""K3 family — Torch reference ops over indexed mutable state with certified routes."""

from .sparse_state import K3TorchReferenceProvider

PROVIDERS = (K3TorchReferenceProvider(),)

__all__ = ["PROVIDERS", "K3TorchReferenceProvider"]
