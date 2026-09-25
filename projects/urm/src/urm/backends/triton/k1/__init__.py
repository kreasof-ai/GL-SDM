"""K1 family — native Triton ops over the full source history (score/select/reduce).

One file per op. The family directory hosts the ops; this module only re-exports the
combined provider tuple so the registry's per-family import finds them.
"""

from .indexed import K1NativeIndexedTritonProvider
from .online_softmax import K1NativeTritonProvider

PROVIDERS = (K1NativeTritonProvider(), K1NativeIndexedTritonProvider())

__all__ = ["PROVIDERS", "K1NativeTritonProvider", "K1NativeIndexedTritonProvider"]
