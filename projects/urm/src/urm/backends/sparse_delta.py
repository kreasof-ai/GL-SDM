"""Sparse routed delta-update API; implementation remains experimental.

The compiler must validate capabilities before registering this lowering.
Routing is supplied as indices/weights; no architecture or indexer is prescribed.
"""

from urm.experimental.sparse_delta import (
    SparseDeltaFunction,
    chunked_sparse_delta,
    sparse_delta,
)

__all__ = ["SparseDeltaFunction", "chunked_sparse_delta", "sparse_delta"]
