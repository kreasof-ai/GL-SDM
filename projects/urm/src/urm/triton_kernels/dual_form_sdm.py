"""Legacy model-named aliases for the experimental sparse-delta lowering."""

from urm.experimental.triton_sparse_delta import (
    TritonSparseDeltaFunction as TritonDualFormSDMFunction,
)
from urm.experimental.triton_sparse_delta import (
    _sparse_delta_bwd_kernel as _triton_dual_form_bwd_kernel,
)
from urm.experimental.triton_sparse_delta import (
    _sparse_delta_fwd_kernel as _triton_dual_form_fwd_kernel,
)
from urm.experimental.triton_sparse_delta import (
    triton_sparse_delta as triton_dual_form_sdm,
)

__all__ = [
    "TritonDualFormSDMFunction",
    "_triton_dual_form_bwd_kernel",
    "_triton_dual_form_fwd_kernel",
    "triton_dual_form_sdm",
]
