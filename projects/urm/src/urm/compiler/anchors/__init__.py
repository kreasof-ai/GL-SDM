"""Experimental compiler-generated anchors (CODA-style rewriting prototypes).

These kernels are NOT part of routed-reduction v1 and never change its frozen
contract (``urm.triton_kernels.routed_reduce`` stays untouched). Each anchor
here exists because the URM planner selected it after a verified rewrite
folded work into an anchor's epilogue lifetime.
"""

from urm.compiler.anchors.routed_reduction_epilogue import (
    ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION,
    routed_reduce_row_scale,
    routed_reduce_row_scale_metadata,
)

__all__ = [
    "ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION",
    "routed_reduce_row_scale",
    "routed_reduce_row_scale_metadata",
]
