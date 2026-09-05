"""Experimental compiler-generated anchors (CODA-style rewriting prototypes).

These kernels are NOT part of routed-reduction v1 and never change its frozen
contract (``urm.triton_kernels.routed_reduce`` stays untouched). Each anchor
here exists because the URM planner selected it after a verified rewrite
folded work into an anchor's epilogue lifetime.

Importing this package imports Torch/Triton (the anchors ARE GPU code);
CPU-only callers go through the compiler core, which never imports it eagerly.
"""

from urm.compiler.anchors.routed_reduction_epilogue import (
    ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION,
    RoutedEpilogueLaunchConfig,
    RoutedEpilogueLaunchInfo,
    execute_plan_step,
    launch_backward,
    launch_forward,
    make_triton_compile_probe,
    routed_reduce_row_scale,
    routed_reduce_row_scale_metadata,
)
from urm.compiler.anchors.sparse_memory import (
    CompiledSparseMemoryPlan,
    compile_sparse_memory_plan,
)

__all__ = [
    "ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION",
    "CompiledSparseMemoryPlan",
    "RoutedEpilogueLaunchConfig",
    "RoutedEpilogueLaunchInfo",
    "compile_sparse_memory_plan",
    "execute_plan_step",
    "launch_backward",
    "launch_forward",
    "make_triton_compile_probe",
    "routed_reduce_row_scale",
    "routed_reduce_row_scale_metadata",
]
