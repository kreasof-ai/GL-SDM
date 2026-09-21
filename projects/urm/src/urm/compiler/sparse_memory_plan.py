"""Compiler-owned plan construction and scheduling for native Sparse Memory.

This module builds the typed semantic program, runs the compiler pipeline, and
validates that the resulting executable plan selects the native Sparse Memory
anchor with the expected schedule. It performs no GPU work and imports no
optional backend: binding the plan to an executable backend is the runtime's
responsibility (see :mod:`urm.runtime.sparse_memory`).
"""

from __future__ import annotations

from dataclasses import dataclass

from urm.compiler.execution import NATIVE_SPARSE_MEMORY_ANCHOR_NAME
from urm.compiler.planner import CompilationIntent, CompilationResult, UrmCompiler
from urm.compiler.semantic import (
    SDMExecutionMode,
    SparseMemoryMixerSpec,
    sparse_delta_memory_program,
)

#: The schedule the compiler must produce for the native Sparse Memory pipeline.
REQUIRED_SPARSE_MEMORY_SCHEDULE: dict[str, str] = {
    "schedule_family": "native_route_then_partition_scan",
    "route_materialization": "explicit_logical_outputs",
    "fusion": "none",
}


@dataclass(frozen=True, slots=True)
class SparseMemoryPlan:
    """A validated Sparse Memory compilation and its serialized launch config.

    This is the compiler's output: a verified schedule plus the compilation
    that produced it. It carries no executable backend.
    """

    compilation: CompilationResult
    launch_config: dict[str, str | int]

    def serialized_plan(self) -> dict[str, object]:
        return self.compilation.plan.to_dict()


def plan_sparse_memory(
    spec: SparseMemoryMixerSpec,
    *,
    compiler: UrmCompiler | None = None,
) -> SparseMemoryPlan:
    """Compile and validate the native Sparse Memory schedule for ``spec``.

    Raises unless the compiler selects exactly the native Sparse Memory anchor
    with the required schedule. No backend is constructed here.
    """
    program = sparse_delta_memory_program(
        name="compiled_sparse_memory",
        parallel=spec.parallel,
        sequence=spec.sequence,
        slots_per_partition=spec.slots_per_partition,
        value_dim=spec.value_dim,
        writes=spec.writes,
        reads=spec.reads,
        dtype=spec.dtype,
        mode=spec.mode,
        operation=spec.operation,
        read_timing=spec.read_timing,
    )
    intent = (
        CompilationIntent.TRAINING
        if spec.mode is SDMExecutionMode.TRAINING
        else CompilationIntent.INFERENCE
    )
    compilation = (compiler or UrmCompiler()).compile(program, intent=intent)
    dispatch = [
        step for step in compilation.plan.steps if step.kind == "anchor_dispatch"
    ]
    if len(dispatch) != 1 or dispatch[0].anchor != NATIVE_SPARSE_MEMORY_ANCHOR_NAME:
        selected = [step.anchor for step in dispatch]
        raise RuntimeError(
            "compiler did not produce the native Sparse Memory executable: "
            f"selected={selected}"
        )
    config = dict(dispatch[0].launch_config or {})
    if any(
        config.get(key) != value
        for key, value in REQUIRED_SPARSE_MEMORY_SCHEDULE.items()
    ):
        raise RuntimeError(
            f"compiler emitted incompatible Sparse Memory plan: {config}"
        )
    return SparseMemoryPlan(compilation, config)


__all__ = [
    "REQUIRED_SPARSE_MEMORY_SCHEDULE",
    "SparseMemoryPlan",
    "plan_sparse_memory",
]
