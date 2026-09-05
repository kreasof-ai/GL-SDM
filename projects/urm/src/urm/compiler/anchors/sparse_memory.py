"""Executable compiler-produced native Sparse Memory plans."""

from __future__ import annotations

from dataclasses import dataclass

from urm.compiler.execution import NATIVE_SPARSE_MEMORY_ANCHOR_NAME
from urm.compiler.planner import CompilationIntent, CompilationResult, UrmCompiler
from urm.compiler.semantic import (
    SDMExecutionMode,
    SparseMemoryMixerSpec,
    sparse_delta_memory_program,
)


@dataclass(frozen=True, slots=True)
class CompiledSparseMemoryPlan:
    """Bound executable whose dispatch is authorized by a serialized plan."""

    compilation: CompilationResult
    backend: object
    launch_config: dict[str, str | int]

    @property
    def spec(self) -> SparseMemoryMixerSpec:
        return self.backend.spec

    @property
    def read_backend(self):
        return self.backend.read_backend

    @property
    def write_backend(self):
        return self.backend.write_backend

    @property
    def state_backend(self):
        return self.backend.state_backend

    @property
    def read_spec(self):
        return self.backend.read_spec

    @property
    def write_spec(self):
        return self.backend.write_spec

    @property
    def state_spec(self):
        return self.backend.state_spec

    def prepare(self, *args, **kwargs):
        return self.backend.prepare(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self.backend.execute(*args, **kwargs)

    def serialized_plan(self) -> dict[str, object]:
        return self.compilation.plan.to_dict()


def compile_sparse_memory_plan(
    spec: SparseMemoryMixerSpec,
    *,
    compiler: UrmCompiler | None = None,
) -> CompiledSparseMemoryPlan:
    """Compile, verify, and bind the exact native Sparse Memory schedule."""
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
    required = {
        "schedule_family": "native_route_then_partition_scan",
        "route_materialization": "explicit_logical_outputs",
        "fusion": "none",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            f"compiler emitted incompatible Sparse Memory plan: {config}"
        )

    from urm.backends.sparse_memory import TritonSparseMemoryBackend

    backend = TritonSparseMemoryBackend(spec)
    read_schedule = backend.read_backend.launch_schedule()
    state_schedule = backend.state_backend.launch_schedule()
    expected = {
        "route_block_half": read_schedule["block_half"],
        "read_route_block": read_schedule["block_route"],
        "write_route_block": (
            backend.write_backend.launch_schedule()["block_route"]
            if backend.write_backend is not None
            else 0
        ),
        "state_block_d": state_schedule["block_d"],
        "state_num_warps": state_schedule["num_warps"],
        "state_num_stages": state_schedule["num_stages"],
        "read_route_num_warps": read_schedule["num_warps"],
        "write_route_num_warps": (
            backend.write_backend.launch_schedule()["num_warps"]
            if backend.write_backend is not None
            else 4
        ),
        "route_backward_num_warps": 4,
        "route_num_stages": read_schedule["num_stages"],
    }
    mismatches = {
        key: {"serialized": config.get(key), "runtime": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "serialized Sparse Memory schedule does not match production launch: "
            f"{mismatches}"
        )
    return CompiledSparseMemoryPlan(compilation, backend, config)


__all__ = ["CompiledSparseMemoryPlan", "compile_sparse_memory_plan"]
