"""Runtime binding of compiled Sparse Memory plans to their native backend.

The compiler produces a validated :class:`~urm.compiler.partition.k3.SparseMemoryPlan`;
this module owns the executable binding: it constructs the native GPU backend,
checks the serialized schedule against the backend's actual launch schedule,
and exposes the bound executable. Importing the GPU backend is deferred until a
plan is compiled, so importing this module stays dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass

from urm.compiler.pipeline import CompilationResult, UrmCompiler
from urm.ir.program import SparseMemoryMixerSpec
from urm.compiler.partition.k3 import plan_sparse_memory


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
    plan = plan_sparse_memory(spec, compiler=compiler)

    from urm.backends.triton.k3.memory import TritonSparseMemoryBackend

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
        key: {"serialized": plan.launch_config.get(key), "runtime": value}
        for key, value in expected.items()
        if plan.launch_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "serialized Sparse Memory schedule does not match production launch: "
            f"{mismatches}"
        )
    return CompiledSparseMemoryPlan(plan.compilation, backend, plan.launch_config)


__all__ = ["CompiledSparseMemoryPlan", "compile_sparse_memory_plan"]
