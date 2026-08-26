"""GPU integration: a Z3-selected, independently verified schedule executes.

Bounded to one shape and one solve; proves the pipeline end-to-end on real
hardware - constraint model, solver passes, imperative verification, then an
actual Triton launch whose output matches the eager reference. Skips without
CUDA or without the optional solver extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for solver-guided GPU integration", allow_module_level=True
    )

from urm.compiler.constraints import Assignment
from urm.compiler.kernel_plan import decode_schedule_point, verify_schedule_assignment
from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
from urm.compiler.schedule_space import PlanKind, SchedulePoint
from urm.compiler.semantic import DType, row_scaled_routed_reduction_program
from urm.compiler.solver import OptimizationPass, z3_available

pytestmark = pytest.mark.skipif(
    not z3_available(), reason="z3-solver optional extra not installed"
)

BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _reference(indices, weights, values, row_scale):
    base = torch.einsum("qk,qkd->qd", weights.float(), values.float()[indices.long()])
    return row_scale.float()[:, None] * base


def test_solver_selected_schedule_runs_and_matches_reference() -> None:
    import epilogue_schedules as sched

    queries, route_width, sources, value_dim = 64, 4, 128, 96
    program = row_scaled_routed_reduction_program(
        queries=queries,
        route_width=route_width,
        sources=sources,
        value_dim=value_dim,
        value_dtype=DType.BFLOAT16,
    )
    compiler = UrmCompiler()
    model = compiler.build_constraints(
        program,
        next(
            c.candidate_id
            for c in compiler.enumerate_candidates(program, CompilationIntent.TRAINING)
            if c.kind == "rewrite"
        ),
        CompilationIntent.TRAINING,
        schedule_params=ScheduleParams(),
    )
    solved = OptimizationPass().run(model)
    assert solved.status.value == "sat"

    assignment: Assignment = solved.assignment
    report = verify_schedule_assignment(model, assignment)
    assert report.ok, report.failures[:3]

    point: SchedulePoint = decode_schedule_point(model, assignment)
    assert point.plan == PlanKind.FUSED.value

    indices, weights, values, row_scale = sched.make_inputs(
        queries, route_width, sources, value_dim, "bfloat16", seed=5
    )
    output = sched.forward_launch(point, indices, weights, values, row_scale)[0]
    expected = _reference(indices, weights, values, row_scale)
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)


def test_deterministic_training_is_structurally_rejected() -> None:
    """No legal schedule exists for bitwise-deterministic training today."""
    program = row_scaled_routed_reduction_program(
        queries=32, route_width=4, sources=64, value_dim=64
    )
    compiler = UrmCompiler()
    model = compiler.build_constraints(
        program,
        next(
            c.candidate_id
            for c in compiler.enumerate_candidates(program, CompilationIntent.TRAINING)
            if c.kind == "rewrite"
        ),
        CompilationIntent.TRAINING,
        schedule_params=ScheduleParams(deterministic=True),
    )
    from urm.compiler.solver import FeasibilityPass

    result = FeasibilityPass().run(model)
    assert result.status.value == "unsat"
    names = " ".join(result.unsat_core_names)
    assert "deterministic_training_requires_ordered_grads" in names
