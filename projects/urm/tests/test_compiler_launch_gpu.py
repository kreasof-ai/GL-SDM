"""GPU integration: a schedule selected by UrmCompiler.compile() reaches the
PRODUCTION routed-row-scale Triton launcher.

Not a benchmark replay: the executable plan serialized by compile() is parsed
back into a launch configuration and executed through the production anchor
launchers. Asserts actual launch metadata (grid, num_warps from the compiled
Triton handle) matches the selected configuration, that changing a legal
pinned configuration changes real launch metadata, and that forward and ALL
backward gradients match the eager reference within the existing envelopes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for compiler->kernel launch integration", allow_module_level=True
    )

from urm.compiler.anchors.routed_reduction_epilogue import (
    RoutedEpilogueLaunchConfig,
    execute_plan_step,
    make_triton_compile_probe,
    routed_reduce_row_scale,
)
from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
from urm.compiler.search import CompileStatus
from urm.compiler.semantic import DType, row_scaled_routed_reduction_program
from urm.compiler.solver import z3_available

pytestmark = pytest.mark.skipif(
    not z3_available(), reason="z3-solver optional extra not installed"
)

FORWARD_TOL = {"atol": 2e-2, "rtol": 2e-2}  # bfloat16 envelope (existing tests)
BACKWARD_TOL = {"atol": 8e-2, "rtol": 4e-2}


def _compile(schedule_params: ScheduleParams | None = None):
    program = row_scaled_routed_reduction_program(
        queries=64,
        route_width=4,
        sources=128,
        value_dim=96,
        value_dtype=DType.BFLOAT16,
    )
    compiler = UrmCompiler(compile_probe=make_triton_compile_probe())
    result = compiler.compile(
        program,
        intent=CompilationIntent.TRAINING,
        schedule_params=schedule_params or ScheduleParams(),
    )
    decision = result.schedule_decision
    assert decision is not None
    return result, decision


def _plan_launch_config(result) -> dict:
    payload = result.plan.to_dict()
    configs = [
        step["launch_config"]
        for step in payload["steps"]
        if step["kind"] == "anchor_dispatch" and step["launch_config"]
    ]
    assert configs, "anchor-dispatch steps must carry the launch configuration"
    return configs[0]


def _reference(indices, weights, values, row_scale):
    base = torch.einsum("qk,qkd->qd", weights.float(), values.float()[indices.long()])
    return row_scale.float()[:, None] * base


def _sample(seed: int = 5):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    indices = torch.randint(0, 128, (64, 4), device="cuda", generator=generator)
    weights = torch.randn((64, 4), device="cuda", generator=generator).bfloat16()
    values = torch.randn((128, 96), device="cuda", generator=generator).bfloat16()
    scale = torch.randn((64,), device="cuda", generator=generator).bfloat16()
    return indices, weights, values, scale


def test_compiled_schedule_reaches_production_triton_launch() -> None:
    result, decision = _compile()
    # The real GPU probe compiled and launched this exact configuration.
    assert decision.compile_status is CompileStatus.SUCCEEDED

    config = RoutedEpilogueLaunchConfig.from_dict(_plan_launch_config(result))
    indices, weights, values, scale = _sample()
    output, info = execute_plan_step(
        _plan_launch_config(result), indices, weights, values, scale
    )
    # Actual launch metadata equals the selected configuration.
    assert info.block_d == config.block_d
    assert info.num_warps == config.num_warps
    assert info.grid == (64, -(-96 // config.block_d))
    handle = info.handle
    assert handle is not None
    assert int(handle.metadata.num_warps) == config.num_warps
    expected = _reference(indices, weights, values, scale)
    torch.testing.assert_close(output.float(), expected, **FORWARD_TOL)


def test_changing_legal_pinned_config_changes_actual_launch_metadata() -> None:
    pinned_a = ScheduleParams(block_hints={"BLOCK_D": 32}, warp_count=1)
    result_a, decision_a = _compile(schedule_params=pinned_a)
    result_b, decision_b = _compile(
        schedule_params=ScheduleParams(block_hints={"BLOCK_D": 64}, warp_count=4),
    )
    assert decision_a.schedule_point.block_d == 32
    assert decision_a.schedule_point.num_warps == 1
    assert decision_b.schedule_point.block_d == 64
    assert decision_b.schedule_point.num_warps == 4

    indices, weights, values, scale = _sample(seed=9)
    _out_a, info_a = execute_plan_step(
        _plan_launch_config(result_a), indices, weights, values, scale
    )
    _out_b, info_b = execute_plan_step(
        _plan_launch_config(result_b), indices, weights, values, scale
    )
    assert int(info_a.handle.metadata.num_warps) == 1
    assert int(info_b.handle.metadata.num_warps) == 4
    assert info_a.grid != info_b.grid  # BLOCK_D change resizes the D-axis grid


def test_all_backward_gradients_match_eager_reference() -> None:
    result, decision = _compile()
    assert decision.compile_status is CompileStatus.SUCCEEDED
    config = RoutedEpilogueLaunchConfig.from_dict(_plan_launch_config(result))

    indices, weights, values, scale = _sample(seed=13)
    w = weights.clone().requires_grad_(True)
    v = values.clone().requires_grad_(True)
    r = scale.clone().requires_grad_(True)

    output = routed_reduce_row_scale(indices, w, v, r, config=config)
    generator = torch.Generator(device="cuda").manual_seed(77)
    grad = torch.randn(
        output.shape, device="cuda", dtype=torch.float32, generator=generator
    )
    output.backward(grad)

    w_ref = weights.clone().requires_grad_(True)
    v_ref = values.clone().requires_grad_(True)
    r_ref = scale.clone().requires_grad_(True)
    _reference(indices, w_ref, v_ref, r_ref).backward(grad)

    for got, ref in ((w.grad, w_ref.grad), (v.grad, v_ref.grad), (r.grad, r_ref.grad)):
        assert got is not None and torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), ref.float(), **BACKWARD_TOL)


@pytest.mark.parametrize(
    "decomp,sched",
    [
        ("per_query", "segmented"),
        ("per_query", "full_row"),
        ("per_route", "segmented"),
    ],
)
def test_all_backward_decompositions_match_eager_reference(
    decomp: str, sched: str
) -> None:
    config = RoutedEpilogueLaunchConfig(
        block_d=32,
        num_warps=2,
        num_stages=1,
        grad_values_decomposition=decomp,
        grad_values_schedule=sched,
    )
    indices, weights, values, scale = _sample(seed=17)
    w = weights.clone().requires_grad_(True)
    v = values.clone().requires_grad_(True)
    r = scale.clone().requires_grad_(True)

    output = routed_reduce_row_scale(indices, w, v, r, config=config)
    generator = torch.Generator(device="cuda").manual_seed(77)
    grad = torch.randn(
        output.shape, device="cuda", dtype=torch.float32, generator=generator
    )
    output.backward(grad)

    w_ref = weights.clone().requires_grad_(True)
    v_ref = values.clone().requires_grad_(True)
    r_ref = scale.clone().requires_grad_(True)
    _reference(indices, w_ref, v_ref, r_ref).backward(grad)

    for got, ref in ((w.grad, w_ref.grad), (v.grad, v_ref.grad), (r.grad, r_ref.grad)):
        assert got is not None and torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), ref.float(), **BACKWARD_TOL)


def test_base_schedules_probe_base_anchor() -> None:
    from urm.compiler.schedule_space import SchedulePoint
    from urm.compiler.search import CompileContext

    probe = make_triton_compile_probe(
        queries=4, route_width=2, sources=8, value_dim=32, dtype_name="bfloat16"
    )
    dummy_point = SchedulePoint(
        plan="base",
        block_d=32,
        num_warps=2,
        num_stages=1,
        grad_values_decomposition="per_query",
        grad_values_schedule="segmented",
        dtype="bfloat16",
    )
    context = CompileContext(
        anchor_name="routed_reduction_v1",
        plan="base",
        intent="inference",
        queries=4,
        sources=8,
        route_width=2,
        value_dim=32,
        dtype="bfloat16",
        block_d=32,
        num_warps=2,
        num_stages=1,
        grad_values_decomposition="per_query",
        grad_values_schedule="segmented",
        schedule_point=dummy_point,
    )
    res = probe(context)
    assert res.ok
