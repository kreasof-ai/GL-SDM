"""Differential + property tests for the URM compiler core (CPU-only)."""

from __future__ import annotations

import pytest

from urm.compiler.cost import (
    CostEstimate,
    DeviceLimits,
    combine,
    exchange_cost,
    routed_reduction_cost,
    row_scale_transform_cost,
)
from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.effects import EffectClass, EffectKind
from urm.compiler.execution import (
    AnchorKind,
    AnchorRequest,
    VisitorDescriptor,
    VisitorKind,
    default_registry,
    make_selector,
)
from urm.compiler.locality import Locality, LocalityConstraint, movement_allowed
from urm.compiler.placement import (
    DeviceMesh,
    PlacementBinding,
    PlacementMap,
    RouteLeg,
)
from urm.compiler.planner import (
    ExecutablePlan,
    ScheduleParams,
    UrmCompiler,
    plan_route_distribution,
)
from urm.compiler.rewrite import (
    DELAY_ROW_SCALE_THROUGH_GEMM,
    FOLD_ROW_SCALE_EPILOGUE,
    EquivalenceClass,
    RewriteEngine,
    SavedStatePolicy,
)
from urm.compiler.semantic import (
    CollectiveExchange,
    DType,
    LogicalDomain,
    Matmul,
    MergePolicy,
    OrderedRecurrence,
    RouteSpec,
    ScoreNormalization,
    SelectionKind,
    SemanticProgram,
    StateUpdate,
    TensorHandle,
    Transform,
    TransformKind,
    WeightedReduce,
    routed_reduction_program,
    row_scaled_routed_reduction_program,
)

# -- locality ------------------------------------------------------------------


def test_locality_ladder_is_totally_ordered() -> None:
    ladder = [
        Locality.REGISTER,
        Locality.LANE,
        Locality.TILE,
        Locality.BLOCK,
        Locality.DEVICE,
        Locality.MESH,
    ]
    from itertools import pairwise

    for lower, higher in pairwise(ladder):
        assert lower < higher
        assert higher.covers(lower)
    assert not movement_allowed(Locality.DEVICE, Locality.MESH)
    assert movement_allowed(Locality.DEVICE, Locality.TILE)


def test_locality_windows_intersect() -> None:
    tile_window = LocalityConstraint(Locality.TILE, Locality.DEVICE)
    device_only = LocalityConstraint(Locality.DEVICE, Locality.DEVICE)
    assert tile_window.intersect(device_only) == device_only
    register_only = LocalityConstraint(Locality.REGISTER, Locality.REGISTER)
    assert tile_window.intersect(register_only) is None


# -- semantic IR ---------------------------------------------------------------


def _reduce_spec() -> RouteSpec:
    return RouteSpec(
        query_domain=LogicalDomain.SEQUENCE,
        source_domain=LogicalDomain.EXPERT,
        selection=SelectionKind.TOP_K,
        normalization=ScoreNormalization.SOFTMAX,
        top_k=2,
    )


def test_program_validation_rejects_unknown_tensors() -> None:
    with pytest.raises(CompilerError, match="never defined"):
        SemanticProgram.build(
            name="bad",
            inputs=(TensorHandle("w", DType.FLOAT32, ("q", "k")),),
            ops=(
                WeightedReduce(
                    name="r",
                    inputs=("missing", "w"),
                    outputs=("base",),
                    spec=_reduce_spec(),
                ),
            ),
            outputs=("base",),
        )


def test_state_update_requires_explicit_policy_and_unique_commits() -> None:
    with pytest.raises(ValueError, match="explicit merge policy"):
        StateUpdate(name="u", inputs=("d",), outputs=("s",))
    ordered = StateUpdate(
        name="u",
        inputs=("d",),
        outputs=("s",),
        state="mem",
        policy=MergePolicy.ORDERED,
    )
    assert ordered.effect.writes >= {EffectClass.ORDERED_STATE_TRANSITION}
    commit = StateUpdate(
        name="u",
        inputs=("d",),
        outputs=("s",),
        state="mem",
        policy=MergePolicy.SUM,
        commit_boundary=True,
    )
    assert EffectKind.COMMITS in commit.effect.flags()
    with pytest.raises(CompilerError, match="more than one commit"):
        SemanticProgram.build(
            name="double_commit",
            inputs=(
                TensorHandle("d", DType.FLOAT32, ("n", "v")),
                TensorHandle("d2", DType.FLOAT32, ("n", "v")),
            ),
            ops=(
                StateUpdate(
                    name="u1",
                    inputs=("d",),
                    outputs=("s1",),
                    state="mem",
                    policy=MergePolicy.SUM,
                    commit_boundary=True,
                ),
                StateUpdate(
                    name="u2",
                    inputs=("d2",),
                    outputs=("s2",),
                    state="mem",
                    policy=MergePolicy.SUM,
                    commit_boundary=True,
                ),
            ),
            outputs=("s1", "s2"),
        )


def test_ordered_recurrence_is_a_movement_barrier() -> None:
    scan = OrderedRecurrence(
        name="scan", inputs=("x",), outputs=("y",), algorithm="gated_delta_rule"
    )
    crossed = set(scan.effect.all_classes)
    from urm.compiler.effects import BARRIERS

    assert crossed & BARRIERS


def test_collective_intent_requires_placement_at_compile_time() -> None:
    program = SemanticProgram.build(
        name="collective",
        inputs=(TensorHandle("x", DType.FLOAT32, ("n", "d")),),
        ops=(CollectiveExchange(name="ex", inputs=("x",), outputs=("y",)),),
        outputs=("y",),
    )
    compiler = UrmCompiler()
    with pytest.raises(CompilerError, match="no placement was given"):
        compiler.compile(program)


# -- rewrite engine ------------------------------------------------------------


def test_fold_row_scale_epilogue_positive_and_negative() -> None:
    positive = row_scaled_routed_reduction_program(
        queries=8, route_width=4, sources=16, value_dim=32
    )
    engine = RewriteEngine()
    result = engine.apply(positive)
    assert result.changed
    reduce_op = result.program.find("reduce")
    assert isinstance(reduce_op, WeightedReduce) and reduce_op.epilogue is not None
    assert result.program.find("apply_row_scale") is None
    attempts = {attempt.outcome for attempt in result.trace.attempts}
    assert attempts == {"accepted"}
    kinds = [obligation.kind for obligation in result.trace.obligations]
    # Backward is certified for every supported dtype: the recompute
    # obligation stands, but no contradictory forward-only obligation exists.
    assert "recompute_backward" in kinds
    assert "forward_only" not in kinds

    # Negative: nonlinear transform between reduce and scale.
    program = SemanticProgram.build(
        name="nonlinear_chain",
        inputs=routed_reduction_program().inputs
        + (TensorHandle("row_scale", DType.FLOAT32, ("queries",)),),
        ops=(
            *routed_reduction_program().ops,
            Transform(
                name="nonlin",
                inputs=("base",),
                outputs=("activated",),
                kind=TransformKind.GELU,
            ),
            Transform(
                name="apply_row_scale",
                inputs=("activated", "row_scale"),
                outputs=("output",),
                kind=TransformKind.ROW_SCALE,
            ),
        ),
        outputs=("output",),
    )
    negative_result = RewriteEngine([FOLD_ROW_SCALE_EPILOGUE]).apply(program)
    assert not negative_result.changed
    rejection = next(
        attempt
        for attempt in negative_result.trace.attempts
        if attempt.outcome == "rejected"
    )
    assert rejection.reason_code == DiagnosticCode.REWRITE_PRECONDITION_FAILED.value


def test_row_scale_fold_rejected_across_transaction_boundary() -> None:
    """Mutation effects between producer and subject block the rewrite."""
    inner = routed_reduction_program(queries=8, route_width=2, sources=8, value_dim=8)
    program = SemanticProgram.build(
        name="across_commit",
        inputs=(
            *inner.inputs,
            TensorHandle("delta", DType.FLOAT32, ("n", "v")),
            TensorHandle("row_scale", DType.FLOAT32, ("queries",)),
        ),
        ops=(
            *inner.ops,
            StateUpdate(
                name="commit_state",
                inputs=("delta",),
                outputs=("state_out",),
                state="mem",
                policy=MergePolicy.SUM,
                commit_boundary=True,
            ),
            Transform(
                name="apply_row_scale",
                inputs=("base", "row_scale"),
                outputs=("output",),
                kind=TransformKind.ROW_SCALE,
            ),
        ),
        outputs=("output",),
    )
    result = RewriteEngine([FOLD_ROW_SCALE_EPILOGUE]).apply(program)
    assert not result.changed
    rejection = next(a for a in result.trace.attempts if a.outcome == "rejected")
    assert rejection.reason_code == DiagnosticCode.REWRITE_EFFECT_UNSAFE.value
    assert "transactional" in (rejection.detail or "")


def test_row_scale_fold_rejected_with_multiple_consumers() -> None:
    """If `base` is consumed twice, folding would duplicate or drop work."""
    inner = routed_reduction_program(queries=8, route_width=2, sources=8, value_dim=8)
    program = SemanticProgram.build(
        name="two_consumers",
        inputs=(
            *inner.inputs,
            TensorHandle("row_scale", DType.FLOAT32, ("queries",)),
            TensorHandle("aux", DType.FLOAT32, ("queries",)),
        ),
        ops=(
            *inner.ops,
            Transform(
                name="apply_row_scale",
                inputs=("base", "row_scale"),
                outputs=("output",),
                kind=TransformKind.ROW_SCALE,
            ),
            Transform(
                name="aux_use",
                inputs=("base", "aux"),
                outputs=("aux_out",),
                kind=TransformKind.RELU,
            ),
        ),
        outputs=("output", "aux_out"),
    )
    result = RewriteEngine([FOLD_ROW_SCALE_EPILOGUE]).apply(program)
    assert not result.changed
    rejection = next(a for a in result.trace.attempts if a.outcome == "rejected")
    assert "2 consumers" in (rejection.detail or "")


def test_trace_is_deterministic_and_serializable() -> None:
    program = row_scaled_routed_reduction_program(
        queries=8, route_width=4, sources=16, value_dim=32
    )
    engine = RewriteEngine()
    first = engine.apply(program)
    second = engine.apply(program)
    assert first.trace.to_dict() == second.trace.to_dict()
    assert first.program == second.program
    import json

    json.dumps(first.trace.to_dict())


def test_delayed_scaling_rule_contract_fields() -> None:
    rule = DELAY_ROW_SCALE_THROUGH_GEMM
    # Algebraically exact over real arithmetic only: the rewrite reorders
    # floating-point operations, so it is classified FLOATING_POINT with
    # dtype-specific envelopes (never `exact`).
    assert rule.equivalence is EquivalenceClass.FLOATING_POINT
    assert rule.tolerance_envelope is not None
    assert (
        rule.tolerance_envelope["bfloat16_atol"]
        > rule.tolerance_envelope["float32_atol"]
    )
    assert rule.saved_state_policy is SavedStatePolicy.NONE
    assert not rule.forward_only
    folded = FOLD_ROW_SCALE_EPILOGUE
    assert folded.equivalence is EquivalenceClass.FLOATING_POINT
    assert (
        folded.tolerance_envelope["bfloat16_atol"]
        > folded.tolerance_envelope["float32_atol"]
    )
    # Backward is certified for weights, values AND the row scale.
    assert not folded.forward_only
    assert folded.backward_contract is not None
    assert set(folded.backward_contract.verified_dtypes) >= {
        DType.FLOAT32,
        DType.FLOAT16,
        DType.BFLOAT16,
    }
    assert folded.traffic_bytes_delta < 0


def _gemm_scaled_program() -> SemanticProgram:
    return SemanticProgram.build(
        name="gemm_scaled",
        inputs=(
            TensorHandle("x", DType.FLOAT32, ("m", "k")),
            TensorHandle("W", DType.FLOAT32, ("k", "n")),
            TensorHandle("r", DType.FLOAT32, ("m",)),
        ),
        ops=(
            Transform(
                name="rowscale",
                inputs=("x", "r"),
                outputs=("xs",),
                kind=TransformKind.ROW_SCALE,
            ),
            Matmul(name="linear", inputs=("xs", "W"), outputs=("y",)),
        ),
        outputs=("y",),
    )


def test_delayed_scaling_gemm_rewrite_applies() -> None:
    program = _gemm_scaled_program()
    engine = RewriteEngine([DELAY_ROW_SCALE_THROUGH_GEMM])
    candidates = engine.candidates_for(program)
    assert [rule.name for rule, _match in candidates] == [
        "delay_row_scale_through_linear_matmul"
    ]
    result = engine.apply(program)
    assert result.changed
    names = result.program.op_names
    assert names == ("linear", "rowscale__delayed")
    delayed_matmul = result.program.find("linear")
    assert isinstance(delayed_matmul, Matmul)
    assert delayed_matmul.inputs == ("x", "W")


# -- anchor selection ------------------------------------------------------------


def test_anchor_selection_prefers_v1_and_declines_without_epilogue_anchor() -> None:
    registry = default_registry()
    plain = registry.select(AnchorRequest(kind=AnchorKind.ROUTED_REDUCTION))
    assert plain.ok and plain.anchor is not None
    assert plain.anchor.name == "routed_reduction_v1"

    tiled_visitor = (
        VisitorDescriptor(
            kind=VisitorKind.FINAL_SCALE_CONVERT,
            element_dtype="float32",
            locality=Locality.TILE,
        ),
    )
    fused = registry.select(
        AnchorRequest(kind=AnchorKind.ROUTED_REDUCTION, visitors=tiled_visitor)
    )
    assert fused.ok and fused.anchor is not None
    assert fused.anchor.name == "routed_reduction_row_scale_epilogue_v0"
    assert fused.anchor.experimental

    # An unregistered visitor kind is declined explicitly, never fudged.
    exotic = (
        VisitorDescriptor(
            kind=VisitorKind.STATEFUL_TILE_TRANSFORM,
            element_dtype="float32",
            locality=Locality.TILE,
        ),
    )
    decision = registry.select(
        AnchorRequest(kind=AnchorKind.COLLECTIVE_EXCHANGE, visitors=exotic)
    )
    # The collective anchor cannot honor that visitor: it declines explicitly
    # instead of silently dropping it (charter invariant 9).
    assert not decision.ok
    assert decision.decline is not None


def test_selector_returns_decline_for_unsupported_visitors() -> None:
    selector = make_selector(())
    decision = selector(AnchorRequest(kind=AnchorKind.GEMM))
    assert decision is None  # abstains so other selectors may answer


# -- placement / communication planning ------------------------------------------


def _two_by_two_mesh_placement() -> PlacementMap:
    mesh = DeviceMesh(name="sim", shape=(2, 2))
    bindings = (
        PlacementBinding(tensor="values", domain="experts", owner_devices=(0, 1, 2, 3)),
        PlacementBinding(
            tensor="queries", domain="sequence", owner_devices=(0, 1, 2, 3)
        ),
    )
    return PlacementMap(mesh=mesh, bindings=bindings)


def test_route_classification_local_vs_remote() -> None:
    placement = _two_by_two_mesh_placement()
    assert placement.classify("values", 0, "queries", 0) is RouteLeg.LOCAL_MEMORY
    assert placement.classify("values", 1, "queries", 0) is RouteLeg.PEER_EXCHANGE


def test_plan_preserves_every_route_with_stable_ordering() -> None:
    placement = _two_by_two_mesh_placement()
    edges = [(q, (7 * q + k) % 16) for q in range(8) for k in range(2)]
    steps = plan_route_distribution(edges=edges, placement=placement, payload_bytes=4)
    # No duplicate or missing payloads: exchange payload counts sum exactly.
    exchange_steps = [step for step in steps if step.kind == "exchange"]
    sent = sum(ex.payload_count for step in exchange_steps for ex in step.exchanges)
    remote_edges = sum(
        1
        for q, s in edges
        if placement.classify("values", s, "queries", q) is RouteLeg.PEER_EXCHANGE
    )
    assert sent == remote_edges
    # Determinism
    again = plan_route_distribution(edges=edges, placement=placement, payload_bytes=4)
    assert steps == again
    # Grouping key matches destination device
    for step in exchange_steps:
        for ex in step.exchanges:
            assert ex.grouped_key == f"dst:{ex.dst_device}"


def test_capacity_policy_drop_is_explicit() -> None:
    placement = _two_by_two_mesh_placement()
    edges = [(0, 0), (0, 5)]
    steps = plan_route_distribution(
        edges=edges, placement=placement, capacity_drops=[0], payload_bytes=2
    )
    total = sum(
        ex.payload_count
        for step in steps
        if step.kind == "exchange"
        for ex in step.exchanges
    )
    local_only = all(step.kind != "exchange" for step in steps)
    # Edge 0 was dropped; edge 1 is remote (owner 1 != owner 0).
    assert total == 1 and not local_only


def test_transactional_routes_preserve_commit_boundaries() -> None:
    from urm.compiler.semantic import StateUpdate

    program = SemanticProgram.build(
        name="tx_routes",
        inputs=(
            TensorHandle("indices", DType.INT64, ("queries", "k")),
            TensorHandle("weights", DType.FLOAT32, ("queries", "k")),
            TensorHandle("values", DType.FLOAT32, ("sources", "d")),
            TensorHandle("delta", DType.FLOAT32, ("n", "d")),
        ),
        ops=(
            StateUpdate(
                name="merge",
                inputs=("delta",),
                outputs=("merged",),
                state="pages",
                policy=MergePolicy.SUM,
                commit_boundary=True,
            ),
        ),
        outputs=("merged",),
    )
    compiler = UrmCompiler()
    placement = _two_by_two_mesh_placement()
    result = compiler.compile(program, placement=placement)
    commits = [step for step in result.plan.steps if step.kind == "commit"]
    assert len(commits) == 1
    assert "version" in (commits[0].note or "")


# -- cost model --------------------------------------------------------------------


def test_cost_estimates_are_deterministic_and_flagged_analytical() -> None:
    fused = routed_reduction_cost(
        queries=1024, sources=256, route_width=8, value_dim=512
    )
    transform = row_scale_transform_cost(queries=1024, value_dim=512)
    total = combine(fused, transform)
    assert total.useful_flops == fused.useful_flops + transform.useful_flops
    assert total.logical_bytes == fused.logical_bytes + transform.logical_bytes
    assert total.to_dict()["provenance"] == "analytical_estimate"
    again = routed_reduction_cost(
        queries=1024, sources=256, route_width=8, value_dim=512
    )
    assert again == fused


def test_exchange_costs_report_wire_bytes_and_startup() -> None:
    estimate = exchange_cost(payloads=64, payload_bytes=2, hop_count=1)
    assert estimate.communication_bytes == 64 * 2 * 2
    assert estimate.collective_startup_us > 0
    assert estimate.critical_path_us > estimate.collective_startup_us


def test_device_limits_prefer_measured_artifact(tmp_path) -> None:
    fallback = DeviceLimits.load(tmp_path / "missing.json")
    assert not fallback.measured
    artifact = tmp_path / "device-limits.json"
    artifact.write_text(
        '{"bandwidth": {"sustainable_gbps": 513.3}, '
        '"fp32_cuda_core": {"fp32_cuda_core_tfps_measured": 23.1}}'
    )
    measured = DeviceLimits.load(artifact)
    assert measured.measured and measured.hbm_gbps == 513.3


def test_cost_estimate_to_dict_round_trip() -> None:
    estimate = CostEstimate(
        useful_flops=10,
        logical_bytes=20,
        physical_bytes_estimate=30,
        launch_count=1,
        temporary_bytes=5,
    )
    import json

    assert json.loads(json.dumps(estimate.to_dict()))["useful_flops"] == 10


# -- NAS-facing compilation matrix ---------------------------------------------------


@pytest.mark.parametrize(
    "route_width,sources,value_dim",
    [
        (1, 1, 1),  # fully degenerate
        (3, 5, 7),  # non-power-of-two everywhere
        (2, 64, 128),  # decode-like
        (32, 4096, 4096),  # prefill-like
    ],
)
def test_compilation_covers_degenerate_and_non_power_of_two_shapes(
    route_width, sources, value_dim
) -> None:
    program = row_scaled_routed_reduction_program(
        queries=16,
        route_width=route_width,
        sources=sources,
        value_dim=value_dim,
    )
    result = UrmCompiler().compile(program)
    assert result.plan.escape_hatch_count == 0
    assert len(result.trace.anchors) == 1
    assert "epilogue" in result.trace.anchors[0]


def test_schedule_and_architecture_params_serialize_separately() -> None:
    architecture = {"family": "top2_moe", "logical_dims": {"tokens": 4096}}
    schedule = ScheduleParams(block_hints={"BLOCK_D": 128})
    import json

    arch_json = json.dumps(architecture)
    sched_json = json.dumps(
        {
            "anchor_overrides": schedule.anchor_overrides,
            "block_hints": schedule.block_hints,
            "dtype_hints": schedule.dtype_hints,
        }
    )
    combined = json.loads(arch_json)
    combined["schedule"] = json.loads(sched_json)
    assert combined["family"] == "top2_moe"
    assert combined["schedule"]["block_hints"]["BLOCK_D"] == 128
    assert "anchor_overrides" not in arch_json


def test_executable_plan_serializes_without_escape_hatches() -> None:
    program = row_scaled_routed_reduction_program(
        queries=8, route_width=2, sources=16, value_dim=16
    )
    plan = UrmCompiler().compile(program).plan
    assert isinstance(plan, ExecutablePlan)
    import json

    payload = json.loads(json.dumps(plan.to_dict()))
    assert payload["escape_hatch_count"] == 0
    assert payload["steps"]
