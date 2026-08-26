"""CPU tests: compiler-to-kernel schedule integration closure.

Proves the normal compile() path produces a concrete, verified, serialized
schedule decision; that candidates and plans cannot disagree; that hints
change the launch configuration carried by the executable plan; that the
no-Z3 heuristic path passes the SAME verifier; and the bounded
nogood/retry behavior around an injectable compile probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urm.compiler import search as search_module
from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.kernel_plan import (
    build_schedule_model,
    decode_schedule_point,
    exhaustive_schedule_sweep,
    plan_kinds_for_candidate,
    schedule_point_to_assignment,
)
from urm.compiler.planner import (
    BASE_CANDIDATE_ID,
    CompilationIntent,
    ScheduleParams,
    UrmCompiler,
)
from urm.compiler.schedule_space import PlanKind, SchedulePoint
from urm.compiler.search import (
    CompilationSearch,
    CompileProbeResult,
    CompileStatus,
)
from urm.compiler.semantic import (
    SemanticProgram,
    row_scaled_routed_reduction_program,
)

FUSED_ID = "rewrite:fold_row_scale_into_routed_reduction_epilogue@apply_row_scale"


def _program(value_dim: int = 16) -> SemanticProgram:
    return row_scaled_routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=value_dim
    )


# -- compile() carries a concrete verified decision ------------------------------


def test_compile_returns_concrete_verified_schedule_decision() -> None:
    result = UrmCompiler().compile(_program(), intent=CompilationIntent.TRAINING)
    decision = result.schedule_decision
    assert decision is not None
    point = decision.schedule_point
    assert isinstance(point, SchedulePoint)
    assert point.plan == PlanKind.FUSED.value  # fused rewrite was selected
    assert point.block_d in (32, 64, 128, 256)
    # Every attempt passed the independent verifier BEFORE lowering.
    assert decision.attempts
    assert all(attempt.verified for attempt in decision.attempts)
    assert decision.verification_checks_run
    # No GPU probe was supplied: status must be honest, not a claimed success.
    assert decision.compile_status is CompileStatus.NOT_PROBED


def test_schedule_serialization_is_deterministic_and_solver_free() -> None:
    compiler = UrmCompiler()
    first = compiler.compile(_program()).to_dict()
    second = compiler.compile(_program()).to_dict()
    # Wall-clock solver statistics are timing, not schedule content; every
    # structural field must serialize identically.
    for payload in (first, second):
        payload["schedule_decision"].pop("solver_statistics")
        payload.pop("solver_statistics")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    text = json.dumps(first)
    # No raw solver objects may leak - mentioning the optional dependency in
    # a decline REASON is plain data; expressions/handles are not.
    for marker in ("BoolRef", "Z3PPObject", "AstRef", "z3.z3"):
        assert marker not in text
    # The executable plan carries the selected launch configuration.
    decision = first["schedule_decision"]
    steps = first["plan"]["steps"]
    configs = [s["launch_config"] for s in steps if s["launch_config"]]
    assert configs
    assert all(cfg == decision["launch_config"] for cfg in configs)


def test_schedule_point_assignment_round_trip() -> None:
    program = _program()
    fused_candidate = next(
        c for c in UrmCompiler().enumerate_candidates(program) if c.kind == "rewrite"
    )
    model = build_schedule_model(
        program=program,
        candidate=fused_candidate,
        intent=CompilationIntent.INFERENCE,
        schedule_params=ScheduleParams(),
        device_limits=UrmCompiler().device_limits,
    )
    _legal, ranked, _total = exhaustive_schedule_sweep(model)
    assert ranked
    point = decode_schedule_point(model, ranked[0][0])
    lifted = schedule_point_to_assignment(model, point)
    assert decode_schedule_point(model, lifted) == point


# -- candidate/plan binding --------------------------------------------------------


def test_candidate_and_plan_kind_cannot_disagree() -> None:
    compiler = UrmCompiler()
    program = _program()

    base_result = compiler.compile_candidate(program, BASE_CANDIDATE_ID)
    assert base_result.schedule_decision is None
    assert base_result.plan.steps[0].launch_config is None
    assert base_result.selected_candidate_id == BASE_CANDIDATE_ID

    fused_result = compiler.compile_candidate(
        program, FUSED_ID, intent=CompilationIntent.TRAINING
    )
    assert fused_result.schedule_decision is not None
    assert fused_result.schedule_decision.schedule_point.plan == PlanKind.FUSED.value
    assert fused_result.plan.steps[0].launch_config is not None

    fused_model = compiler.build_constraints(program, FUSED_ID)
    legal_fused, _ranked, _total = exhaustive_schedule_sweep(fused_model)
    assert legal_fused
    assert all(a["plan_fused"] is True for a in legal_fused)


def test_plan_binding_is_declared_per_candidate() -> None:
    candidates = UrmCompiler().enumerate_candidates(_program())
    by_id = {c.candidate_id: c for c in candidates}
    assert plan_kinds_for_candidate(by_id[BASE_CANDIDATE_ID]) == ()
    fused = by_id[FUSED_ID]
    assert plan_kinds_for_candidate(fused) == (PlanKind.FUSED,)
    assert (
        build_schedule_model(
            program=_program(),
            candidate=fused,
            intent=CompilationIntent.TRAINING,
            schedule_params=ScheduleParams(),
            device_limits=UrmCompiler().device_limits,
        ).metadata["allowed_plans"]
        == "fused"
    )


# -- hints reach the serialized launch configuration --------------------------------


@pytest.mark.parametrize(
    "hints",
    [
        ScheduleParams(block_hints={"BLOCK_D": 64}, warp_count=1),
        ScheduleParams(block_hints={"BLOCK_D": 32}, warp_count=2, stage_count=2),
        ScheduleParams(block_hints={"BLOCK_D": 128}, warp_count=4, stage_count=4),
    ],
)
def test_hints_change_launch_configuration_in_executable_plan(hints) -> None:
    result = UrmCompiler().compile_candidate(
        _program(32),
        FUSED_ID,
        intent=CompilationIntent.TRAINING,
        schedule_params=hints,
    )
    decision = result.schedule_decision
    assert decision is not None
    point = decision.schedule_point
    assert point.block_d == hints.block_hints["BLOCK_D"]
    assert point.num_warps == hints.warp_count
    if hints.stage_count is not None:
        assert point.num_stages == hints.stage_count
    payload = result.plan.to_dict()["steps"]
    configs = [step["launch_config"] for step in payload if step["launch_config"]]
    assert configs
    assert {cfg["block_d"] for cfg in configs} == {point.block_d}
    assert {cfg["num_warps"] for cfg in configs} == {point.num_warps}


# -- fallback and verification guards -----------------------------------------------


def test_no_z3_heuristic_path_passes_the_same_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("urm.compiler.solver.z3_available", lambda: False)
    result = UrmCompiler().compile(_program(), intent=CompilationIntent.TRAINING)
    decision = result.schedule_decision
    assert decision is not None
    assert decision.selection_policy == "cost_heuristic"
    assert decision.fallback_used is True
    # Identical independent verification ran on the fallback assignment.
    assert decision.attempts
    assert all(a.verified for a in decision.attempts)
    assert decision.verification_checks_run


def test_unverified_assignment_never_reaches_lowering_or_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urm.compiler.search import CompileContext
    from urm.compiler.verification import VerificationFailure, VerificationReport

    probe_calls: list[CompileContext] = []

    def probe(ctx: CompileContext) -> CompileProbeResult:
        probe_calls.append(ctx)
        return CompileProbeResult(ok=True)

    def rejecting_verifier(_model, _assignment) -> VerificationReport:
        return VerificationReport(
            ok=False,
            failures=(VerificationFailure(check="synthetic", message="rejected"),),
            checks_run=("synthetic",),
        )

    compiler = UrmCompiler(
        compile_probe=probe,
        max_nogoods=1,
        schedule_verifier=rejecting_verifier,
    )
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile(_program())
    codes = {d.code for d in excinfo.value.diagnostics}
    assert DiagnosticCode.NOGOOD_RETRY_LIMIT in codes
    # The probe never saw an unverified assignment.
    assert probe_calls == []


# -- compile-feedback retry loop ------------------------------------------------------


def _first_failure_probe(calls: list[object]):
    def probe(ctx: object) -> CompileProbeResult:
        calls.append(ctx)
        if len(calls) == 1:
            return CompileProbeResult(ok=False, reason="synthetic compile failure")
        return CompileProbeResult(ok=True, registers_per_thread=24)

    return probe


def test_compile_probe_failure_adds_exact_nogood_then_recovers() -> None:
    calls: list[object] = []
    program = _program()
    compiler = UrmCompiler(compile_probe=_first_failure_probe(calls), max_nogoods=4)
    result = compiler.compile(program, intent=CompilationIntent.TRAINING)
    decision = result.schedule_decision
    assert decision is not None
    assert len(decision.attempts) == 2
    point0 = getattr(calls[0], "schedule_point", calls[0])
    point1 = getattr(calls[1], "schedule_point", calls[1])
    assert point0.as_dict() != point1.as_dict()
    assert decision.compile_status is CompileStatus.SUCCEEDED
    # Counters mean exactly what they say.
    assert decision.compile_failures_observed == 1
    assert decision.nogoods_added == 1
    assert decision.recoveries == 1
    assert decision.retry_budget_exhausted is False

    # The nogood excludes EXACTLY the failed attempt's own assignment.
    reference_model = compiler.build_constraints(
        program, FUSED_ID, intent=CompilationIntent.TRAINING
    )
    expected_failed = schedule_point_to_assignment(reference_model, point0)
    forbidden = decision.attempts[0].nogood_forbidden
    assert forbidden is not None
    assert forbidden == {
        name: expected_failed[name] for name in sorted(expected_failed)
    }
    # ... and it does not exclude the point that ultimately succeeded.
    second_point_assignment = schedule_point_to_assignment(
        reference_model, decision.schedule_point
    )
    fires_on_success = all(
        second_point_assignment[name] == value for name, value in forbidden.items()
    )
    assert not fires_on_success
    assert decision.attempts[1].nogood_forbidden is None


def test_compile_probe_type_error_fails_cleanly_without_legacy_fallback() -> None:
    """A TypeError raised inside the probe fails closed and records an exact nogood."""
    call_count = 0

    def buggy_probe(ctx) -> CompileProbeResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate an internal TypeError (e.g. bug inside the probe implementation)
            raise TypeError(
                "unexpected NoneType in internal tensor dimension calculation"
            )
        return CompileProbeResult(ok=True, registers_per_thread=32)

    program = _program()
    compiler = UrmCompiler(compile_probe=buggy_probe, max_nogoods=4)
    result = compiler.compile(program, intent=CompilationIntent.TRAINING)
    decision = result.schedule_decision
    assert decision is not None
    # Called exactly once per assignment attempt: attempt 0 failed, attempt 1 succeeded
    assert call_count == 2
    assert len(decision.attempts) == 2
    attempt0 = decision.attempts[0]
    assert attempt0.compile_status is CompileStatus.FAILED
    assert "probe raised TypeError" in str(attempt0.compile_detail)
    assert attempt0.nogood_added is True
    assert attempt0.nogood_forbidden is not None
    assert decision.compile_status is CompileStatus.SUCCEEDED
    assert decision.compile_failures_observed == 1
    assert decision.nogoods_added == 1
    assert decision.recoveries == 1


def test_retry_exhaustion_returns_structured_diagnostic() -> None:
    def always_fails(_point: object) -> CompileProbeResult:
        return CompileProbeResult(ok=False, reason="hard resource limit")

    compiler = UrmCompiler(compile_probe=always_fails, max_nogoods=2)
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile(_program(), intent=CompilationIntent.TRAINING)
    codes = {d.code for d in excinfo.value.diagnostics}
    assert DiagnosticCode.NOGOOD_RETRY_LIMIT in codes
    assert "retry budget" in excinfo.value.diagnostics[0].message


# -- deterministic training is now honestly UNSAT through compile() ------------------


def test_deterministic_training_compilation_is_structurally_unsat() -> None:
    """No legal deterministic training schedule exists; inference stays fine."""
    compiler = UrmCompiler()
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile(
            _program(),
            intent=CompilationIntent.TRAINING,
            schedule_params=ScheduleParams(deterministic=True),
        )
    codes = {d.code for d in excinfo.value.diagnostics}
    assert (
        DiagnosticCode.UNSAT_CONSTRAINTS in codes
        or DiagnosticCode.INTENT_CONFLICT in codes
    )
    # Deterministic INFERENCE keeps compiling.
    result = compiler.compile(
        _program(), schedule_params=ScheduleParams(deterministic=True)
    )
    assert result.schedule_decision is not None
    assert all(a.verified for a in result.schedule_decision.attempts)


# -- direct search component behavior ---------------------------------------------------


def test_search_component_runs_standalone() -> None:
    program = _program()
    compiler = UrmCompiler()
    model = compiler.build_constraints(
        program, FUSED_ID, intent=CompilationIntent.TRAINING
    )
    decision = CompilationSearch(model=model, max_nogoods=8).run()
    assert decision.selection_policy in {"solver_guided", "cost_heuristic"}
    assert decision.model_hash == model.summary_hash()
    assert decision.to_dict()["schedule"] == decision.schedule_point.as_dict()


def test_launch_config_covers_every_solver_selectable_dimension() -> None:
    from urm.compiler.search import launch_config_of

    point = SchedulePoint(
        plan="fused",
        block_d=64,
        num_warps=2,
        num_stages=2,
        grad_values_decomposition="per_route",
        grad_values_schedule="segmented",
        dtype="float16",
    )
    config = launch_config_of(point)
    assert set(config) == set(point.as_dict())


def test_compile_probe_receives_actual_specialization_context() -> None:
    received_contexts = []

    def probe(ctx) -> CompileProbeResult:
        received_contexts.append(ctx)
        return CompileProbeResult(
            ok=True, registers_per_thread=32, shared_mem_bytes=2048
        )

    program = _program(value_dim=48)
    compiler = UrmCompiler(compile_probe=probe)
    result = compiler.compile(program, intent=CompilationIntent.TRAINING)
    assert received_contexts
    ctx = received_contexts[0]
    assert ctx.anchor_name == "routed_reduction_row_scale_epilogue_v0"
    assert ctx.plan == "fused"
    assert ctx.intent == "training"
    assert ctx.queries == 8
    assert ctx.route_width == 2
    assert ctx.sources == 8
    assert ctx.value_dim == 48
    assert ctx.dtype in ("float32", "float16", "bfloat16")
    assert ctx.schedule_point == result.schedule_decision.schedule_point


def test_injected_resource_results_survive_decision_and_serialization() -> None:
    from urm.compiler.search import KernelResourceUsage

    def probe(_ctx) -> CompileProbeResult:
        return CompileProbeResult(
            ok=True,
            registers_per_thread=28,
            shared_mem_bytes=1024,
            kernel_resources={
                "forward": KernelResourceUsage("fwd_kernel", 24, 512),
                "grad_values": KernelResourceUsage("gv_kernel", 28, 1024),
            },
        )

    compiler = UrmCompiler(compile_probe=probe)
    result = compiler.compile(_program(), intent=CompilationIntent.TRAINING)
    decision = result.schedule_decision
    assert decision is not None
    assert decision.registers_per_thread == 28
    assert decision.shared_mem_bytes == 1024
    assert decision.kernel_resources is not None
    assert decision.kernel_resources["forward"]["registers_per_thread"] == 24
    assert decision.kernel_resources["grad_values"]["shared_mem_bytes"] == 1024

    serialized = result.to_dict()
    assert serialized["schedule_decision"]["registers_per_thread"] == 28
    assert serialized["schedule_decision"]["shared_mem_bytes"] == 1024
    assert (
        serialized["schedule_decision"]["kernel_resources"]["grad_values"][
            "registers_per_thread"
        ]
        == 28
    )

    # JSON roundtrip
    roundtripped = json.loads(json.dumps(serialized))
    assert roundtripped["schedule_decision"]["registers_per_thread"] == 28
    assert roundtripped["schedule_decision"]["shared_mem_bytes"] == 1024


def test_per_route_full_row_is_absent_from_model_and_reference_legal_sets() -> None:
    from urm.compiler.schedule_space import (
        ScheduleProblem,
        is_legal,
        legal_schedules,
    )

    problem = ScheduleProblem()
    bad_point = SchedulePoint(
        plan="fused",
        block_d=32,
        num_warps=2,
        num_stages=1,
        grad_values_decomposition="per_route",
        grad_values_schedule="full_row",
        dtype="float32",
    )
    assert not is_legal(bad_point, problem)
    assert bad_point not in legal_schedules(problem)

    # In model
    compiler = UrmCompiler()
    model = compiler.build_constraints(_program(), FUSED_ID)
    legal, _ranked, _total = exhaustive_schedule_sweep(model)
    decoded = [decode_schedule_point(model, a) for a in legal]
    assert all(
        not (
            p.grad_values_decomposition == "per_route"
            and p.grad_values_schedule == "full_row"
        )
        for p in decoded
    )


def test_model_and_reference_legality_compares_exact_point_sets() -> None:
    from urm.compiler.schedule_space import (
        ScheduleProblem,
        legal_schedules,
    )

    compiler = UrmCompiler()
    program = _program()
    model = compiler.build_constraints(program, FUSED_ID)
    legal_assignments, _ranked, _total = exhaustive_schedule_sweep(model)

    meta = model.metadata
    problem = ScheduleProblem(
        queries=int(meta["queries"]),
        sources=int(meta["sources"]),
        route_width=int(meta["route_width"]),
        value_dim=int(meta["value_dim"]),
        dtypes=tuple(meta["dtypes"].split(",")),
        training=meta["training"] == "1",
        deterministic=meta["deterministic"] == "1",
        fused_anchor_available=True,
    )
    ref_legal = [p for p in legal_schedules(problem) if p.plan == "fused"]
    ref_keys = {p.stable_key for p in ref_legal}
    model_keys = {decode_schedule_point(model, a).stable_key for a in legal_assignments}
    assert ref_keys == model_keys


def test_search_module_stays_cpu_safe() -> None:
    """search.py must not eagerly import torch/triton (GPU stays injectable)."""
    source_file = search_module.__file__
    assert source_file is not None
    text = Path(source_file).read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "import triton" not in text
    assert "urm.compiler.anchors" not in text
