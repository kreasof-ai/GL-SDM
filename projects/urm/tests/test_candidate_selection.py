"""Candidate selection, ScheduleParams, and intent restrictions (CPU-only)."""

from __future__ import annotations

import pytest

from urm.compiler.constraints import ConstraintModel
from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.planner import (
    BASE_CANDIDATE_ID,
    CompilationIntent,
    ScheduleParams,
    SelectionPolicy,
    UrmCompiler,
    validate_schedule_params,
)
from urm.compiler.semantic import (
    DType,
    SemanticProgram,
    Transform,
    row_scaled_routed_reduction_program,
)

FUSED_ID = "rewrite:fold_row_scale_into_routed_reduction_epilogue@apply_row_scale"


def _program() -> SemanticProgram:
    return row_scaled_routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )


# -- candidate enumeration -------------------------------------------------------


def test_base_plan_is_always_the_first_candidate() -> None:
    candidates = UrmCompiler().enumerate_candidates(_program())
    assert candidates[0].candidate_id == BASE_CANDIDATE_ID
    assert candidates[0].kind == "base"
    assert any(candidate.kind == "rewrite" for candidate in candidates)


def test_enumeration_does_not_mutate_the_program() -> None:
    program = _program()
    before = program.ops
    compiler = UrmCompiler()
    first = compiler.enumerate_candidates(program)
    second = compiler.enumerate_candidates(program)
    assert program.ops == before  # identical objects: nothing mutated
    assert first == second


def test_candidate_ids_are_stable() -> None:
    ids_a = [c.candidate_id for c in UrmCompiler().enumerate_candidates(_program())]
    ids_b = [c.candidate_id for c in UrmCompiler().enumerate_candidates(_program())]
    assert ids_a == ids_b
    assert FUSED_ID in ids_a


def test_explicit_candidate_selection_compiles_that_candidate() -> None:
    compiler = UrmCompiler()
    result = compiler.compile_candidate(_program(), FUSED_ID)
    assert result.selected_candidate_id == FUSED_ID
    assert result.selection_policy is SelectionPolicy.EXPLICIT
    assert "epilogue" in result.trace.anchors[0]
    rejected_ids = {r.candidate_id for r in result.rejected_alternatives}
    assert BASE_CANDIDATE_ID in rejected_ids


def test_unknown_and_illegal_candidates_are_structured_errors() -> None:
    compiler = UrmCompiler()
    with pytest.raises(CompilerError) as unknown:
        compiler.compile_candidate(_program(), "rewrite:nope@op")
    assert unknown.value.diagnostics[0].code is DiagnosticCode.CANDIDATE_NOT_FOUND


def test_automatic_selection_records_policy_and_alternatives() -> None:
    compiler = UrmCompiler()
    result = compiler.compile(_program())
    assert result.selection_policy in {
        SelectionPolicy.SOLVER_GUIDED,
        SelectionPolicy.COST_HEURISTIC,
    }
    assert result.selected_candidate_id == FUSED_ID or result.selected_candidate_id == (
        BASE_CANDIDATE_ID
    )
    # The trace records the selected candidate and rejected alternatives.
    payload = result.to_dict()
    assert payload["selected_candidate_id"] == result.selected_candidate_id
    assert isinstance(payload["rejected_alternatives"], list)


# -- training vs inference restrictions ---------------------------------------------


def test_training_accepts_certified_backward_rewrite() -> None:
    compiler = UrmCompiler()
    candidates = compiler.enumerate_candidates(_program(), CompilationIntent.TRAINING)
    fused = next(c for c in candidates if c.candidate_id == FUSED_ID)
    assert fused.legal  # backward certified for fp32/fp16/bf16
    result = compiler.compile_candidate(
        _program(), FUSED_ID, intent=CompilationIntent.TRAINING
    )
    kinds = {obligation[0] for obligation in result.plan.obligations}
    assert "recompute_backward" in kinds
    assert result.unresolved_obligations == ()  # resolved by the epilogue anchor


def test_forward_only_rewrites_are_rejected_for_training() -> None:
    from urm.compiler.rewrite import (
        EquivalenceClass,
        ForwardOnlyRestriction,
        RewriteRule,
    )

    forward_only_rule = RewriteRule(
        name="forward_only_placeholder",
        description="a rule with no certified backward",
        subject_kind=Transform,
        producer_kind=None,
        matcher=lambda program, match: True,
        preconditions=(),
        equivalence=EquivalenceClass.FLOATING_POINT,
        tolerance_envelope=None,
        forward_mapping=lambda program, match: (),
        backward_contract=None,
        forward_only_restriction=ForwardOnlyRestriction.BACKWARD_UNVERIFIED,
    )
    compiler = UrmCompiler(rules=[forward_only_rule])
    candidates = compiler.enumerate_candidates(_program(), CompilationIntent.TRAINING)
    offending = [
        c for c in candidates if c.kind == "rewrite" and not c.backward_verified
    ]
    assert offending
    assert all(c.reason_code == DiagnosticCode.INTENT_CONFLICT.value for c in offending)
    with pytest.raises(CompilerError, match="intent_conflict|not legal"):
        compiler.compile_candidate(
            _program(),
            offending[0].candidate_id,
            intent=CompilationIntent.TRAINING,
        )


def test_training_never_accepts_unresolved_obligations() -> None:
    """A training compile of the base plan must still lower differentiably."""
    compiler = UrmCompiler()
    # The materialized base plan lowers through v1 + torch autograd anchors;
    # both certify backward, so training succeeds with zero unresolved duties.
    result = compiler.compile(_program(), intent=CompilationIntent.TRAINING)
    assert result.unresolved_obligations == ()
    for anchor_name in result.trace.anchors:
        assert "reserved" not in anchor_name


def test_inference_and_forward_only_analysis_allow_more() -> None:
    compiler = UrmCompiler()
    for intent in (
        CompilationIntent.INFERENCE,
        CompilationIntent.FORWARD_ONLY_ANALYSIS,
    ):
        result = compiler.compile(_program(), intent=intent)
        assert result.plan.steps


# -- ScheduleParams -------------------------------------------------------------------


def test_invalid_schedule_params_produce_structured_diagnostics() -> None:
    bad = ScheduleParams(
        block_hints={"BLOCK_D": -4},
        warp_count=3,
        stage_count=7,
        dtype_hints={"x": "float9"},
        layout_hints={"w": "zigzag"},
    )
    diagnostics = validate_schedule_params(bad)
    codes = {d.code for d in diagnostics}
    assert DiagnosticCode.SCHEDULE_HINT_INVALID in codes
    with pytest.raises(CompilerError):
        UrmCompiler().compile(_program(), schedule_params=bad)


def test_schedule_params_are_validated_and_applied_at_compile() -> None:
    compiler = UrmCompiler()
    params = ScheduleParams(anchor_overrides={"reduce": "routed_reduction_v1"})
    # Explicit base candidate: no epilogue visitor, so the v1 override holds.
    result = compiler.compile_candidate(
        _program(), BASE_CANDIDATE_ID, schedule_params=params
    )
    assert any(step.anchor == "routed_reduction_v1" for step in result.plan.steps)
    assert result.schedule_params["anchor_overrides"] == {
        "reduce": "routed_reduction_v1"
    }


def test_anchor_override_cannot_break_visitor_requirements() -> None:
    """Overriding the fused plan's anchor to v1 is a structured decline."""
    compiler = UrmCompiler()
    params = ScheduleParams(anchor_overrides={"reduce": "routed_reduction_v1"})
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile_candidate(_program(), FUSED_ID, schedule_params=params)
    assert excinfo.value.diagnostics[0].code in {
        DiagnosticCode.NO_ANCHOR_AVAILABLE,
        DiagnosticCode.ANCHOR_DECLINED,
    }
    assert "declined visitors" in excinfo.value.diagnostics[0].message


def test_anchor_override_to_unknown_anchor_is_declined() -> None:
    compiler = UrmCompiler()
    params = ScheduleParams(anchor_overrides={"*": "not_an_anchor"})
    with pytest.raises(CompilerError):
        compiler.compile(_program(), schedule_params=params)


def test_override_that_cannot_honor_visitor_is_declined() -> None:
    program = _program()
    compiler = UrmCompiler()
    # Force the fused rewrite then override its reduction anchor with v1,
    # which lacks the FINAL_SCALE_CONVERT visitor: structured decline.
    params = ScheduleParams(anchor_overrides={"reduce": "routed_reduction_v1"})
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile_candidate(program, FUSED_ID, schedule_params=params)
    codes = {d.code for d in excinfo.value.diagnostics}
    assert (
        DiagnosticCode.ANCHOR_DECLINED in codes
        or DiagnosticCode.NO_ANCHOR_AVAILABLE in codes
    )


def test_training_rejects_anchor_without_verified_dtype_backward() -> None:
    """Training dispatch requires the anchor's backward to cover the dtype."""
    from urm.compiler.execution import (
        AnchorKind,
        AnchorRegistry,
        ExecutionAnchor,
        make_selector,
    )

    limited = ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="routed_reduction_fp32_only",
        backward_verified_dtypes=frozenset({"float32"}),
    )
    registry = AnchorRegistry()
    registry.register(make_selector([limited]))
    compiler = UrmCompiler(anchors=registry)
    program = row_scaled_routed_reduction_program(
        queries=8,
        route_width=2,
        sources=8,
        value_dim=16,
        value_dtype=DType.BFLOAT16,
    )
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile_candidate(
            program, BASE_CANDIDATE_ID, intent=CompilationIntent.TRAINING
        )
    assert excinfo.value.diagnostics[0].code is DiagnosticCode.INTENT_CONFLICT


def test_deterministic_flag_flows_into_metadata() -> None:
    compiler = UrmCompiler()
    result = compiler.compile(
        _program(),
        schedule_params=ScheduleParams(deterministic=True),
    )
    assert result.schedule_params["deterministic"] is True


def test_schedule_params_change_selected_plans() -> None:
    """Acceptance gate: hints demonstrably alter or constrain plans."""
    from urm.compiler.kernel_plan import exhaustive_schedule_sweep

    compiler = UrmCompiler()
    baseline = compiler.build_constraints(_program(), FUSED_ID)
    pinned = compiler.build_constraints(
        _program(),
        FUSED_ID,
        schedule_params=ScheduleParams(block_hints={"BLOCK_D": 256}, warp_count=8),
    )
    base_legal, _base_ranked, _ = exhaustive_schedule_sweep(baseline)
    pinned_legal, _pinned_ranked, _ = exhaustive_schedule_sweep(pinned)
    assert base_legal, "unconstrained space must be satisfiable"
    assert pinned_legal, "hinted space must remain satisfiable"
    assert all(a["block_d"] == 256 for a in pinned_legal)
    assert all(a["num_warps"] == 8 for a in pinned_legal)
    assert any(a["block_d"] != 256 for a in base_legal)
    assert baseline.summary_hash() != pinned.summary_hash()


def test_build_constraints_rejects_unknown_candidate() -> None:
    compiler = UrmCompiler()
    with pytest.raises(CompilerError) as excinfo:
        compiler.build_constraints(_program(), "nope")
    assert excinfo.value.diagnostics[0].code is DiagnosticCode.CANDIDATE_NOT_FOUND


def test_fused_anchor_override_cannot_produce_base_schedule_or_omit_fused_semantic_inputs() -> (
    None
):
    """A fused anchor override cannot lower a base candidate lacking fused inputs."""
    compiler = UrmCompiler()
    params = ScheduleParams(
        anchor_overrides={"*": "routed_reduction_row_scale_epilogue_v0"}
    )
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile_candidate(
            _program(), BASE_CANDIDATE_ID, schedule_params=params
        )
    codes = {d.code for d in excinfo.value.diagnostics}
    assert (
        DiagnosticCode.ANCHOR_DECLINED in codes
        or DiagnosticCode.NO_ANCHOR_AVAILABLE in codes
    )


def test_incompatible_or_unknown_override_fails_explicitly() -> None:
    compiler = UrmCompiler()
    # Unknown anchor name in override
    bad_anchor = ScheduleParams(anchor_overrides={"*": "totally_unknown_anchor"})
    with pytest.raises(CompilerError) as excinfo1:
        compiler.compile(_program(), schedule_params=bad_anchor)
    assert excinfo1.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID

    # Unknown operation name in override
    bad_op = ScheduleParams(anchor_overrides={"nonexistent_op": "routed_reduction_v1"})
    with pytest.raises(CompilerError) as excinfo2:
        compiler.compile(_program(), schedule_params=bad_op)
    assert excinfo2.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID


def test_unknown_or_unused_hints_are_rejected() -> None:
    # Unknown block hint key
    diag1 = validate_schedule_params(ScheduleParams(block_hints={"BLOCK_M": 64}))
    assert any(d.code is DiagnosticCode.SCHEDULE_HINT_INVALID for d in diag1)

    # Invalid block size
    diag2 = validate_schedule_params(ScheduleParams(block_hints={"BLOCK_D": 50}))
    assert any(d.code is DiagnosticCode.SCHEDULE_HINT_INVALID for d in diag2)

    # Unsupported dtype hints
    diag3 = validate_schedule_params(ScheduleParams(dtype_hints={"x": "float32"}))
    assert any(d.code is DiagnosticCode.SCHEDULE_HINT_INVALID for d in diag3)

    # Unsupported layout hints
    diag4 = validate_schedule_params(ScheduleParams(layout_hints={"x": "row_major"}))
    assert any(d.code is DiagnosticCode.SCHEDULE_HINT_INVALID for d in diag4)


def test_stage_count_3_is_rejected_as_invalid_hint() -> None:
    diag = validate_schedule_params(ScheduleParams(stage_count=3))
    assert any(d.code is DiagnosticCode.SCHEDULE_HINT_INVALID for d in diag)
    assert any("stage_count=3" in d.message for d in diag)


def test_base_v1_does_not_report_launch_parameters_that_it_ignores() -> None:
    compiler = UrmCompiler()
    result = compiler.compile_candidate(_program(), BASE_CANDIDATE_ID)
    assert result.schedule_decision is None
    for step in result.plan.steps:
        if step.anchor == "routed_reduction_v1":
            assert step.launch_config is None


def test_capability_registry_fails_closed_for_unknown_rewrite_bindings() -> None:
    from urm.compiler.kernel_plan import plan_kinds_for_candidate
    from urm.compiler.planner import CompilationCandidate

    dummy = CompilationCandidate(
        candidate_id="rewrite:dummy_unknown@op",
        kind="rewrite",
        rule="dummy_unknown_rule",
    )
    with pytest.raises(ValueError, match="unregistered rewrite rule"):
        plan_kinds_for_candidate(dummy)


def test_constraint_models_are_backend_independent() -> None:
    """No raw solver expressions may appear in the model summary."""
    compiler = UrmCompiler()
    model = compiler.build_constraints(_program(), FUSED_ID)
    import json

    text = json.dumps(model.to_summary())
    assert "z3" not in text.lower()
    assert isinstance(model, ConstraintModel)


# -- Regression tests for override and hint truthfulness -------------------------------


def test_valid_explicit_fused_override_compiles_successfully() -> None:
    """Explicit override to the fused anchor produces fused schedule and launch config."""
    program = _program()
    compiler = UrmCompiler()
    params = ScheduleParams(
        anchor_overrides={"reduce": "routed_reduction_row_scale_epilogue_v0"}
    )
    result = compiler.compile(program, schedule_params=params)
    assert result.plan.steps[0].anchor == "routed_reduction_row_scale_epilogue_v0"
    assert result.plan.steps[0].launch_config is not None
    assert result.schedule_decision is not None
    assert result.schedule_decision.schedule_point.plan == "fused"


def test_explicit_override_with_missing_semantic_inputs_declines_cleanly() -> None:
    """Overriding to fused anchor when row scale input is absent fails with structured error."""
    from urm.compiler.semantic import routed_reduction_program

    program = routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )
    compiler = UrmCompiler()
    params = ScheduleParams(
        anchor_overrides={"reduce": "routed_reduction_row_scale_epilogue_v0"}
    )
    with pytest.raises(CompilerError) as excinfo:
        compiler.compile(program, schedule_params=params)
    codes = {d.code for d in excinfo.value.diagnostics}
    assert (
        DiagnosticCode.NO_ANCHOR_AVAILABLE in codes
        or DiagnosticCode.ANCHOR_DECLINED in codes
    )


def test_unconsumed_explicit_overrides_are_rejected() -> None:
    """Every unconsumed explicit override is rejected with structured diagnostic."""
    program = _program()
    compiler = UrmCompiler()

    # 1. Known but rewritten-away operation (apply_row_scale is fused into reduce)
    with pytest.raises(CompilerError) as exc1:
        compiler.compile(
            program,
            schedule_params=ScheduleParams(
                anchor_overrides={"apply_row_scale": "routed_reduction_v1"}
            ),
        )
    assert exc1.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID
    assert "rewritten or fused" in exc1.value.diagnostics[0].message

    # 2. Known interior gather (fused into routed-reduction dispatch)
    from urm.compiler.semantic import routed_reduction_program

    gather_prog = routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )
    with pytest.raises(CompilerError) as exc2:
        compiler.compile(
            gather_prog,
            schedule_params=ScheduleParams(
                anchor_overrides={"gather": "routed_reduction_v1"}
            ),
        )
    assert exc2.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID
    assert "interior gather" in exc2.value.diagnostics[0].message

    # 3. Known non-anchorable operation (Transform without anchor site)
    from urm.compiler.semantic import TransformKind

    base = routed_reduction_program(queries=8, route_width=2, sources=8, value_dim=16)
    nonlin_prog = SemanticProgram.build(
        name="nonlin_prog",
        inputs=base.inputs,
        ops=(
            *base.ops,
            Transform(
                name="gelu_transform",
                inputs=("base",),
                outputs=("out",),
                kind=TransformKind.GELU,
            ),
        ),
        outputs=("out",),
    )
    with pytest.raises(CompilerError) as exc3:
        compiler.compile(
            nonlin_prog,
            schedule_params=ScheduleParams(
                anchor_overrides={"gelu_transform": "routed_reduction_v1"}
            ),
        )
    assert exc3.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID
    assert "no anchor-dispatch site" in exc3.value.diagnostics[0].message

    # 4. Incompatible anchor kind (torch_linear is a GEMM anchor, not ROUTED_REDUCTION)
    with pytest.raises(CompilerError) as exc4:
        compiler.compile(
            program,
            schedule_params=ScheduleParams(anchor_overrides={"reduce": "torch_linear"}),
        )
    codes4 = {d.code for d in exc4.value.diagnostics}
    assert (
        DiagnosticCode.NO_ANCHOR_AVAILABLE in codes4
        or DiagnosticCode.ANCHOR_DECLINED in codes4
    )

    # 5. Wildcard override is NOT treated as an unconsumed explicit key
    res_wildcard = compiler.compile(
        program,
        schedule_params=ScheduleParams(
            anchor_overrides={"*": "routed_reduction_row_scale_epilogue_v0"}
        ),
    )
    assert res_wildcard.plan.steps[0].anchor == "routed_reduction_row_scale_epilogue_v0"


def test_schedule_knobs_rejected_for_unscheduled_lowerings() -> None:
    """Providing block, warp, or stage hints when lowering unscheduled base raises error."""
    from urm.compiler.semantic import routed_reduction_program

    program = routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )
    compiler = UrmCompiler()

    # Base lowering with block hint
    with pytest.raises(CompilerError) as exc1:
        compiler.compile(
            program, schedule_params=ScheduleParams(block_hints={"BLOCK_D": 256})
        )
    assert exc1.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID
    assert (
        "unscheduled and does not consume launch configurations"
        in exc1.value.diagnostics[0].message
    )

    # Base lowering with warp count
    with pytest.raises(CompilerError) as exc2:
        compiler.compile(program, schedule_params=ScheduleParams(warp_count=8))
    assert exc2.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID

    # Base lowering with stage count
    with pytest.raises(CompilerError) as exc3:
        compiler.compile(program, schedule_params=ScheduleParams(stage_count=4))
    assert exc3.value.diagnostics[0].code is DiagnosticCode.SCHEDULE_HINT_INVALID


def test_anchor_capability_is_schedule_domain_source_of_truth() -> None:
    """Changing an anchor's declared capabilities alters the generated constraint model."""
    from urm.compiler.execution import AnchorKind, ExecutionAnchor
    from urm.compiler.kernel_plan import exhaustive_schedule_sweep

    # Custom anchor with constrained warps=(4,) and blocks=(128,)
    custom_anchor = ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="custom_test_anchor",
        schedulable=True,
        consumes_launch_config=True,
        supported_plan_kinds=frozenset({"fused"}),
        supported_blocks=(128,),
        supported_warps=(4,),
        supported_stages=(1, 2),
        supported_decompositions=("per_query",),
        supported_schedules=("segmented",),
    )

    compiler = UrmCompiler()
    program = _program()
    model = compiler.build_constraints(program, FUSED_ID, anchor=custom_anchor)
    legal, _, _ = exhaustive_schedule_sweep(model)
    assert legal, "constrained space should be satisfiable"
    assert all(a["block_d"] == 128 for a in legal)
    assert all(a["num_warps"] == 4 for a in legal)
    assert all(a.get("decomp_per_query") is True for a in legal)
    assert all(a.get("gradsched_segmented") is True for a in legal)
