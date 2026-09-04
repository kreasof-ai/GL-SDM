"""Torch-independent SDM semantic, intent, and anchor-selection gates."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.execution import (
    SDM_EXTERNAL_ANCHOR_NAME,
    TRUSTED_ANCHORS,
    AnchorKind,
    AnchorRegistry,
    ExecutionAnchor,
    make_sdm_selector,
)
from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
from urm.compiler.semantic import (
    DType,
    MergePolicy,
    SDMExecutionMode,
    SparseDeltaMemoryAccess,
    sparse_delta_memory_program,
)

CANONICAL_ANCHOR = next(
    anchor for anchor in TRUSTED_ANCHORS if anchor.name == SDM_EXTERNAL_ANCHOR_NAME
)


def _compiler(
    *,
    anchor: ExecutionAnchor = CANONICAL_ANCHOR,
    supported: bool = True,
    code: str = "supported",
    reason: str | None = None,
) -> UrmCompiler:
    registry = AnchorRegistry()
    status = SimpleNamespace(supported=supported, code=code, reason=reason)
    registry.register(make_sdm_selector(anchor, lambda: status))
    return UrmCompiler(anchors=registry)


def _program(
    mode: SDMExecutionMode = SDMExecutionMode.INFERENCE,
    dtype: DType = DType.BFLOAT16,
):
    return sparse_delta_memory_program(
        sequence=16,
        slots_per_partition=256,
        value_dim=32,
        writes=4,
        reads=4,
        dtype=dtype,
        mode=mode,
    )


def _assert_error(call, code: DiagnosticCode, subject: str) -> CompilerError:
    with pytest.raises(CompilerError) as caught:
        call()
    assert caught.value.diagnostics[0].code is code
    assert caught.value.diagnostics[0].subject == subject
    return caught.value


def test_typed_semantic_program_is_not_routed_reduction() -> None:
    program = sparse_delta_memory_program(
        parallel=2,
        sequence=7,
        slots_per_partition=256,
        value_dim=37,
        writes=4,
        reads=3,
    )
    assert len(program.ops) == 1
    op = program.ops[0]
    assert isinstance(op, SparseDeltaMemoryAccess)
    assert op.spec.value_dim == 37
    assert op.spec.collision_policy is MergePolicy.ORDERED
    assert op.spec.within_token_collision_policy is MergePolicy.REJECT
    assert program.outputs[-4:] == (
        "write_addresses",
        "write_weights",
        "read_addresses",
        "read_weights",
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"slots_per_partition": 255}, "perfect square"),
        ({"slots_per_partition": 81}, "divisible by 8"),
        ({"writes": 129}, "exceed"),
        ({"dtype": DType.FLOAT16}, "float32 and bfloat16"),
        (
            {"mode": SDMExecutionMode.TRAINING, "sequence": 1},
            "sequence >= 16",
        ),
    ],
)
def test_semantic_program_rejects_unsupported_modes(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        sparse_delta_memory_program(**kwargs)


def test_training_semantics_reject_inference_intent_before_runtime_selection() -> None:
    error = _assert_error(
        lambda: _compiler(supported=False).compile(
            _program(SDMExecutionMode.TRAINING),
            intent=CompilationIntent.INFERENCE,
        ),
        DiagnosticCode.INTENT_CONFLICT,
        "sdm_access",
    )
    assert "mode 'training'" in str(error)


def test_inference_semantics_reject_training_intent() -> None:
    error = _assert_error(
        lambda: _compiler().compile(
            _program(SDMExecutionMode.INFERENCE),
            intent=CompilationIntent.TRAINING,
        ),
        DiagnosticCode.INTENT_CONFLICT,
        "sdm_access",
    )
    assert "mode 'inference'" in str(error)


def test_valid_inference_and_training_intents_select_canonical_anchor() -> None:
    inference = _compiler().compile(
        _program(SDMExecutionMode.INFERENCE),
        intent=CompilationIntent.INFERENCE,
    )
    training = _compiler().compile(
        _program(SDMExecutionMode.TRAINING),
        intent=CompilationIntent.TRAINING,
    )
    assert inference.trace.anchors == (SDM_EXTERNAL_ANCHOR_NAME,)
    assert training.trace.anchors == (SDM_EXTERNAL_ANCHOR_NAME,)


def test_training_declines_anchor_without_verified_backward() -> None:
    uncertified = replace(CANONICAL_ANCHOR, backward_verified_dtypes=frozenset())
    error = _assert_error(
        lambda: _compiler(anchor=uncertified).compile(
            _program(SDMExecutionMode.TRAINING, DType.FLOAT32),
            intent=CompilationIntent.TRAINING,
        ),
        DiagnosticCode.INTENT_CONFLICT,
        "sdm_access",
    )
    assert "backward is not verified for float32" in str(error)


def test_exact_canonical_override_uses_supported_runtime_selector() -> None:
    params = ScheduleParams(anchor_overrides={"sdm_access": SDM_EXTERNAL_ANCHOR_NAME})
    result = _compiler().compile(_program(), schedule_params=params)
    assert result.trace.anchors == (SDM_EXTERNAL_ANCHOR_NAME,)


def test_exact_canonical_override_cannot_bypass_runtime_decline() -> None:
    params = ScheduleParams(anchor_overrides={"sdm_access": SDM_EXTERNAL_ANCHOR_NAME})
    error = _assert_error(
        lambda: _compiler(
            supported=False,
            code="missing_dependency",
            reason="upstream checkout is absent",
        ).compile(_program(), schedule_params=params),
        DiagnosticCode.DEPENDENCY_MISSING,
        "sdm_access",
    )
    assert "checkout is absent" in str(error)


@pytest.mark.parametrize("override_key", ["sdm_access", "*"])
def test_incompatible_known_override_is_rejected(override_key: str) -> None:
    params = ScheduleParams(anchor_overrides={override_key: "routed_reduction_v1"})
    error = _assert_error(
        lambda: _compiler().compile(_program(), schedule_params=params),
        DiagnosticCode.ANCHOR_DECLINED,
        "sdm_access",
    )
    assert "incompatible with the canonical SDM external anchor" in str(error)


def test_unknown_override_is_schedule_error_with_operation_subject() -> None:
    params = ScheduleParams(anchor_overrides={"sdm_access": "unknown_sdm_anchor"})
    _assert_error(
        lambda: _compiler().compile(_program(), schedule_params=params),
        DiagnosticCode.SCHEDULE_HINT_INVALID,
        "sdm_access",
    )


def test_compiler_declines_semantic_drift_before_dispatch() -> None:
    program = _program()
    op = program.ops[0]
    assert isinstance(op, SparseDeltaMemoryAccess)
    drifted = replace(op, spec=replace(op.spec, collision_policy=MergePolicy.SUM))
    _assert_error(
        lambda: _compiler().compile(program.replaced((drifted,))),
        DiagnosticCode.UNSUPPORTED_SEMANTICS,
        "sdm_access",
    )


def test_compiler_preserves_revision_decline_reason() -> None:
    error = _assert_error(
        lambda: _compiler(
            supported=False,
            code="incompatible_revision",
            reason="installed deadbeef does not equal pinned commit",
        ).compile(_program()),
        DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
        "sdm_access",
    )
    assert "deadbeef" in str(error)


def test_sdm_anchor_kind_remains_distinct_from_page_local_urm_work() -> None:
    assert CANONICAL_ANCHOR.kind is AnchorKind.SPARSE_DELTA_MEMORY
    assert CANONICAL_ANCHOR.name == SDM_EXTERNAL_ANCHOR_NAME
