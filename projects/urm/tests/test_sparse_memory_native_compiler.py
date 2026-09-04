"""Torch-independent compiler gates for the fully native Sparse Memory path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.execution import (
    NATIVE_SPARSE_MEMORY_ANCHOR_NAME,
    SDM_EXTERNAL_ANCHOR_NAME,
    TRUSTED_ANCHORS,
    AnchorRegistry,
    make_native_sparse_memory_selector,
    make_sdm_selector,
)
from urm.compiler.planner import ScheduleParams, UrmCompiler
from urm.compiler.semantic import (
    DType,
    SparseReadTiming,
    SparseStateOperation,
    sparse_delta_memory_program,
)


def _compiler(*, native_supported=True, upstream_supported=True):
    native = next(
        item
        for item in TRUSTED_ANCHORS
        if item.name == NATIVE_SPARSE_MEMORY_ANCHOR_NAME
    )
    upstream = next(
        item for item in TRUSTED_ANCHORS if item.name == SDM_EXTERNAL_ANCHOR_NAME
    )
    registry = AnchorRegistry()
    native_status = SimpleNamespace(
        supported=native_supported,
        code="supported" if native_supported else "unsupported_shape",
        reason=None if native_supported else "outside native route envelope",
    )
    upstream_status = SimpleNamespace(
        supported=upstream_supported,
        code="supported" if upstream_supported else "missing_dependency",
        reason=None if upstream_supported else "upstream absent",
    )
    registry.register(
        make_native_sparse_memory_selector(native, lambda _spec: native_status)
    )
    registry.register(make_sdm_selector(upstream, lambda: upstream_status))
    return UrmCompiler(anchors=registry)


def test_composite_sparse_memory_selects_native_with_exact_schedule() -> None:
    result = _compiler().compile(
        sparse_delta_memory_program(
            sequence=16,
            slots_per_partition=256,
            value_dim=95,
            writes=4,
            reads=7,
            dtype=DType.BFLOAT16,
        )
    )
    assert result.trace.anchors == (NATIVE_SPARSE_MEMORY_ANCHOR_NAME,)
    assert result.plan.steps[0].launch_config == {
        "schedule_family": "native_route_then_partition_scan",
        "route_block_half": 16,
        "read_route_block": 8,
        "write_route_block": 4,
        "state_block_d": 128,
        "route_materialization": "explicit_logical_outputs",
        "fusion": "none",
    }


def test_read_only_composite_semantics_are_compiler_visible() -> None:
    program = sparse_delta_memory_program(
        sequence=1,
        slots_per_partition=256,
        value_dim=37,
        writes=0,
        reads=4,
        operation=SparseStateOperation.READ_ONLY,
        read_timing=SparseReadTiming.CURRENT_STATE,
        dtype=DType.FLOAT32,
    )
    assert {item.name for item in program.inputs} == {"read_scores", "memory"}
    assert _compiler().compile(program).trace.anchors == (
        NATIVE_SPARSE_MEMORY_ANCHOR_NAME,
    )


def test_native_decline_falls_back_only_to_revision_aware_external_anchor() -> None:
    program = sparse_delta_memory_program(sequence=16, slots_per_partition=256)
    result = _compiler(native_supported=False).compile(program)
    assert result.trace.anchors == (SDM_EXTERNAL_ANCHOR_NAME,)
    params = ScheduleParams(
        anchor_overrides={"sdm_access": NATIVE_SPARSE_MEMORY_ANCHOR_NAME}
    )
    with pytest.raises(CompilerError) as caught:
        _compiler(native_supported=False).compile(program, schedule_params=params)
    assert caught.value.diagnostics[0].code is DiagnosticCode.ANCHOR_DECLINED
    assert caught.value.diagnostics[0].subject == "sdm_access"


def test_exact_external_override_remains_runtime_revision_aware() -> None:
    program = sparse_delta_memory_program(sequence=16, slots_per_partition=256)
    params = ScheduleParams(anchor_overrides={"sdm_access": SDM_EXTERNAL_ANCHOR_NAME})
    assert _compiler().compile(program, schedule_params=params).trace.anchors == (
        SDM_EXTERNAL_ANCHOR_NAME,
    )
    with pytest.raises(CompilerError) as caught:
        _compiler(upstream_supported=False).compile(program, schedule_params=params)
    assert caught.value.diagnostics[0].code is DiagnosticCode.DEPENDENCY_MISSING
