"""Dependency-light tests for the frozen native SparseStateMixer v0 contract."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.execution import (
    NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,
    TRUSTED_ANCHORS,
    AnchorKind,
    AnchorRegistry,
    make_sparse_state_mixer_selector,
)
from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
from urm.compiler.semantic import (
    DType,
    MergePolicy,
    SparseReadTiming,
    SparseStateExecutionMode,
    SparseStateLayout,
    SparseStateMixerAccess,
    SparseStateMixerSpec,
    SparseStateOperation,
    sparse_state_mixer_program,
)
from urm.sparse_state_mixer import FROZEN_V0_ENVELOPE, sparse_state_spec_status


def _spec(**changes) -> SparseStateMixerSpec:
    base = SparseStateMixerSpec(
        parallel=2,
        sequence=7,
        slots_per_partition=257,
        value_dim=37,
        writes=4,
        reads=3,
        dtype=DType.BFLOAT16,
        operation=SparseStateOperation.UPDATE,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    return replace(base, **changes)


def test_native_semantic_consumes_certified_routes_not_scores() -> None:
    program = sparse_state_mixer_program(
        parallel=2,
        sequence=7,
        slots_per_partition=257,
        value_dim=37,
        writes=4,
        reads=3,
    )
    assert isinstance(program.ops[0], SparseStateMixerAccess)
    assert {item.name for item in program.inputs} == {
        "write_addresses",
        "write_weights",
        "values",
        "beta",
        "log_decay",
        "read_addresses",
        "read_weights",
        "memory",
    }
    serialized = repr(program).lower()
    assert "score" not in serialized
    assert "facebook" not in serialized
    assert "sparse_delta_memory" not in serialized


def test_read_only_and_update_timing_are_explicit() -> None:
    read_only = SparseStateMixerSpec(
        parallel=1,
        sequence=1,
        slots_per_partition=8,
        value_dim=3,
        writes=0,
        reads=1,
        dtype=DType.FLOAT32,
        operation=SparseStateOperation.READ_ONLY,
        read_timing=SparseReadTiming.CURRENT_STATE,
    )
    assert read_only.operation is SparseStateOperation.READ_ONLY
    assert _spec(read_timing=SparseReadTiming.BEFORE_UPDATE).read_timing is (
        SparseReadTiming.BEFORE_UPDATE
    )
    with pytest.raises(ValueError, match="writes=0"):
        replace(read_only, writes=1)
    with pytest.raises(ValueError, match="pre-update or post-update"):
        _spec(read_timing=SparseReadTiming.CURRENT_STATE)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"dtype": DType.FLOAT16}, "float32 and bfloat16"),
        ({"accumulation_dtype": DType.BFLOAT16}, "float32 accumulation"),
        ({"collision_policy": MergePolicy.SUM}, "ordered"),
        ({"within_token_collision_policy": MergePolicy.SUM}, "within-token"),
        ({"state_layout": "flat"}, "partition-slot-value"),
        ({"page_size": 2}, "one logical slot"),
    ],
)
def test_semantic_drift_fails_closed(changes, match) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**changes)


def test_predeclared_shape_and_hardware_envelope_returns_structured_declines() -> None:
    assert FROZEN_V0_ENVELOPE.maximum_route_width == 64
    assert sparse_state_spec_status(
        _spec(), device_type="cuda", compute_capability=(8, 6)
    ).supported
    shape = sparse_state_spec_status(
        _spec(value_dim=FROZEN_V0_ENVELOPE.maximum_value_dim + 1),
        device_type="cuda",
        compute_capability=(8, 6),
    )
    assert shape.code == "unsupported_shape"
    hardware = sparse_state_spec_status(
        _spec(), device_type="cuda", compute_capability=(7, 5)
    )
    assert hardware.code == "unsupported_hardware"
    layout = sparse_state_spec_status(
        _spec(), device_type="cuda", contiguous=False, compute_capability=(8, 6)
    )
    assert layout.code == "unsupported_layout"


def test_contract_and_backend_module_import_without_torch() -> None:
    code = """
import builtins
import sys
sys.path.insert(0, 'src')
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch import attempted')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import urm.sparse_state_mixer
import urm.backends.sparse_state_mixer
import urm.compiler.semantic
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_training_mode_remains_semantic_not_a_kernel_flag() -> None:
    spec = _spec(mode=SparseStateExecutionMode.TRAINING)
    assert spec.mode.value == "training"
    assert spec.state_layout is SparseStateLayout.PARTITION_SLOT_VALUE


def _compiler(*, supported=True, code="supported", reason=None) -> UrmCompiler:
    from types import SimpleNamespace

    anchor = next(
        item
        for item in TRUSTED_ANCHORS
        if item.name == NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME
    )
    registry = AnchorRegistry()
    registry.register(
        make_sparse_state_mixer_selector(
            anchor,
            lambda _spec: SimpleNamespace(
                supported=supported, code=code, reason=reason
            ),
        )
    )
    return UrmCompiler(anchors=registry)


def _training_program():
    return sparse_state_mixer_program(
        parallel=1,
        sequence=5,
        slots_per_partition=64,
        value_dim=37,
        writes=3,
        reads=2,
        mode=SparseStateExecutionMode.TRAINING,
    )


def test_compiler_selects_native_for_valid_inference_and_training() -> None:
    inference = _compiler().compile(sparse_state_mixer_program())
    training = _compiler().compile(
        _training_program(), intent=CompilationIntent.TRAINING
    )
    assert inference.trace.anchors == (NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,)
    assert training.trace.anchors == (NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,)
    anchor = next(
        item
        for item in TRUSTED_ANCHORS
        if item.name == NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME
    )
    assert anchor.kind is AnchorKind.SPARSE_STATE_MIXER


@pytest.mark.parametrize(
    "program,intent",
    [
        (_training_program(), CompilationIntent.INFERENCE),
        (sparse_state_mixer_program(), CompilationIntent.TRAINING),
    ],
)
def test_compiler_rejects_contradictory_intent(program, intent) -> None:
    with pytest.raises(CompilerError) as caught:
        _compiler().compile(program, intent=intent)
    assert caught.value.diagnostics[0].code is DiagnosticCode.INTENT_CONFLICT
    assert caught.value.diagnostics[0].subject == "sparse_state_mixer"


def test_compiler_preserves_native_capability_decline() -> None:
    with pytest.raises(CompilerError) as caught:
        _compiler(
            supported=False,
            code="unsupported_hardware",
            reason="SM80 or newer is required",
        ).compile(sparse_state_mixer_program())
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code is DiagnosticCode.UNSUPPORTED_HARDWARE
    assert diagnostic.subject == "sparse_state_mixer"


def test_exact_override_cannot_bypass_native_capability_selector() -> None:
    params = ScheduleParams(
        anchor_overrides={"sparse_state_mixer": NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME}
    )
    with pytest.raises(CompilerError) as caught:
        _compiler(
            supported=False,
            code="unsupported_shape",
            reason="shape exceeds v0",
        ).compile(sparse_state_mixer_program(), schedule_params=params)
    assert caught.value.diagnostics[0].code is DiagnosticCode.ANCHOR_DECLINED
