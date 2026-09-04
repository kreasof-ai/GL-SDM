"""CPU-safe contract and compiler tests for the optional original-SDM adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urm.adapters.sparse_delta_memory import (
    EXPECTED_SDM_COMMIT,
    MODE_READ_ONLY,
    SDMAddressTrace,
    SDMOperationSpec,
    SDMState,
    _checkout_root,
    probe_sdm_support,
    sdm_upstream_identity,
)
from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.planner import ScheduleParams, UrmCompiler
from urm.compiler.semantic import (
    DType,
    SDMExecutionMode,
    SparseDeltaMemoryAccess,
    sparse_delta_memory_program,
)


def test_upstream_pin_and_install_identity_are_explicit() -> None:
    assert EXPECTED_SDM_COMMIT == "183e7df809131b80ad4393741029d0f20fc3640b"
    identity = sdm_upstream_identity()
    assert identity["expected_commit"] == EXPECTED_SDM_COMMIT
    assert identity["repository"].endswith("facebookresearch/sparse-delta-memory")
    if identity.get("status") != "not_applicable":
        assert identity["license"] == "CC-BY-NC-4.0"
        assert "PYTHONPATH" in identity["installation"]
        assert identity["source_usage"].startswith("external checkout")


def test_checkout_root_requires_real_git_checkout(tmp_path) -> None:
    module = tmp_path / "lingua" / "sparse_delta_memory" / "layer.py"
    module.parent.mkdir(parents=True)
    module.touch()
    assert _checkout_root(str(module)) is None


def test_upstream_kernel_source_is_not_vendored() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert not (project_root / "src" / "lingua" / "sparse_delta_memory").exists()


def test_trace_rejects_duplicates_cross_partition_and_empty() -> None:
    weights = torch.ones((1, 1, 2), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="duplicates"):
        SDMAddressTrace.from_tensors(
            torch.tensor([[[1, 1]]]),
            weights,
            torch.tensor([[[2, 3]]]),
            weights,
            slots_per_partition=64,
        )
    with pytest.raises(ValueError, match="partition"):
        SDMAddressTrace.from_tensors(
            torch.tensor([[[1, 2]], [[2, 66]]]),
            torch.ones((2, 1, 2)),
            torch.tensor([[[3, 4]], [[67, 68]]]),
            torch.ones((2, 1, 2)),
            slots_per_partition=64,
        )
    with pytest.raises(ValueError, match="empty"):
        SDMAddressTrace.from_tensors(
            torch.empty((1, 0, 1), dtype=torch.int64),
            torch.empty((1, 0, 1)),
            torch.empty((1, 0, 1), dtype=torch.int64),
            torch.empty((1, 0, 1)),
            slots_per_partition=64,
        )


def test_cpu_execution_is_explicitly_rejected_at_the_call_boundary() -> None:
    trace = SDMAddressTrace.from_tensors(
        torch.tensor([[[1]]]),
        torch.ones((1, 1, 1)),
        torch.tensor([[[2]]]),
        torch.ones((1, 1, 1)),
        slots_per_partition=64,
    )
    with pytest.raises(ValueError, match="require CUDA"):
        SDMOperationSpec.from_call(
            trace,
            SDMState(torch.zeros((64, 7))),
            mode=MODE_READ_ONLY,
        )


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
    assert op.spec.collision_semantics == "ordered_across_tokens_unique_within_token"
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


def test_compiler_selects_exact_adapter_or_structurally_declines() -> None:
    program = sparse_delta_memory_program(
        sequence=8, slots_per_partition=256, value_dim=32, writes=4, reads=4
    )
    support = probe_sdm_support()
    if support.supported:
        result = UrmCompiler().compile(program)
        assert result.trace.anchors == (
            "facebook_sparse_delta_memory_183e7df_external_adapter",
        )
        assert result.plan.steps[0].anchor == result.trace.anchors[0]
        assert "external_adapter" in result.trace.anchors[0]
        assert result.cost.logical_bytes > 0
        assert "analytical SDM" in " ".join(result.cost.notes)
    else:
        with pytest.raises(CompilerError) as error:
            UrmCompiler().compile(program)
        codes = {item.code for item in error.value.diagnostics}
        expected = {
            "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
            "incompatible_revision": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
            "modified_upstream_checkout": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
            "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
            "incompatible_runtime": DiagnosticCode.DEPENDENCY_MISSING,
        }[support.code]
        assert expected in codes


def test_compiler_declines_semantic_drift_before_dispatch(monkeypatch) -> None:
    program = sparse_delta_memory_program(
        sequence=8, slots_per_partition=256, value_dim=32, writes=4, reads=4
    )
    op = program.ops[0]
    assert isinstance(op, SparseDeltaMemoryAccess)
    drifted = replace(op, spec=replace(op.spec, key_weighted_decay=True))
    bad = program.replaced((drifted,))
    with pytest.raises(CompilerError) as error:
        UrmCompiler().compile(bad)
    assert error.value.diagnostics[0].code is DiagnosticCode.UNSUPPORTED_SEMANTICS


def test_compiler_decline_preserves_revision_reason(monkeypatch) -> None:
    import urm.adapters.sparse_delta_memory as adapter_module

    original = adapter_module.probe_sdm_support

    class Fake:
        supported = False
        code = "incompatible_revision"
        reason = "installed deadbeef does not equal pinned commit"

    monkeypatch.setattr(adapter_module, "probe_sdm_support", lambda: Fake())
    try:
        with pytest.raises(CompilerError) as error:
            UrmCompiler().compile(
                sparse_delta_memory_program(
                    sequence=8,
                    slots_per_partition=256,
                    value_dim=32,
                    writes=4,
                    reads=4,
                )
            )
    finally:
        monkeypatch.setattr(adapter_module, "probe_sdm_support", original)
    assert error.value.diagnostics[0].code is DiagnosticCode.UPSTREAM_REVISION_MISMATCH
    assert "deadbeef" in error.value.diagnostics[0].message


def test_exact_anchor_override_cannot_bypass_runtime_decline(monkeypatch) -> None:
    import urm.adapters.sparse_delta_memory as adapter_module

    class Fake:
        supported = False
        code = "missing_dependency"
        reason = "upstream checkout is absent"

    monkeypatch.setattr(adapter_module, "probe_sdm_support", lambda: Fake())
    program = sparse_delta_memory_program(
        sequence=8, slots_per_partition=256, value_dim=32, writes=4, reads=4
    )
    params = ScheduleParams(
        anchor_overrides={
            "sdm_access": "facebook_sparse_delta_memory_183e7df_external_adapter"
        }
    )
    with pytest.raises(CompilerError) as error:
        UrmCompiler().compile(program, schedule_params=params)
    assert error.value.diagnostics[0].code is DiagnosticCode.DEPENDENCY_MISSING
