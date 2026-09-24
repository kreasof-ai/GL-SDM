"""GPU gates for the exact K2/K3 Triton compile probes.

The schedule stage's probe mechanism is generic: it compiles and launches the
production kernels for the exact target specialization and reads register/
shared-memory facts back from the compiled kernel cache. These tests exercise
the K2 (diagonal + matrix-state recurrence) and K3 (route generation + state
mixer) probes on both inference and training intents.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.compiler.schedule.probes.triton_k2 import make_triton_k2_compile_probe
from urm.compiler.schedule.probes.triton_k3 import make_triton_k3_compile_probe
from urm.compiler.schedule.search import CompileContext


def _context(intent: str) -> CompileContext:
    return CompileContext(
        anchor_name="probe",
        plan="base",
        intent=intent,
        queries=4,
        sources=8,
        route_width=2,
        value_dim=32,
        dtype="bfloat16",
        block_d=32,
        num_warps=4,
        num_stages=2,
        grad_values_decomposition="per_query",
        grad_values_schedule="segmented",
        schedule_point=None,
    )


@pytest.mark.parametrize("intent", ["inference", "training"])
def test_k2_compile_probe_compiles_exact_diagonal_and_matrix_kernels(intent) -> None:
    probe = make_triton_k2_compile_probe()
    result = probe(_context(intent))
    assert result.ok, result.reason
    # Both production kernel families compiled: diagonal and matrix-state.
    assert any("diagonal" in name for name in result.kernel_resources)
    assert any("matrix" in name for name in result.kernel_resources)
    assert result.registers_per_thread is not None
    if intent == "training":
        # Training compiles the backward specializations too.
        assert any("backward" in name or "grad" in name for name in result.kernel_resources)


@pytest.mark.parametrize("intent", ["inference", "training"])
def test_k3_compile_probe_compiles_exact_route_and_state_kernels(intent) -> None:
    probe = make_triton_k3_compile_probe()
    result = probe(_context(intent))
    assert result.ok, result.reason
    assert any("route" in name for name in result.kernel_resources)
    assert any("state" in name for name in result.kernel_resources)
    assert result.registers_per_thread is not None
    if intent == "training":
        assert any("backward" in name or "grad" in name for name in result.kernel_resources)


def test_k3_probe_reports_failure_as_result() -> None:
    # Route widths beyond the slot count are a probe failure, not an exception.
    probe = make_triton_k3_compile_probe(slots_per_partition=64, writes=128)
    result = probe(_context("inference"))
    assert not result.ok
    assert result.reason
