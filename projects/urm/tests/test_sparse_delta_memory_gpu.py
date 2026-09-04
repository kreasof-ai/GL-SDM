"""GPU differential gates against the exact pinned original SDM checkout."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from urm.adapters.sparse_delta_memory import (
    MODE_INFERENCE,
    MODE_READ_ONLY,
    MODE_TRAINING,
    SDMState,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from urm.adapters.sparse_delta_memory_reference import (
    deterministic_tie_free_product_key_scores,
    end_to_end_differential_backward_report,
    oracle_product_key,
    oracle_sparse_read,
    oracle_write_read,
    torch_product_key,
    torch_sparse_read,
    torch_write_read,
)
from urm.compiler.execution import SDM_EXTERNAL_ANCHOR_NAME, TRUSTED_ANCHORS

BACKWARD_TOLERANCES = {
    torch.float32: {
        "gradient_atol": 2.5e-5,
        "gradient_rtol": 2.0e-4,
        "forward_atol": 2.5e-3,
        "forward_rtol": 2.0e-3,
    },
    torch.bfloat16: {
        "gradient_atol": 5.0e-5,
        "gradient_rtol": 2.0e-2,
        "forward_atol": 5.0e-3,
        "forward_rtol": 2.0e-2,
    },
}

SUPPORT = probe_sdm_support()
if not SUPPORT.supported:
    pytest.skip(
        f"original SDM dependency unavailable [{SUPPORT.code}]: {SUPPORT.reason}",
        allow_module_level=True,
    )


def _adapter(
    *,
    mode=MODE_INFERENCE,
    dim=37,
    writes=4,
    reads=4,
    chunk=16,
    dtype=torch.float32,
):
    return UrmSparseDeltaMemoryAdapter(
        slots_per_partition=256,
        value_dim=dim,
        num_writes=writes,
        num_reads=reads,
        chunk_size=chunk,
        mode=mode,
        device="cuda",
        dtype=dtype,
    )


def _inputs(adapter, *, p=2, t=7, dim=37, seed=7, dtype=torch.float32):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    key_dim = 32
    write_scores = torch.randn(
        (p, t, key_dim), device="cuda", dtype=dtype, generator=generator
    )
    read_scores = torch.randn(
        (p, t, key_dim), device="cuda", dtype=dtype, generator=generator
    )
    trace = adapter.generate_trace(write_scores, read_scores)
    memory = (
        torch.randn((p * 256, dim), device="cuda", dtype=dtype, generator=generator)
        * 0.1
    )
    values = (
        torch.randn((p, t, dim), device="cuda", dtype=dtype, generator=generator) * 0.1
    )
    beta = torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator)
    decay = (
        -torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator) * 0.25
    )
    return write_scores, read_scores, trace, memory, values, beta, decay


def test_exact_addresses_and_trace_layout_match_both_references() -> None:
    adapter = _adapter()
    write_scores, read_scores, trace, *_ = _inputs(adapter, p=2, t=5)
    for scores, indices, weights, width in (
        (write_scores, trace.write_indices, trace.write_weights, 4),
        (read_scores, trace.read_indices, trace.read_weights, 4),
    ):
        torch_values, torch_indices = torch_product_key(scores, width, 16)
        offsets = torch.arange(2, device="cuda").view(2, 1, 1) * 256
        assert torch.equal(indices, torch_indices + offsets)
        torch.testing.assert_close(weights, torch.softmax(torch_values, dim=-1))

        np_values, np_indices = oracle_product_key(scores.cpu().numpy(), width, 16)
        assert np.array_equal(
            indices.cpu().numpy(), np_indices + np.arange(2)[:, None, None] * 256
        )
        np.testing.assert_allclose(
            weights.cpu().numpy(),
            torch.softmax(torch.from_numpy(np_values), dim=-1).numpy(),
            atol=1e-6,
            rtol=1e-6,
        )
    assert trace.write_indices.shape == (2, 5, 4)
    assert trace.write_indices.dtype == torch.int64


def test_sparse_read_matches_oracle_torch_and_direct_upstream() -> None:
    adapter = _adapter(mode=MODE_READ_ONLY)
    *_, trace, memory, _values, _beta, _decay = _inputs(adapter)
    direct = adapter.direct_calls["read"](
        memory, trace.read_weights, trace.read_indices
    )
    snapshot = memory.clone()
    state = SDMState(memory, sequence_length=5)
    adapted = adapter.read(state, trace)
    eager = torch_sparse_read(memory, trace.read_indices, trace.read_weights)
    oracle = oracle_sparse_read(
        memory.cpu().numpy(),
        trace.read_indices.cpu().numpy(),
        trace.read_weights.cpu().numpy(),
    )
    torch.testing.assert_close(adapted, direct, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(adapted, eager, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(adapted.cpu().numpy(), oracle, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state.memory, snapshot, atol=0, rtol=0)
    assert state.sequence_length == 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_direct_adapter_oracle_and_state_equivalence(dtype) -> None:
    adapter = _adapter(dtype=dtype)
    *_, trace, memory, values, beta, decay = _inputs(adapter, dtype=dtype)
    direct_memory = memory.clone()
    direct, returned = adapter.direct_calls["update"](
        direct_memory,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    state = SDMState(memory.clone(), sequence_length=11)
    adapted, state = adapter.execute(state, trace, values, beta, decay)
    eager, eager_state = torch_write_read(
        memory,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    atol = 2e-2 if dtype is torch.bfloat16 else 2e-5
    rtol = 2e-2 if dtype is torch.bfloat16 else 2e-5
    torch.testing.assert_close(adapted.float(), direct.float(), atol=0, rtol=0)
    torch.testing.assert_close(state.memory.float(), returned.float(), atol=0, rtol=0)
    torch.testing.assert_close(adapted.float(), eager.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        state.memory.float(), eager_state.float(), atol=atol, rtol=rtol
    )
    assert state.sequence_length == 18

    if dtype is torch.float32:
        oracle_out, oracle_state = oracle_write_read(
            memory.cpu().numpy(),
            trace.write_indices.cpu().numpy(),
            trace.write_weights.cpu().numpy(),
            values.cpu().numpy(),
            beta.cpu().numpy(),
            decay.cpu().numpy(),
            trace.read_indices.cpu().numpy(),
            trace.read_weights.cpu().numpy(),
        )
        np.testing.assert_allclose(
            adapted.cpu().numpy(), oracle_out, atol=2e-5, rtol=2e-5
        )
        np.testing.assert_allclose(
            state.memory.cpu().numpy(), oracle_state, atol=2e-5, rtol=2e-5
        )


def test_collision_heavy_ordering_minimal_width_and_non_power_dim() -> None:
    adapter = _adapter(writes=1, reads=1, dim=37)
    write_scores, read_scores, _, memory, values, beta, decay = _inputs(
        adapter, p=1, t=9, dim=37
    )
    # Repeated product-key scores produce a genuine generated trace with heavy
    # cross-token collisions; no arbitrary/manual trace enters adapter dispatch.
    write_scores = write_scores[:, :1].expand(-1, 9, -1).contiguous()
    read_scores = read_scores[:, :1].expand(-1, 9, -1).contiguous()
    trace = adapter.generate_trace(write_scores, read_scores)
    expected, expected_state = torch_write_read(
        memory,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    actual, state = adapter.execute(
        SDMState(memory.clone()), trace, values, beta, decay
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.memory, expected_state, atol=2e-5, rtol=2e-5)
    assert actual.shape == (1, 9, 37)


def test_repeated_incremental_invocation_persists_cache() -> None:
    adapter = _adapter()
    *_, trace, memory, values, beta, decay = _inputs(adapter, p=1, t=6)
    state = SDMState(memory.clone(), sequence_length=3)
    pieces = []
    for token in range(6):
        token_trace = trace.token_slice(token, token + 1)
        out, state = adapter.execute(
            state,
            token_trace,
            values[:, token : token + 1].contiguous(),
            beta[:, token : token + 1].contiguous(),
            decay[:, token : token + 1].contiguous(),
        )
        pieces.append(out)
    expected, expected_state = torch_write_read(
        memory,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.memory, expected_state, atol=2e-5, rtol=2e-5)
    assert state.sequence_length == 9


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_backward_is_differentially_certified(dtype) -> None:
    adapter = _adapter(mode=MODE_TRAINING, dim=32, chunk=16, dtype=dtype)
    *_, _trace, memory, values, beta, decay = _inputs(
        adapter, p=1, t=16, dim=32, dtype=dtype
    )
    write_scores = deterministic_tie_free_product_key_scores(
        parallel=1,
        sequence=16,
        half_key=16,
        num_keys=4,
        device="cuda",
        dtype=dtype,
        seed=20260905,
    )
    read_scores = deterministic_tie_free_product_key_scores(
        parallel=1,
        sequence=16,
        half_key=16,
        num_keys=4,
        device="cuda",
        dtype=dtype,
        seed=20260906,
    )
    report = end_to_end_differential_backward_report(
        adapter,
        write_scores,
        read_scores,
        memory,
        values,
        beta,
        decay,
        **BACKWARD_TOLERANCES[dtype],
    )
    assert report["passed"] is True, report
    assert report["product_key_tie_free"] is True
    assert report["addresses"]["passed"] is True
    for kind in ("write", "read"):
        assert all(report["addresses"][kind].values())
    assert report["route_weights"]["passed"] is True
    for kind in ("write", "read"):
        assert all(report["route_weights"][kind].values())
    assert set(report["gradients"]) == {
        "write_scores",
        "read_scores",
        "initial_memory",
        "values",
        "beta",
        "log_decay",
    }
    for gradient in report["gradients"].values():
        assert all(gradient["finite"].values())
        assert gradient["direct_vs_torch"]["close"] is True
        assert gradient["adapter_vs_torch"]["close"] is True
        assert gradient["adapter_vs_direct"]["close"] is True


def test_advertised_backward_dtypes_equal_differential_gate_coverage() -> None:
    anchor = next(
        item for item in TRUSTED_ANCHORS if item.name == SDM_EXTERNAL_ANCHOR_NAME
    )
    advertised = {str(dtype).removeprefix("torch.") for dtype in BACKWARD_TOLERANCES}
    assert anchor.backward_verified_dtypes == advertised


def test_same_upstream_callable_is_below_direct_and_adapter_dispatch() -> None:
    adapter = _adapter()
    direct = adapter.direct_calls["update"]
    assert direct.__func__ is adapter._update_fn.__func__
    assert direct.__module__ == "lingua.sparse_delta_memory.layer"


def test_dispatch_binds_state_dtype_to_adapter_configuration() -> None:
    adapter = _adapter(mode=MODE_READ_ONLY, dtype=torch.float32)
    *_, trace, memory, _values, _beta, _decay = _inputs(adapter)
    with pytest.raises(ValueError, match="state device/dtype"):
        adapter.read(SDMState(memory.bfloat16()), trace)


def test_unsupported_runtime_and_address_inputs_fail_closed(monkeypatch) -> None:
    adapter = _adapter()
    scores = torch.randn((1, 1, 32), dtype=torch.float32)
    with pytest.raises(ValueError, match="configured CUDA device/dtype"):
        adapter.generate_trace(scores, scores)
    half_scores = scores.cuda().half()
    with pytest.raises(ValueError, match="configured CUDA device/dtype"):
        adapter.generate_trace(half_scores, half_scores)

    monkeypatch.delenv("LIBRARY_PATH", raising=False)
    support = probe_sdm_support()
    assert support.supported is False
    assert support.code == "missing_dependency"
    assert "LIBRARY_PATH" in support.reason
