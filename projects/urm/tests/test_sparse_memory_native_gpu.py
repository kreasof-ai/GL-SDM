"""End-to-end score-to-persistent-state native differential gates."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.adapters.sparse_delta_memory import (
    MODE_INFERENCE,
    MODE_TRAINING,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from urm.adapters.sparse_delta_memory_reference import torch_product_key
from urm.backends.sparse_memory import TritonSparseMemoryBackend
from urm.backends.sparse_route import CertifiedSparseRouteScores
from urm.backends.sparse_state_mixer import SparseState
from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.compiler.semantic import (
    DType,
    SDMExecutionMode,
    SparseMemoryMixerSpec,
    SparseReadTiming,
    SparseStateOperation,
)

SUPPORT = probe_sdm_support()
FORWARD_TOLERANCES = {
    torch.float32: {"atol": 2e-2, "rtol": 2e-3},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}
BACKWARD_TOLERANCES = {
    torch.float32: {"atol": 3e-5, "rtol": 3e-4},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
}


def _scores(spec, dtype, width, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    route_spec = TritonSparseMemoryBackend(spec).read_spec
    if width != spec.reads:
        route_spec = TritonSparseMemoryBackend(spec).write_spec
    rows = []
    row_spec = type(route_spec)(1, 1, spec.slots_per_partition, width, route_spec.dtype)
    while len(rows) < spec.parallel * spec.sequence:
        candidate = torch.randn(
            (1, 1, route_spec.score_width),
            device="cuda",
            dtype=dtype,
            generator=generator,
        ).contiguous()
        try:
            CertifiedSparseRouteScores.certify(row_spec, candidate)
        except ValueError:
            continue
        rows.append(candidate)
    return (
        torch.cat(rows, dim=1)
        .reshape(spec.parallel, spec.sequence, route_spec.score_width)
        .contiguous()
    )


def _case(dtype, *, sequence=16, dim=37, writes=4, reads=4, training=False):
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    spec = SparseMemoryMixerSpec(
        parallel=1,
        sequence=sequence,
        slots_per_partition=256,
        value_dim=dim,
        writes=writes,
        reads=reads,
        dtype=dtype_spec,
        mode=SDMExecutionMode.TRAINING if training else SDMExecutionMode.INFERENCE,
    )
    write_scores = _scores(spec, dtype, writes, 411)
    read_scores = _scores(spec, dtype, reads, 719)
    generator = torch.Generator(device="cuda").manual_seed(1907 + dim)
    memory = (
        torch.randn((1, 256, dim), device="cuda", dtype=dtype, generator=generator)
        * 0.05
    ).contiguous()
    values = (
        torch.randn((1, sequence, dim), device="cuda", dtype=dtype, generator=generator)
        * 0.05
    ).contiguous()
    beta = torch.rand(
        (1, sequence, 1), device="cuda", dtype=dtype, generator=generator
    ).contiguous()
    decay = (
        -torch.rand((1, sequence, 1), device="cuda", dtype=dtype, generator=generator)
        * 0.1
    ).contiguous()
    return spec, write_scores, read_scores, memory, values, beta, decay


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_matches_transparent_reference(dtype) -> None:
    spec, write_scores, read_scores, memory, values, beta, decay = _case(dtype)
    backend = TritonSparseMemoryBackend(spec)
    prepared = backend.prepare(
        read_scores,
        write_scores=write_scores,
        values=values,
        beta=beta,
        log_decay=decay,
    )
    native = backend.execute(SparseState(memory.clone()), prepared)
    write_values, write_addresses = torch_product_key(write_scores, spec.writes, 16)
    read_values, read_addresses = torch_product_key(read_scores, spec.reads, 16)
    reference, final_memory = torch_sparse_state_mixer(
        memory,
        read_addresses,
        torch.softmax(read_values, dim=-1),
        write_indices=write_addresses,
        write_weights=torch.softmax(write_values, dim=-1),
        values=values,
        beta=beta,
        log_decay=decay,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    assert torch.equal(native.write_addresses.to(torch.int64), write_addresses)
    assert torch.equal(native.read_addresses.to(torch.int64), read_addresses)
    torch.testing.assert_close(
        native.readings.float(), reference.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native.state.memory.float(),
        final_memory.float(),
        **FORWARD_TOLERANCES[dtype],
    )


def test_fully_native_read_only_and_persistent_decode() -> None:
    read_spec = SparseMemoryMixerSpec(
        1,
        1,
        256,
        33,
        0,
        4,
        DType.FLOAT32,
        operation=SparseStateOperation.READ_ONLY,
        read_timing=SparseReadTiming.CURRENT_STATE,
    )
    read_scores = _scores(read_spec, torch.float32, 4, 917)
    memory = torch.randn((1, 256, 33), device="cuda")
    read_backend = TritonSparseMemoryBackend(read_spec)
    result = read_backend.execute(
        SparseState(memory.clone()), read_backend.prepare(read_scores)
    )
    values, addresses = torch_product_key(read_scores, 4, 16)
    expected, expected_state = torch_sparse_state_mixer(
        memory, addresses, torch.softmax(values, dim=-1)
    )
    torch.testing.assert_close(result.readings, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(result.state.memory, expected_state)

    spec, write_scores, read_scores, memory, values, beta, decay = _case(
        torch.float32, sequence=1
    )
    backend = TritonSparseMemoryBackend(spec)
    prepared = backend.prepare(
        read_scores,
        write_scores=write_scores,
        values=values,
        beta=beta,
        log_decay=decay,
    )
    state = SparseState(memory.clone())
    pointer = state.memory.data_ptr()
    backend.execute(state, prepared)
    backend.execute(state, prepared)
    assert state.memory.data_ptr() == pointer
    assert state.sequence_length == 2


@pytest.mark.skipif(not SUPPORT.supported, reason="pinned upstream is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_forward_matches_pinned_upstream(dtype) -> None:
    spec, write_scores, read_scores, memory, values, beta, decay = _case(dtype)
    native_backend = TritonSparseMemoryBackend(spec)
    native = native_backend.execute(
        SparseState(memory.clone()),
        native_backend.prepare(
            read_scores,
            write_scores=write_scores,
            values=values,
            beta=beta,
            log_decay=decay,
        ),
    )
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=256,
        value_dim=spec.value_dim,
        num_writes=spec.writes,
        num_reads=spec.reads,
        chunk_size=16,
        mode=MODE_INFERENCE,
        device="cuda",
        dtype=dtype,
    )
    trace = upstream.generate_trace(write_scores, read_scores)
    readings, final = upstream.direct_calls["update"](
        memory.reshape(256, spec.value_dim).clone(),
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    assert torch.equal(native.write_addresses.to(torch.int64), trace.write_indices)
    assert torch.equal(native.read_addresses.to(torch.int64), trace.read_indices)
    torch.testing.assert_close(
        native.readings.float(), readings.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native.state.memory.reshape_as(final).float(),
        final.float(),
        **FORWARD_TOLERANCES[dtype],
    )


@pytest.mark.skipif(not SUPPORT.supported, reason="pinned upstream is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_score_to_state_gradients_match_reference_and_upstream(
    dtype,
) -> None:
    spec, *base = _case(dtype, training=True)
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    reading_cotangent = torch.randn(
        (1, 16, spec.value_dim), device="cuda", generator=generator
    )
    memory_cotangent = torch.randn(
        (1, 256, spec.value_dim), device="cuda", generator=generator
    )

    def leaves():
        return [item.detach().clone().requires_grad_(True) for item in base]

    def run_native():
        write_scores, read_scores, memory, values, beta, decay = leaves()
        backend = TritonSparseMemoryBackend(spec)
        result = backend.execute(
            SparseState(memory),
            backend.prepare(
                read_scores,
                write_scores=write_scores,
                values=values,
                beta=beta,
                log_decay=decay,
            ),
        )
        loss = (result.readings.float() * reading_cotangent).mean() + (
            result.state.memory.float() * memory_cotangent
        ).mean()
        return result, torch.autograd.grad(
            loss, (write_scores, read_scores, memory, values, beta, decay)
        )

    def run_reference():
        write_scores, read_scores, memory, values, beta, decay = leaves()
        write_values, write_addresses = torch_product_key(write_scores, spec.writes, 16)
        read_values, read_addresses = torch_product_key(read_scores, spec.reads, 16)
        readings, final = torch_sparse_state_mixer(
            memory,
            read_addresses,
            torch.softmax(read_values, dim=-1),
            write_indices=write_addresses,
            write_weights=torch.softmax(write_values, dim=-1),
            values=values,
            beta=beta,
            log_decay=decay,
            read_timing=SparseReadTiming.AFTER_UPDATE,
        )
        loss = (readings.float() * reading_cotangent).mean() + (
            final.float() * memory_cotangent
        ).mean()
        return (write_addresses, read_addresses), torch.autograd.grad(
            loss, (write_scores, read_scores, memory, values, beta, decay)
        )

    def run_upstream():
        write_scores, read_scores, memory, values, beta, decay = leaves()
        adapter = UrmSparseDeltaMemoryAdapter(
            slots_per_partition=256,
            value_dim=spec.value_dim,
            num_writes=spec.writes,
            num_reads=spec.reads,
            chunk_size=16,
            mode=MODE_TRAINING,
            device="cuda",
            dtype=dtype,
        )
        address = adapter.direct_calls["address"]
        write_values, write_addresses = address(write_scores, spec.writes, 16)
        read_values, read_addresses = address(read_scores, spec.reads, 16)
        write_weights = adapter.layer.write_act(write_values)
        read_weights = adapter.layer.read_act(read_values)
        flat_memory = memory.reshape(256, spec.value_dim)
        grad_final = (
            memory_cotangent.reshape_as(flat_memory) / flat_memory.numel()
        ).to(dtype)
        readings, _ = adapter.direct_calls["update"](
            flat_memory + 0,
            write_addresses,
            write_weights,
            values,
            beta,
            decay,
            read_addresses,
            read_weights,
            grad_final_memory=grad_final.contiguous(),
        )
        loss = (readings.float() * reading_cotangent).mean()
        return (write_addresses, read_addresses), torch.autograd.grad(
            loss, (write_scores, read_scores, memory, values, beta, decay)
        )

    native, native_gradients = run_native()
    reference_addresses, reference_gradients = run_reference()
    upstream_addresses, upstream_gradients = run_upstream()
    assert torch.equal(native.write_addresses.to(torch.int64), reference_addresses[0])
    assert torch.equal(native.read_addresses.to(torch.int64), reference_addresses[1])
    assert torch.equal(upstream_addresses[0], reference_addresses[0])
    assert torch.equal(upstream_addresses[1], reference_addresses[1])
    for native_gradient, reference_gradient, upstream_gradient in zip(
        native_gradients, reference_gradients, upstream_gradients, strict=True
    ):
        assert torch.isfinite(native_gradient).all()
        torch.testing.assert_close(
            native_gradient.float(),
            reference_gradient.float(),
            **BACKWARD_TOLERANCES[dtype],
        )
        torch.testing.assert_close(
            native_gradient.float(),
            upstream_gradient.float(),
            **BACKWARD_TOLERANCES[dtype],
        )
