"""End-to-end score-to-persistent-state native differential gates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from benchmarks.comparators.sdm.upstream import (
    MODE_INFERENCE,
    MODE_TRAINING,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from benchmarks.comparators.sdm.reference import torch_product_key
from urm.backends.providers.k3.triton_route_launcher import (
    CertifiedSparseRouteScores,
    TritonSparseRouteBackend,
)
from urm.backends.providers.k3.triton_state_launcher import (
    CertifiedSparseStateRoutes,
    SparseState,
    TritonSparseStateMixerBackend,
)
from urm.backends.providers.k3.torch import torch_sparse_state_mixer
from urm.ir.program import (
    DType,
    SparseReadTiming,
    SparseRouteSelectionSpec,
    SparseStateExecutionMode,
    SparseStateMixerSpec,
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


def _route_spec(parallel, sequence, slots_per_partition, width, dtype_spec):
    return SparseRouteSelectionSpec(
        parallel, sequence, slots_per_partition, width, dtype_spec
    )


def _scores(case, dtype, width, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    route_spec = _route_spec(
        case.parallel, case.sequence, case.slots_per_partition, width, case.dtype_spec
    )
    row_spec = _route_spec(1, 1, case.slots_per_partition, width, case.dtype_spec)
    rows = []
    while len(rows) < case.parallel * case.sequence:
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
        .reshape(case.parallel, case.sequence, route_spec.score_width)
        .contiguous()
    )


def _case(dtype, *, sequence=16, dim=37, writes=4, reads=4, training=False):
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    case = SimpleNamespace(
        parallel=1,
        sequence=sequence,
        slots_per_partition=256,
        value_dim=dim,
        writes=writes,
        reads=reads,
        dtype_spec=dtype_spec,
        operation=SparseStateOperation.UPDATE,
        read_timing=SparseReadTiming.AFTER_UPDATE,
        mode=(
            SparseStateExecutionMode.TRAINING
            if training
            else SparseStateExecutionMode.INFERENCE
        ),
    )
    write_scores = _scores(case, dtype, writes, 411)
    read_scores = _scores(case, dtype, reads, 719)
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
    return case, write_scores, read_scores, memory, values, beta, decay


def _state_spec(case):
    return SparseStateMixerSpec(
        parallel=case.parallel,
        sequence=case.sequence,
        slots_per_partition=case.slots_per_partition,
        value_dim=case.value_dim,
        writes=case.writes,
        reads=case.reads,
        dtype=case.dtype_spec,
        operation=case.operation,
        read_timing=case.read_timing,
        mode=case.mode,
    )


def _execute_native(case, read_scores, write_scores, memory, values, beta, decay):
    """Compose route + state backends: scores -> certified routes -> state."""
    state_spec = _state_spec(case)
    state_backend = TritonSparseStateMixerBackend(state_spec)
    read_route_spec = _route_spec(
        case.parallel,
        case.sequence,
        case.slots_per_partition,
        case.reads,
        case.dtype_spec,
    )
    read_output = TritonSparseRouteBackend(read_route_spec).generate_certified(
        CertifiedSparseRouteScores.certify(read_route_spec, read_scores)
    )
    write_output = None
    if case.operation is SparseStateOperation.UPDATE:
        write_route_spec = _route_spec(
            case.parallel,
            case.sequence,
            case.slots_per_partition,
            case.writes,
            case.dtype_spec,
        )
        write_output = TritonSparseRouteBackend(write_route_spec).generate_certified(
            CertifiedSparseRouteScores.certify(write_route_spec, write_scores)
        )
    routes = CertifiedSparseStateRoutes.from_native_generation(
        state_spec, read_output, write_output=write_output
    )
    prepared = state_backend.prepare(
        routes, values=values, beta=beta, log_decay=decay
    )
    state = SparseState(memory=memory, sequence_length=0)
    readings, state = state_backend.execute(state, prepared)
    return read_output, write_output, readings, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_matches_transparent_reference(dtype) -> None:
    case, write_scores, read_scores, memory, values, beta, decay = _case(dtype)
    read_output, write_output, readings, state = _execute_native(
        case, read_scores, write_scores, memory.clone(), values, beta, decay
    )
    write_values, write_addresses = torch_product_key(write_scores, case.writes, 16)
    read_values, read_addresses = torch_product_key(read_scores, case.reads, 16)
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
    assert torch.equal(write_output.addresses.to(torch.int64), write_addresses)
    assert torch.equal(read_output.addresses.to(torch.int64), read_addresses)
    torch.testing.assert_close(
        readings.float(), reference.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        state.memory.float(),
        final_memory.float(),
        **FORWARD_TOLERANCES[dtype],
    )


def test_fully_native_read_only_and_persistent_decode() -> None:
    read_case = SimpleNamespace(
        parallel=1,
        sequence=1,
        slots_per_partition=256,
        value_dim=33,
        writes=0,
        reads=4,
        dtype_spec=DType.FLOAT32,
        operation=SparseStateOperation.READ_ONLY,
        read_timing=SparseReadTiming.CURRENT_STATE,
        mode=SparseStateExecutionMode.INFERENCE,
    )
    read_scores = _scores(read_case, torch.float32, 4, 917)
    memory = torch.randn((1, 256, 33), device="cuda")
    read_state_spec = _state_spec(read_case)
    read_state_backend = TritonSparseStateMixerBackend(read_state_spec)
    read_route_spec = _route_spec(
        read_case.parallel,
        read_case.sequence,
        read_case.slots_per_partition,
        read_case.reads,
        read_case.dtype_spec,
    )
    read_output = TritonSparseRouteBackend(read_route_spec).generate_certified(
        CertifiedSparseRouteScores.certify(read_route_spec, read_scores)
    )
    read_routes = CertifiedSparseStateRoutes.from_native_generation(
        read_state_spec, read_output
    )
    read_prepared = read_state_backend.prepare(read_routes)
    readings, read_state = read_state_backend.execute(
        SparseState(memory=memory.clone(), sequence_length=0), read_prepared
    )
    values, addresses = torch_product_key(read_scores, 4, 16)
    expected, expected_state = torch_sparse_state_mixer(
        memory, addresses, torch.softmax(values, dim=-1)
    )
    torch.testing.assert_close(readings, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(read_state.memory, expected_state)

    case, write_scores, read_scores, memory, values, beta, decay = _case(
        torch.float32, sequence=1
    )
    state_spec = _state_spec(case)
    state_backend = TritonSparseStateMixerBackend(state_spec)
    read_route_spec = _route_spec(
        case.parallel, case.sequence, case.slots_per_partition, case.reads, case.dtype_spec
    )
    write_route_spec = _route_spec(
        case.parallel, case.sequence, case.slots_per_partition, case.writes, case.dtype_spec
    )
    read_output = TritonSparseRouteBackend(read_route_spec).generate_certified(
        CertifiedSparseRouteScores.certify(read_route_spec, read_scores)
    )
    write_output = TritonSparseRouteBackend(write_route_spec).generate_certified(
        CertifiedSparseRouteScores.certify(write_route_spec, write_scores)
    )
    routes = CertifiedSparseStateRoutes.from_native_generation(
        state_spec, read_output, write_output=write_output
    )
    prepared = state_backend.prepare(
        routes, values=values, beta=beta, log_decay=decay
    )
    state = SparseState(memory=memory.clone(), sequence_length=0)
    pointer = state.memory.data_ptr()
    state_backend.execute(state, prepared)
    state_backend.execute(state, prepared)
    assert state.memory.data_ptr() == pointer
    assert state.sequence_length == 2


@pytest.mark.skipif(not SUPPORT.supported, reason="pinned upstream is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_forward_matches_pinned_upstream(dtype) -> None:
    case, write_scores, read_scores, memory, values, beta, decay = _case(dtype)
    read_output, write_output, readings, state = _execute_native(
        case, read_scores, write_scores, memory.clone(), values, beta, decay
    )
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=256,
        value_dim=case.value_dim,
        num_writes=case.writes,
        num_reads=case.reads,
        chunk_size=16,
        mode=MODE_INFERENCE,
        device="cuda",
        dtype=dtype,
    )
    trace = upstream.generate_trace(write_scores, read_scores)
    upstream_readings, final = upstream.direct_calls["update"](
        memory.reshape(256, case.value_dim).clone(),
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        decay,
        trace.read_indices,
        trace.read_weights,
    )
    assert torch.equal(write_output.addresses.to(torch.int64), trace.write_indices)
    assert torch.equal(read_output.addresses.to(torch.int64), trace.read_indices)
    torch.testing.assert_close(
        readings.float(), upstream_readings.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        state.memory.reshape_as(final).float(),
        final.float(),
        **FORWARD_TOLERANCES[dtype],
    )


@pytest.mark.skipif(not SUPPORT.supported, reason="pinned upstream is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fully_native_e2e_score_to_state_gradients_match_reference_and_upstream(
    dtype,
) -> None:
    case, *base = _case(dtype, training=True)
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    reading_cotangent = torch.randn(
        (1, 16, case.value_dim), device="cuda", generator=generator
    )
    memory_cotangent = torch.randn(
        (1, 256, case.value_dim), device="cuda", generator=generator
    )

    def leaves():
        return [item.detach().clone().requires_grad_(True) for item in base]

    def run_native():
        write_scores, read_scores, memory, values, beta, decay = leaves()
        read_output, write_output, readings, state = _execute_native(
            case, read_scores, write_scores, memory, values, beta, decay
        )
        loss = (readings.float() * reading_cotangent).mean() + (
            state.memory.float() * memory_cotangent
        ).mean()
        gradients = torch.autograd.grad(
            loss, (write_scores, read_scores, memory, values, beta, decay)
        )
        return (write_output, read_output), gradients

    def run_reference():
        write_scores, read_scores, memory, values, beta, decay = leaves()
        write_values, write_addresses = torch_product_key(
            write_scores, case.writes, 16
        )
        read_values, read_addresses = torch_product_key(read_scores, case.reads, 16)
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
            value_dim=case.value_dim,
            num_writes=case.writes,
            num_reads=case.reads,
            chunk_size=16,
            mode=MODE_TRAINING,
            device="cuda",
            dtype=dtype,
        )
        address = adapter.direct_calls["address"]
        write_values, write_addresses = address(write_scores, case.writes, 16)
        read_values, read_addresses = address(read_scores, case.reads, 16)
        write_weights = adapter.layer.write_act(write_values)
        read_weights = adapter.layer.read_act(read_values)
        flat_memory = memory.reshape(256, case.value_dim)
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

    (native_write, native_read), native_gradients = run_native()
    reference_addresses, reference_gradients = run_reference()
    upstream_addresses, upstream_gradients = run_upstream()
    assert torch.equal(native_write.addresses.to(torch.int64), reference_addresses[0])
    assert torch.equal(native_read.addresses.to(torch.int64), reference_addresses[1])
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
