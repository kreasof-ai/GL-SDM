"""Pinned comparator parity for the native kernel-only SparseStateMixer path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.adapters.sparse_delta_memory import (
    MODE_INFERENCE,
    MODE_READ_ONLY,
    MODE_TRAINING,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from urm.adapters.sparse_state_mixer_external import SDMSparseStateMixerFallback

SUPPORT = probe_sdm_support()
if not SUPPORT.supported:
    pytest.skip(
        f"pinned original SDM is unavailable [{SUPPORT.code}]: {SUPPORT.reason}",
        allow_module_level=True,
    )

from urm.backends.sparse_state_mixer import (
    CertifiedSparseStateRoutes,
    SparseState,
    TritonSparseStateMixerBackend,
)
from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.compiler.semantic import (
    DType,
    SparseReadTiming,
    SparseStateExecutionMode,
    SparseStateMixerSpec,
    SparseStateOperation,
)

FORWARD_TOLERANCES = {
    # Chunked comparator association is most visible under repeated collisions
    # (observed state max 0.010806). Freeze the same 0.02 absolute envelope as
    # the external-baseline artifact before native performance tuning.
    torch.float32: {"atol": 2e-2, "rtol": 3e-3},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}
BACKWARD_TOLERANCES = {
    torch.float32: {"atol": 3e-5, "rtol": 3e-4},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
}


def _routes(parallel, sequence, slots, width, dtype, *, offset, collision_heavy):
    rows = []
    for partition in range(parallel):
        tokens = []
        for token in range(sequence):
            base = offset if collision_heavy else offset + token * (width + 1)
            tokens.append(
                [(base + partition * 7 + route * 11) % slots for route in range(width)]
            )
        rows.append(tokens)
    indices = torch.tensor(rows, device="cuda", dtype=torch.int64).sort(-1).values
    if bool((indices[..., 1:] <= indices[..., :-1]).any().item()):
        raise AssertionError("test route generator produced duplicates")
    generator = torch.Generator(device="cuda").manual_seed(
        3301 + sequence + width + offset
    )
    weights = torch.softmax(
        torch.randn((parallel, sequence, width), device="cuda", generator=generator).to(
            dtype
        ),
        dim=-1,
    ).contiguous()
    return indices.contiguous(), weights


def _global(indices, slots):
    parallel = indices.shape[0]
    offsets = torch.arange(parallel, device="cuda", dtype=torch.int64).view(
        parallel, 1, 1
    )
    return (indices + offsets * slots).contiguous()


def _sample(dtype, *, sequence, dim, width, collision_heavy=False):
    parallel, slots = 1, 256
    write_indices, write_weights = _routes(
        parallel,
        sequence,
        slots,
        width,
        dtype,
        offset=1,
        collision_heavy=collision_heavy,
    )
    read_indices, read_weights = _routes(
        parallel,
        sequence,
        slots,
        width,
        dtype,
        offset=2,
        collision_heavy=collision_heavy,
    )
    generator = torch.Generator(device="cuda").manual_seed(4401 + sequence + dim)
    memory = (
        torch.randn(
            (parallel, slots, dim), device="cuda", dtype=dtype, generator=generator
        )
        * 0.05
    ).contiguous()
    values = (
        torch.randn(
            (parallel, sequence, dim),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.05
    ).contiguous()
    beta = torch.rand(
        (parallel, sequence, 1), device="cuda", dtype=dtype, generator=generator
    ).contiguous()
    decay = (
        -torch.rand(
            (parallel, sequence, 1),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.1
    ).contiguous()
    return (
        slots,
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        decay,
        read_indices,
        read_weights,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "sequence,dim,width,collision_heavy",
    [(1, 37, 1, False), (7, 95, 4, True), (128, 64, 4, False)],
)
def test_update_forward_matches_pinned_upstream(
    dtype, sequence, dim, width, collision_heavy
) -> None:
    (
        slots,
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        decay,
        read_indices,
        read_weights,
    ) = _sample(
        dtype,
        sequence=sequence,
        dim=dim,
        width=width,
        collision_heavy=collision_heavy,
    )
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    spec = SparseStateMixerSpec(
        1,
        sequence,
        slots,
        dim,
        width,
        width,
        dtype_spec,
        SparseStateOperation.UPDATE,
        SparseReadTiming.AFTER_UPDATE,
    )
    routes = CertifiedSparseStateRoutes.certify(
        spec,
        read_indices,
        read_weights,
        write_indices=write_indices,
        write_weights=write_weights,
    )
    native = TritonSparseStateMixerBackend(spec)
    prepared = native.prepare(routes, values=values, beta=beta, log_decay=decay)
    native_output, native_state = native.execute(SparseState(memory.clone()), prepared)
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=slots,
        value_dim=dim,
        num_writes=width,
        num_reads=width,
        mode=MODE_INFERENCE,
        dtype=dtype,
    )
    upstream_memory = memory.flatten(0, 1).clone()
    upstream_output, _ = upstream.direct_calls["update"](
        upstream_memory,
        _global(write_indices, slots),
        write_weights,
        values,
        beta,
        decay,
        _global(read_indices, slots),
        read_weights,
    )
    torch.testing.assert_close(
        native_output.float(), upstream_output.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native_state.memory.flatten(0, 1).float(),
        upstream_memory.float(),
        **FORWARD_TOLERANCES[dtype],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_read_only_matches_pinned_upstream(dtype) -> None:
    sequence, slots, dim, reads = 13, 256, 37, 4
    indices, weights = _routes(
        1, sequence, slots, reads, dtype, offset=3, collision_heavy=True
    )
    generator = torch.Generator(device="cuda").manual_seed(5501)
    memory = torch.randn(
        (1, slots, dim), device="cuda", dtype=dtype, generator=generator
    ).contiguous()
    spec = SparseStateMixerSpec(
        1,
        sequence,
        slots,
        dim,
        0,
        reads,
        DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16,
        SparseStateOperation.READ_ONLY,
        SparseReadTiming.CURRENT_STATE,
    )
    routes = CertifiedSparseStateRoutes.certify(spec, indices, weights)
    backend = TritonSparseStateMixerBackend(spec)
    native, _ = backend.execute(SparseState(memory), backend.prepare(routes))
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=slots,
        value_dim=dim,
        num_writes=1,
        num_reads=reads,
        mode=MODE_READ_ONLY,
        dtype=dtype,
    )
    direct = upstream.direct_calls["read"](
        memory.flatten(0, 1), weights, _global(indices, slots)
    )
    torch.testing.assert_close(native.float(), direct.float(), atol=2e-2, rtol=2e-2)


def test_compiler_visible_external_fallback_calls_exact_bound_update() -> None:
    slots, memory, wi, ww, values, beta, decay, ri, rw = _sample(
        torch.float32, sequence=16, dim=37, width=4, collision_heavy=True
    )
    spec = SparseStateMixerSpec(
        1,
        16,
        slots,
        37,
        4,
        4,
        DType.FLOAT32,
        SparseStateOperation.UPDATE,
        SparseReadTiming.AFTER_UPDATE,
    )
    routes = CertifiedSparseStateRoutes.certify(
        spec,
        ri.to(torch.int32),
        rw,
        write_indices=wi.to(torch.int32),
        write_weights=ww,
    )
    fallback = SDMSparseStateMixerFallback(spec)
    bound_update = fallback.bound_callables["update"]
    assert bound_update is fallback.adapter.direct_calls["update"]
    prepared = fallback.prepare(routes, values=values, beta=beta, log_decay=decay)
    direct_memory = memory.flatten(0, 1).clone()
    direct_output, _ = bound_update(
        direct_memory,
        prepared.global_write_indices,
        ww,
        values,
        beta,
        decay,
        prepared.global_read_indices,
        rw,
    )
    adapter_output, adapter_state = fallback.execute(
        SparseState(memory.clone()), prepared
    )
    torch.testing.assert_close(adapter_output, direct_output, atol=0, rtol=0)
    torch.testing.assert_close(
        adapter_state.memory.flatten(0, 1), direct_memory, atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_external_fallback_training_gradients_match_direct(dtype) -> None:
    slots, memory, wi, ww, values, beta, decay, ri, rw = _sample(
        dtype, sequence=16, dim=32, width=4, collision_heavy=True
    )
    spec = SparseStateMixerSpec(
        1,
        16,
        slots,
        32,
        4,
        4,
        DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16,
        SparseStateOperation.UPDATE,
        SparseReadTiming.AFTER_UPDATE,
        mode=SparseStateExecutionMode.TRAINING,
    )
    fallback = SDMSparseStateMixerFallback(spec)
    generator = torch.Generator(device="cuda").manual_seed(7711)
    read_grad = (
        torch.randn(values.shape, device="cuda", generator=generator).to(dtype)
        / values.numel()
    )
    final_grad = (
        torch.randn(memory.shape, device="cuda", generator=generator).to(dtype)
        / memory.numel()
    )

    def leaves():
        return tuple(
            item.detach().clone().requires_grad_(True)
            for item in (memory, ww, values, beta, decay, rw)
        )

    direct = leaves()
    direct_output, _ = fallback.bound_callables["update"](
        direct[0].flatten(0, 1) + 0,
        _global(wi, slots),
        direct[1],
        direct[2],
        direct[3],
        direct[4],
        _global(ri, slots),
        direct[5],
        grad_final_memory=final_grad.flatten(0, 1).contiguous(),
    )
    direct_gradients = torch.autograd.grad(
        direct_output, direct, grad_outputs=read_grad
    )

    adapted = leaves()
    routes = CertifiedSparseStateRoutes.certify(
        spec,
        ri,
        adapted[5],
        write_indices=wi,
        write_weights=adapted[1],
    )
    prepared = fallback.prepare(
        routes,
        values=adapted[2],
        beta=adapted[3],
        log_decay=adapted[4],
    )
    adapter_output, _ = fallback.execute(
        SparseState(adapted[0]), prepared, grad_final_memory=final_grad
    )
    adapter_gradients = torch.autograd.grad(
        adapter_output, adapted, grad_outputs=read_grad
    )
    torch.testing.assert_close(adapter_output, direct_output, atol=0, rtol=0)
    for adapter_gradient, direct_gradient in zip(
        adapter_gradients, direct_gradients, strict=True
    ):
        assert torch.isfinite(adapter_gradient).all()
        torch.testing.assert_close(
            adapter_gradient, direct_gradient, **BACKWARD_TOLERANCES[dtype]
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_backward_matches_reference_and_pinned_upstream(dtype) -> None:
    sequence, dim, width = 16, 32, 4
    sample = _sample(
        dtype, sequence=sequence, dim=dim, width=width, collision_heavy=True
    )
    slots, memory, wi, ww, values, beta, decay, ri, rw = sample
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    spec = SparseStateMixerSpec(
        1,
        sequence,
        slots,
        dim,
        width,
        width,
        dtype_spec,
        SparseStateOperation.UPDATE,
        SparseReadTiming.AFTER_UPDATE,
        mode=SparseStateExecutionMode.TRAINING,
    )
    generator = torch.Generator(device="cuda").manual_seed(6601)
    reading_cotangent = torch.randn(values.shape, device="cuda", generator=generator)
    memory_cotangent = torch.randn(memory.shape, device="cuda", generator=generator)
    names = (
        "initial_memory",
        "write_weights",
        "values",
        "beta",
        "log_decay",
        "read_weights",
    )

    def leaves():
        return [
            item.detach().clone().requires_grad_(True)
            for item in (memory, ww, values, beta, decay, rw)
        ]

    reference_leaves = leaves()
    reference_output, reference_state = torch_sparse_state_mixer(
        reference_leaves[0],
        ri,
        reference_leaves[5],
        write_indices=wi,
        write_weights=reference_leaves[1],
        values=reference_leaves[2],
        beta=reference_leaves[3],
        log_decay=reference_leaves[4],
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    reference_loss = (reference_output.float() * reading_cotangent).mean() + (
        reference_state.float() * memory_cotangent
    ).mean()
    reference_gradients = torch.autograd.grad(reference_loss, reference_leaves)

    native_leaves = leaves()
    native_routes = CertifiedSparseStateRoutes.certify(
        spec,
        ri,
        native_leaves[5],
        write_indices=wi,
        write_weights=native_leaves[1],
    )
    backend = TritonSparseStateMixerBackend(spec)
    native_output, native_state = backend.execute(
        SparseState(native_leaves[0]),
        backend.prepare(
            native_routes,
            values=native_leaves[2],
            beta=native_leaves[3],
            log_decay=native_leaves[4],
        ),
    )
    native_loss = (native_output.float() * reading_cotangent).mean() + (
        native_state.memory.float() * memory_cotangent
    ).mean()
    native_gradients = torch.autograd.grad(native_loss, native_leaves)

    direct_leaves = leaves()
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=slots,
        value_dim=dim,
        num_writes=width,
        num_reads=width,
        mode=MODE_TRAINING,
        dtype=dtype,
    )
    grad_final = (
        memory_cotangent.flatten(0, 1).to(dtype) / memory.numel()
    ).contiguous()
    direct_output, direct_memory = upstream.direct_calls["update"](
        direct_leaves[0].flatten(0, 1) + 0,
        _global(wi, slots),
        direct_leaves[1],
        direct_leaves[2],
        direct_leaves[3],
        direct_leaves[4],
        _global(ri, slots),
        direct_leaves[5],
        grad_final_memory=grad_final,
    )
    direct_state_snapshot = direct_memory.detach().clone()
    direct_loss = (direct_output.float() * reading_cotangent).mean()
    direct_gradients = torch.autograd.grad(direct_loss, direct_leaves)

    torch.testing.assert_close(
        native_output.float(), direct_output.float(), **FORWARD_TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native_state.memory.float(),
        direct_state_snapshot.view_as(memory).float(),
        **FORWARD_TOLERANCES[dtype],
    )
    for name, native_gradient, reference_gradient, direct_gradient in zip(
        names,
        native_gradients,
        reference_gradients,
        direct_gradients,
        strict=True,
    ):
        assert torch.isfinite(native_gradient).all(), name
        torch.testing.assert_close(
            native_gradient.float(),
            reference_gradient.float(),
            **BACKWARD_TOLERANCES[dtype],
        )
        torch.testing.assert_close(
            native_gradient.float(),
            direct_gradient.float(),
            **BACKWARD_TOLERANCES[dtype],
        )
