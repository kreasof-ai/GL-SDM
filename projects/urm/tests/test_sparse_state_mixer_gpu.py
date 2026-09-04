"""Forward differential gates for the URM-native SparseStateMixer kernel."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.sparse_state_mixer import (
    CertifiedSparseStateRoutes,
    SparseState,
    TritonSparseStateMixerBackend,
)
from urm.backends.sparse_state_reference import torch_sparse_state_mixer
from urm.compiler.semantic import (
    DType,
    SparseReadTiming,
    SparseStateMixerSpec,
    SparseStateOperation,
)
from urm.sparse_state_mixer import numpy_sparse_state_mixer

TOLERANCES = {
    torch.float32: {"atol": 2e-5, "rtol": 2e-5},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}
BACKWARD_TOLERANCES = {
    torch.float32: {"atol": 3e-5, "rtol": 3e-4},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
}


def _case(
    *,
    dtype=torch.float32,
    parallel=1,
    sequence=5,
    slots=67,
    dim=37,
    writes=3,
    reads=2,
    timing=SparseReadTiming.AFTER_UPDATE,
    collision_heavy=False,
):
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    operation = (
        SparseStateOperation.READ_ONLY if writes == 0 else SparseStateOperation.UPDATE
    )
    if operation is SparseStateOperation.READ_ONLY:
        timing = SparseReadTiming.CURRENT_STATE
    spec = SparseStateMixerSpec(
        parallel=parallel,
        sequence=sequence,
        slots_per_partition=slots,
        value_dim=dim,
        writes=writes,
        reads=reads,
        dtype=dtype_spec,
        operation=operation,
        read_timing=timing,
    )
    generator = torch.Generator(device="cuda").manual_seed(9103 + dim + sequence)

    def addresses(width, offset):
        rows = []
        for p in range(parallel):
            tokens = []
            for t in range(sequence):
                base = offset if collision_heavy else offset + t * (width + 1)
                tokens.append(
                    torch.tensor(
                        sorted({(base + p * 3 + k * 5) % slots for k in range(width)})
                    )
                )
            rows.append(torch.stack(tokens))
        result = torch.stack(rows).to(device="cuda", dtype=torch.int64)
        if result.shape[-1] != width:
            raise AssertionError("test route generator produced a collision")
        return result.contiguous()

    read_indices = addresses(reads, 2)
    read_weights = torch.softmax(
        torch.randn((parallel, sequence, reads), device="cuda", generator=generator).to(
            dtype
        ),
        dim=-1,
    ).contiguous()
    kwargs = {}
    if writes:
        kwargs["write_indices"] = addresses(writes, 1)
        kwargs["write_weights"] = torch.softmax(
            torch.randn(
                (parallel, sequence, writes), device="cuda", generator=generator
            ).to(dtype),
            dim=-1,
        ).contiguous()
    routes = CertifiedSparseStateRoutes.certify(
        spec, read_indices, read_weights, **kwargs
    )
    memory = (
        torch.randn((parallel, slots, dim), device="cuda", generator=generator).to(
            dtype
        )
        * 0.05
    ).contiguous()
    values = beta = log_decay = None
    if writes:
        values = (
            torch.randn(
                (parallel, sequence, dim), device="cuda", generator=generator
            ).to(dtype)
            * 0.05
        ).contiguous()
        beta = torch.rand(
            (parallel, sequence, 1), device="cuda", dtype=dtype, generator=generator
        ).contiguous()
        log_decay = (
            -torch.rand(
                (parallel, sequence, 1),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.1
        ).contiguous()
    backend = TritonSparseStateMixerBackend(spec)
    prepared = backend.prepare(routes, values=values, beta=beta, log_decay=log_decay)
    return backend, prepared, memory, values, beta, log_decay


def _reference(prepared, memory, values, beta, log_decay):
    routes = prepared.routes
    return torch_sparse_state_mixer(
        memory,
        routes.read_indices,
        routes.read_weights,
        write_indices=routes.write_indices,
        write_weights=routes.write_weights,
        values=values,
        beta=beta,
        log_decay=log_decay,
        read_timing=prepared.spec.read_timing,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence": 1, "slots": 8, "dim": 1, "writes": 0, "reads": 1},
        {"sequence": 7, "slots": 67, "dim": 37, "writes": 3, "reads": 2},
        {
            "sequence": 11,
            "slots": 71,
            "dim": 64,
            "writes": 4,
            "reads": 4,
            "collision_heavy": True,
        },
        {
            "parallel": 2,
            "sequence": 5,
            "slots": 257,
            "dim": 95,
            "writes": 4,
            "reads": 3,
            "timing": SparseReadTiming.BEFORE_UPDATE,
        },
    ],
)
def test_native_forward_matches_torch_and_numpy(dtype, kwargs) -> None:
    backend, prepared, memory, values, beta, log_decay = _case(dtype=dtype, **kwargs)
    expected, expected_state = _reference(prepared, memory, values, beta, log_decay)
    native_state = SparseState(memory.clone())
    actual, native_state = backend.execute(native_state, prepared)
    torch.testing.assert_close(actual.float(), expected.float(), **TOLERANCES[dtype])
    torch.testing.assert_close(
        native_state.memory.float(), expected_state.float(), **TOLERANCES[dtype]
    )
    oracle, oracle_state = numpy_sparse_state_mixer(
        memory.float().cpu().numpy(),
        prepared.routes.read_indices.cpu().numpy(),
        prepared.routes.read_weights.float().cpu().numpy(),
        write_indices=(
            prepared.routes.write_indices.cpu().numpy()
            if prepared.routes.write_indices is not None
            else None
        ),
        write_weights=(
            prepared.routes.write_weights.float().cpu().numpy()
            if prepared.routes.write_weights is not None
            else None
        ),
        values=values.float().cpu().numpy() if values is not None else None,
        beta=beta.float().cpu().numpy() if beta is not None else None,
        log_decay=(log_decay.float().cpu().numpy() if log_decay is not None else None),
        read_timing=prepared.spec.read_timing,
    )
    torch.testing.assert_close(
        actual.float().cpu(), torch.from_numpy(oracle).float(), **TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native_state.memory.float().cpu(),
        torch.from_numpy(oracle_state).float(),
        **TOLERANCES[dtype],
    )


def test_persistent_decode_matches_concatenated_reference() -> None:
    backend, prepared, memory, values, beta, log_decay = _case(
        dtype=torch.float32,
        sequence=1,
        slots=64,
        dim=33,
        writes=2,
        reads=2,
        collision_heavy=True,
    )
    state = SparseState(memory.clone())
    first, state = backend.execute(state, prepared)
    pointer = state.memory.data_ptr()
    second, state = backend.execute(state, prepared)
    reference_first, reference_state = _reference(
        prepared, memory, values, beta, log_decay
    )
    reference_second, reference_state = _reference(
        prepared, reference_state, values, beta, log_decay
    )
    torch.testing.assert_close(first, reference_first, **TOLERANCES[torch.float32])
    torch.testing.assert_close(second, reference_second, **TOLERANCES[torch.float32])
    torch.testing.assert_close(
        state.memory, reference_state, **TOLERANCES[torch.float32]
    )
    assert state.memory.data_ptr() == pointer
    assert state.sequence_length == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "timing", [SparseReadTiming.BEFORE_UPDATE, SparseReadTiming.AFTER_UPDATE]
)
def test_update_backward_covers_every_differentiable_input(dtype, timing) -> None:
    backend, prepared, memory, values, beta, log_decay = _case(
        dtype=dtype,
        sequence=5,
        slots=31,
        dim=37,
        writes=3,
        reads=2,
        timing=timing,
        collision_heavy=True,
    )
    gradient_generator = torch.Generator(device="cuda").manual_seed(771)
    reading_cotangent = torch.randn(
        (1, 5, 37), device="cuda", generator=gradient_generator
    )
    memory_cotangent = torch.randn(
        (1, 31, 37), device="cuda", generator=gradient_generator
    )

    def run_native():
        leaves = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                memory,
                prepared.routes.write_weights,
                values,
                beta,
                log_decay,
                prepared.routes.read_weights,
            )
        ]
        routes = CertifiedSparseStateRoutes.certify(
            prepared.spec,
            prepared.routes.read_indices,
            leaves[5],
            write_indices=prepared.routes.write_indices,
            write_weights=leaves[1],
        )
        native_prepared = backend.prepare(
            routes, values=leaves[2], beta=leaves[3], log_decay=leaves[4]
        )
        readings, state = backend.execute(SparseState(leaves[0]), native_prepared)
        loss = (readings.float() * reading_cotangent).mean() + (
            state.memory.float() * memory_cotangent
        ).mean()
        return readings, state.memory, torch.autograd.grad(loss, leaves)

    def run_reference():
        leaves = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                memory,
                prepared.routes.write_weights,
                values,
                beta,
                log_decay,
                prepared.routes.read_weights,
            )
        ]
        readings, state = torch_sparse_state_mixer(
            leaves[0],
            prepared.routes.read_indices,
            leaves[5],
            write_indices=prepared.routes.write_indices,
            write_weights=leaves[1],
            values=leaves[2],
            beta=leaves[3],
            log_decay=leaves[4],
            read_timing=timing,
        )
        loss = (readings.float() * reading_cotangent).mean() + (
            state.float() * memory_cotangent
        ).mean()
        return readings, state, torch.autograd.grad(loss, leaves)

    native_readings, native_state, native_gradients = run_native()
    reference_readings, reference_state, reference_gradients = run_reference()
    torch.testing.assert_close(
        native_readings.float(), reference_readings.float(), **TOLERANCES[dtype]
    )
    torch.testing.assert_close(
        native_state.float(), reference_state.float(), **TOLERANCES[dtype]
    )
    names = (
        "initial_memory",
        "write_weights",
        "values",
        "beta",
        "log_decay",
        "read_weights",
    )
    for name, actual, expected in zip(
        names, native_gradients, reference_gradients, strict=True
    ):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            msg=lambda message, name=name: f"{name}: {message}",
            **BACKWARD_TOLERANCES[dtype],
        )


def test_read_only_backward_covers_memory_and_weights() -> None:
    backend, prepared, memory, *_ = _case(
        dtype=torch.float32, sequence=7, slots=41, dim=33, writes=0, reads=3
    )
    memory_leaf = memory.detach().clone().requires_grad_(True)
    weights_leaf = prepared.routes.read_weights.detach().clone().requires_grad_(True)
    routes = CertifiedSparseStateRoutes.certify(
        prepared.spec, prepared.routes.read_indices, weights_leaf
    )
    readings, _ = backend.execute(SparseState(memory_leaf), backend.prepare(routes))
    readings.square().mean().backward()
    reference_memory = memory.detach().clone().requires_grad_(True)
    reference_weights = (
        prepared.routes.read_weights.detach().clone().requires_grad_(True)
    )
    reference, _ = torch_sparse_state_mixer(
        reference_memory, prepared.routes.read_indices, reference_weights
    )
    reference.square().mean().backward()
    torch.testing.assert_close(
        memory_leaf.grad, reference_memory.grad, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        weights_leaf.grad, reference_weights.grad, atol=1e-6, rtol=1e-5
    )


def test_fp64_reference_gradcheck() -> None:
    device = "cuda"
    write_indices = torch.tensor([[[1, 4], [1, 4]]], device=device)
    read_indices = torch.tensor([[[0, 4], [1, 5]]], device=device)
    generator = torch.Generator(device=device).manual_seed(47)
    inputs = [
        torch.randn((1, 7, 3), device=device, dtype=torch.float64, generator=generator),
        torch.softmax(
            torch.randn(
                (1, 2, 2), device=device, dtype=torch.float64, generator=generator
            ),
            dim=-1,
        ),
        torch.randn((1, 2, 3), device=device, dtype=torch.float64, generator=generator),
        torch.rand((1, 2, 1), device=device, dtype=torch.float64, generator=generator),
        -torch.rand((1, 2, 1), device=device, dtype=torch.float64, generator=generator),
        torch.softmax(
            torch.randn(
                (1, 2, 2), device=device, dtype=torch.float64, generator=generator
            ),
            dim=-1,
        ),
    ]
    inputs = tuple(item.requires_grad_(True) for item in inputs)

    def formulation(memory, write_weights, values, beta, decay, read_weights):
        return torch_sparse_state_mixer(
            memory,
            read_indices,
            read_weights,
            write_indices=write_indices,
            write_weights=write_weights,
            values=values,
            beta=beta,
            log_decay=decay,
            read_timing=SparseReadTiming.AFTER_UPDATE,
            accumulation_dtype=torch.float64,
        )

    assert torch.autograd.gradcheck(formulation, inputs, eps=1e-6, atol=2e-5, rtol=2e-4)


def test_native_backend_does_not_import_comparator_packages(monkeypatch) -> None:
    real_import = __import__

    def guarded(name, *args, **kwargs):
        forbidden = ("lingua", "flash_attn", "fla", "mamba_ssm", "megablocks")
        if name in forbidden or name.startswith(
            tuple(item + "." for item in forbidden)
        ):
            raise AssertionError(f"comparator import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    backend, prepared, memory, *_ = _case(sequence=1, slots=8, dim=7, writes=0, reads=1)
    output, _ = backend.execute(SparseState(memory), prepared)
    assert torch.isfinite(output).all()


def test_native_executes_in_process_with_upstream_checkout_absent() -> None:
    project = Path(__file__).parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project / "src")
    code = """
import importlib.util
import torch
assert importlib.util.find_spec('lingua') is None
from urm.backends.sparse_state_mixer import CertifiedSparseStateRoutes, SparseState, TritonSparseStateMixerBackend
from urm.compiler.semantic import DType, SparseReadTiming, SparseStateMixerSpec, SparseStateOperation
spec = SparseStateMixerSpec(1, 1, 8, 7, 0, 1, DType.FLOAT32, SparseStateOperation.READ_ONLY, SparseReadTiming.CURRENT_STATE)
indices = torch.tensor([[[3]]], device='cuda')
weights = torch.ones((1, 1, 1), device='cuda')
routes = CertifiedSparseStateRoutes.certify(spec, indices, weights)
backend = TritonSparseStateMixerBackend(spec)
output, state = backend.execute(SparseState(torch.ones((1, 8, 7), device='cuda')), backend.prepare(routes))
torch.cuda.synchronize()
assert torch.equal(output, torch.ones_like(output))
assert state.sequence_length == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
