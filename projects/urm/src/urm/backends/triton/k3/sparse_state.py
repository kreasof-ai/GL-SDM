"""Native K3 sparse-state mixer ops (the closed SparseStateMixer v0 algebra).

The kernels consume certified partition-local routes. One program owns a
partition/value fragment and traverses tokens sequentially, making ordered
cross-token collisions structural rather than atomic or scheduler dependent.
The compiler can select a bounded BF16 training reparameterization using chunk
matrix products and triangular solves. It preserves the typed route/state ABI,
supplies all state/operand cotangents, and retains the ordered scan base.
"""

from __future__ import annotations

import inspect
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from contextlib import nullcontext
from typing import Any, Callable, ContextManager

PROFILE_RANGES = False

NATIVE_SPARSE_STATE_MIXER_NAME = "urm_native_sparse_state_mixer_v0"

# Injectable profiling hook. The backend never imports a profiler; a consumer
# (e.g. benchmarks/profiling) installs one via ``set_state_profiler``. The hook
# maps a phase name to a context manager; the default is a no-op.
_STATE_PROFILER: Callable[[str], ContextManager[Any]] | None = None


from urm.compiler.select.anchors import (
    FROZEN_V0_ENVELOPE,
    sparse_state_launch_schedule,
    sparse_state_chunked_eligible,
    sparse_state_spec_status,
)
from urm.ir.program import SparseReadTiming, SparseStateOperation, SparseStateMixerSpec
from urm.runtime.certification import (
    CertifiedSparseStateOperands,
    CertifiedSparseStateRoutes,
    SparseState,
    SparseStateSupportStatus,
    _OPERAND_CERTIFICATE,
    _dtype_name,
    native_dependencies_available,
)

def set_state_profiler(profiler: Callable[[str], ContextManager[Any]] | None) -> None:
    """Install (or clear) the state-stage profiler hook used under PROFILE_RANGES."""
    global _STATE_PROFILER
    _STATE_PROFILER = profiler

def _state_stage(phase: str) -> ContextManager[Any]:
    if _STATE_PROFILER is None:
        return nullcontext()
    return _STATE_PROFILER(phase)

@triton.jit
def _sparse_state_read_kernel(
    memory,
    read_indices,
    read_weights,
    readings,
    SEQUENCE: tl.constexpr,
    SLOTS: tl.constexpr,
    READ_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    partition = tl.program_id(0)
    token = tl.program_id(1)
    dimension = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension < VALUE_DIM
    route_base = (partition * SEQUENCE + token) * READ_WIDTH
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route in tl.static_range(READ_WIDTH):
        slot = tl.load(read_indices + route_base + route)
        weight = tl.load(read_weights + route_base + route).to(tl.float32)
        row = (partition * SLOTS + slot) * VALUE_DIM
        selected = tl.load(memory + row + dimension, mask=dimension_mask, other=0.0).to(
            tl.float32
        )
        accumulator += weight * selected
    output = (partition * SEQUENCE + token) * VALUE_DIM
    tl.store(readings + output + dimension, accumulator, mask=dimension_mask)

@triton.jit
def _sparse_state_update_kernel(
    memory,
    write_indices,
    write_weights,
    values,
    beta,
    log_decay,
    read_indices,
    read_weights,
    readings,
    saved_write_rows,
    saved_read_rows,
    SEQUENCE: tl.constexpr,
    SLOTS: tl.constexpr,
    WRITE_WIDTH: tl.constexpr,
    READ_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    READ_BEFORE_UPDATE: tl.constexpr,
    SAVE_SELECTED: tl.constexpr,
):
    partition = tl.program_id(0)
    dimension = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension < VALUE_DIM
    for token in tl.range(0, SEQUENCE):
        read_base = (partition * SEQUENCE + token) * READ_WIDTH
        output_base = (partition * SEQUENCE + token) * VALUE_DIM
        if READ_BEFORE_UPDATE:
            reading = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for route in tl.static_range(READ_WIDTH):
                slot = tl.load(read_indices + read_base + route)
                weight = tl.load(read_weights + read_base + route).to(tl.float32)
                row = (partition * SLOTS + slot) * VALUE_DIM
                selected = tl.load(
                    memory + row + dimension, mask=dimension_mask, other=0.0
                ).to(tl.float32)
                if SAVE_SELECTED:
                    tl.store(
                        saved_read_rows + (read_base + route) * VALUE_DIM + dimension,
                        selected,
                        mask=dimension_mask,
                    )
                reading += weight * selected
            tl.store(
                readings + output_base + dimension,
                reading,
                mask=dimension_mask,
            )

        write_base = (partition * SEQUENCE + token) * WRITE_WIDTH
        decay = tl.exp(tl.load(log_decay + partition * SEQUENCE + token).to(tl.float32))
        retrieved = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for route in tl.static_range(WRITE_WIDTH):
            slot = tl.load(write_indices + write_base + route)
            weight = tl.load(write_weights + write_base + route).to(tl.float32)
            row = (partition * SLOTS + slot) * VALUE_DIM
            old = tl.load(memory + row + dimension, mask=dimension_mask, other=0.0).to(
                tl.float32
            )
            if SAVE_SELECTED:
                tl.store(
                    saved_write_rows + (write_base + route) * VALUE_DIM + dimension,
                    old,
                    mask=dimension_mask,
                )
            retrieved += weight * (decay * old)
        gate = tl.load(beta + partition * SEQUENCE + token).to(tl.float32)
        value = tl.load(
            values + output_base + dimension, mask=dimension_mask, other=0.0
        ).to(tl.float32)
        delta = gate * (value - retrieved)
        for route in tl.static_range(WRITE_WIDTH):
            slot = tl.load(write_indices + write_base + route)
            weight = tl.load(write_weights + write_base + route).to(tl.float32)
            row = (partition * SLOTS + slot) * VALUE_DIM
            old = tl.load(memory + row + dimension, mask=dimension_mask, other=0.0).to(
                tl.float32
            )
            tl.store(
                memory + row + dimension,
                decay * old + weight * delta,
                mask=dimension_mask,
            )

        if not READ_BEFORE_UPDATE:
            reading = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for route in tl.static_range(READ_WIDTH):
                slot = tl.load(read_indices + read_base + route)
                weight = tl.load(read_weights + read_base + route).to(tl.float32)
                row = (partition * SLOTS + slot) * VALUE_DIM
                selected = tl.load(
                    memory + row + dimension, mask=dimension_mask, other=0.0
                ).to(tl.float32)
                if SAVE_SELECTED:
                    tl.store(
                        saved_read_rows + (read_base + route) * VALUE_DIM + dimension,
                        selected,
                        mask=dimension_mask,
                    )
                reading += weight * selected
            tl.store(
                readings + output_base + dimension,
                reading,
                mask=dimension_mask,
            )

@triton.jit
def _sparse_state_read_grad_weights_kernel(
    memory,
    read_indices,
    grad_readings,
    grad_read_weights,
    SEQUENCE: tl.constexpr,
    SLOTS: tl.constexpr,
    READ_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route_linear = tl.program_id(0)
    partition_token = route_linear // READ_WIDTH
    partition = partition_token // SEQUENCE
    dimension = tl.arange(0, BLOCK_D)
    mask = dimension < VALUE_DIM
    slot = tl.load(read_indices + route_linear)
    selected = tl.load(
        memory + (partition * SLOTS + slot) * VALUE_DIM + dimension,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad = tl.load(
        grad_readings + partition_token * VALUE_DIM + dimension,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(grad_read_weights + route_linear, tl.sum(selected * grad, axis=0))

@triton.jit
def _sparse_state_read_grad_memory_kernel(
    read_indices,
    read_weights,
    grad_readings,
    grad_memory,
    SEQUENCE: tl.constexpr,
    SLOTS: tl.constexpr,
    READ_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route_linear = tl.program_id(0)
    partition_token = route_linear // READ_WIDTH
    partition = partition_token // SEQUENCE
    dimension = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension < VALUE_DIM
    slot = tl.load(read_indices + route_linear)
    weight = tl.load(read_weights + route_linear).to(tl.float32)
    grad = tl.load(
        grad_readings + partition_token * VALUE_DIM + dimension,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        grad_memory + (partition * SLOTS + slot) * VALUE_DIM + dimension,
        weight * grad,
        mask=mask,
        sem="relaxed",
    )

@triton.jit
def _sparse_state_update_backward_kernel(
    grad_memory,
    saved_write_rows,
    saved_read_rows,
    write_indices,
    write_weights,
    values,
    beta,
    log_decay,
    read_indices,
    read_weights,
    grad_readings,
    grad_write_weights,
    grad_values,
    grad_beta,
    grad_log_decay,
    grad_read_weights,
    SEQUENCE: tl.constexpr,
    SLOTS: tl.constexpr,
    WRITE_WIDTH: tl.constexpr,
    READ_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    READ_BEFORE_UPDATE: tl.constexpr,
):
    partition = tl.program_id(0)
    dimension = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension < VALUE_DIM
    for reverse_token in tl.range(0, SEQUENCE):
        token = SEQUENCE - 1 - reverse_token
        read_base = (partition * SEQUENCE + token) * READ_WIDTH
        write_base = (partition * SEQUENCE + token) * WRITE_WIDTH
        value_base = (partition * SEQUENCE + token) * VALUE_DIM
        grad_reading = tl.load(
            grad_readings + value_base + dimension,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        if not READ_BEFORE_UPDATE:
            for route in tl.static_range(READ_WIDTH):
                slot = tl.load(read_indices + read_base + route)
                weight = tl.load(read_weights + read_base + route).to(tl.float32)
                selected = tl.load(
                    saved_read_rows + (read_base + route) * VALUE_DIM + dimension,
                    mask=dimension_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_read_weights + read_base + route,
                    tl.sum(selected * grad_reading, axis=0),
                    sem="relaxed",
                )
                memory_offset = (partition * SLOTS + slot) * VALUE_DIM + dimension
                current = tl.load(
                    grad_memory + memory_offset,
                    mask=dimension_mask,
                    other=0.0,
                )
                tl.store(
                    grad_memory + memory_offset,
                    current + weight * grad_reading,
                    mask=dimension_mask,
                )

        decay = tl.exp(tl.load(log_decay + partition * SEQUENCE + token).to(tl.float32))
        retrieved = tl.zeros((BLOCK_D,), dtype=tl.float32)
        grad_delta = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for route in tl.static_range(WRITE_WIDTH):
            slot = tl.load(write_indices + write_base + route)
            weight = tl.load(write_weights + write_base + route).to(tl.float32)
            old = tl.load(
                saved_write_rows + (write_base + route) * VALUE_DIM + dimension,
                mask=dimension_mask,
                other=0.0,
            ).to(tl.float32)
            memory_offset = (partition * SLOTS + slot) * VALUE_DIM + dimension
            grad_updated = tl.load(
                grad_memory + memory_offset,
                mask=dimension_mask,
                other=0.0,
            )
            retrieved += weight * decay * old
            grad_delta += weight * grad_updated
        gate = tl.load(beta + partition * SEQUENCE + token).to(tl.float32)
        value = tl.load(
            values + value_base + dimension,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        delta = gate * (value - retrieved)
        grad_retrieved = -gate * grad_delta
        for route in tl.static_range(WRITE_WIDTH):
            slot = tl.load(write_indices + write_base + route)
            weight = tl.load(write_weights + write_base + route).to(tl.float32)
            old = tl.load(
                saved_write_rows + (write_base + route) * VALUE_DIM + dimension,
                mask=dimension_mask,
                other=0.0,
            ).to(tl.float32)
            decayed = decay * old
            memory_offset = (partition * SLOTS + slot) * VALUE_DIM + dimension
            grad_updated = tl.load(
                grad_memory + memory_offset,
                mask=dimension_mask,
                other=0.0,
            )
            tl.atomic_add(
                grad_write_weights + write_base + route,
                tl.sum(grad_updated * delta + grad_retrieved * decayed, axis=0),
                sem="relaxed",
            )
            grad_decayed = grad_updated + weight * grad_retrieved
            tl.store(
                grad_memory + memory_offset,
                decay * grad_decayed,
                mask=dimension_mask,
            )
            tl.atomic_add(
                grad_log_decay + partition * SEQUENCE + token,
                tl.sum(grad_decayed * decayed, axis=0),
                sem="relaxed",
            )
        tl.store(
            grad_values + value_base + dimension,
            gate * grad_delta,
            mask=dimension_mask,
        )
        tl.atomic_add(
            grad_beta + partition * SEQUENCE + token,
            tl.sum(grad_delta * (value - retrieved), axis=0),
            sem="relaxed",
        )

        if READ_BEFORE_UPDATE:
            for route in tl.static_range(READ_WIDTH):
                slot = tl.load(read_indices + read_base + route)
                weight = tl.load(read_weights + read_base + route).to(tl.float32)
                selected = tl.load(
                    saved_read_rows + (read_base + route) * VALUE_DIM + dimension,
                    mask=dimension_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_read_weights + read_base + route,
                    tl.sum(selected * grad_reading, axis=0),
                    sem="relaxed",
                )
                memory_offset = (partition * SLOTS + slot) * VALUE_DIM + dimension
                current = tl.load(
                    grad_memory + memory_offset,
                    mask=dimension_mask,
                    other=0.0,
                )
                tl.store(
                    grad_memory + memory_offset,
                    current + weight * grad_reading,
                    mask=dimension_mask,
                )

def _launch_parameters(value_dim: int) -> tuple[int, int]:
    from urm.compiler.select.anchors import sparse_state_launch_parameters

    return sparse_state_launch_parameters(value_dim)

def _sparse_state_read_forward(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    parallel, sequence, read_width = read_indices.shape
    slots = memory.shape[1]
    value_dim = memory.shape[2]
    block_d, warps = _launch_parameters(value_dim)
    if out is None:
        out = torch.empty(
            (parallel, sequence, value_dim),
            device=memory.device,
            dtype=memory.dtype,
        )
    grid = (parallel, sequence, triton.cdiv(value_dim, block_d))
    _sparse_state_read_kernel[grid](
        memory,
        read_indices,
        read_weights,
        out,
        SEQUENCE=sequence,
        SLOTS=slots,
        READ_WIDTH=read_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    return out

def _sparse_state_read_backward(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    grad_readings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parallel, sequence, read_width = read_indices.shape
    slots, value_dim = memory.shape[1:]
    block_d, warps = _launch_parameters(value_dim)
    reduction_block = max(16, triton.next_power_of_2(value_dim))
    grad_weights_fp32 = torch.empty_like(read_weights, dtype=torch.float32)
    grad_memory_fp32 = torch.zeros_like(memory, dtype=torch.float32)
    grad_readings = grad_readings.contiguous()
    route_grid = (parallel * sequence * read_width,)
    _sparse_state_read_grad_weights_kernel[route_grid](
        memory,
        read_indices,
        grad_readings,
        grad_weights_fp32,
        SEQUENCE=sequence,
        SLOTS=slots,
        READ_WIDTH=read_width,
        VALUE_DIM=value_dim,
        BLOCK_D=reduction_block,
        num_warps=warps,
    )
    memory_grid = (
        parallel * sequence * read_width,
        triton.cdiv(value_dim, block_d),
    )
    _sparse_state_read_grad_memory_kernel[memory_grid](
        read_indices,
        read_weights,
        grad_readings,
        grad_memory_fp32,
        SEQUENCE=sequence,
        SLOTS=slots,
        READ_WIDTH=read_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    return grad_memory_fp32.to(memory.dtype), grad_weights_fp32.to(read_weights.dtype)

def _sparse_state_update_forward(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    read_before_update: bool,
    out: torch.Tensor | None = None,
    save_selected: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parallel, sequence, write_width = write_indices.shape
    read_width = read_indices.shape[-1]
    slots = memory.shape[1]
    value_dim = memory.shape[2]
    block_d, warps = _launch_parameters(value_dim)
    if out is None:
        out = torch.empty(
            (parallel, sequence, value_dim),
            device=memory.device,
            dtype=memory.dtype,
        )
    if save_selected:
        saved_write_rows = torch.empty(
            (parallel, sequence, write_width, value_dim),
            device=memory.device,
            dtype=memory.dtype,
        )
        saved_read_rows = torch.empty(
            (parallel, sequence, read_width, value_dim),
            device=memory.device,
            dtype=memory.dtype,
        )
    else:
        saved_write_rows = memory
        saved_read_rows = memory
    grid = (parallel, triton.cdiv(value_dim, block_d))
    _sparse_state_update_kernel[grid](
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        out,
        saved_write_rows,
        saved_read_rows,
        SEQUENCE=sequence,
        SLOTS=slots,
        WRITE_WIDTH=write_width,
        READ_WIDTH=read_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        READ_BEFORE_UPDATE=read_before_update,
        SAVE_SELECTED=save_selected,
        num_warps=warps,
    )
    return out, memory, saved_write_rows, saved_read_rows

def _sparse_state_update_backward(
    grad_readings: torch.Tensor,
    grad_final_memory: torch.Tensor,
    saved_write_rows: torch.Tensor,
    saved_read_rows: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    read_before_update: bool,
) -> tuple[torch.Tensor, ...]:
    parallel, sequence, write_width = write_indices.shape
    read_width = read_indices.shape[-1]
    slots = grad_final_memory.shape[1]
    value_dim = grad_final_memory.shape[2]
    block_d, warps = _launch_parameters(value_dim)
    grad_memory = grad_final_memory.contiguous().clone()
    grad_write_weights_fp32 = torch.zeros_like(write_weights, dtype=torch.float32)
    grad_values_fp32 = torch.empty_like(values, dtype=torch.float32)
    grad_beta_fp32 = torch.zeros_like(beta, dtype=torch.float32)
    grad_log_decay_fp32 = torch.zeros_like(log_decay, dtype=torch.float32)
    grad_read_weights_fp32 = torch.zeros_like(read_weights, dtype=torch.float32)
    grid = (parallel, triton.cdiv(value_dim, block_d))
    _sparse_state_update_backward_kernel[grid](
        grad_memory,
        saved_write_rows,
        saved_read_rows,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        grad_readings.contiguous(),
        grad_write_weights_fp32,
        grad_values_fp32,
        grad_beta_fp32,
        grad_log_decay_fp32,
        grad_read_weights_fp32,
        SEQUENCE=sequence,
        SLOTS=slots,
        WRITE_WIDTH=write_width,
        READ_WIDTH=read_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        READ_BEFORE_UPDATE=read_before_update,
        num_warps=warps,
    )
    return (
        grad_memory,
        grad_write_weights_fp32.to(write_weights.dtype),
        grad_values_fp32.to(values.dtype),
        grad_beta_fp32.to(beta.dtype),
        grad_log_decay_fp32.to(log_decay.dtype),
        grad_read_weights_fp32.to(read_weights.dtype),
    )

class _SparseStateRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, read_indices, read_weights):
        ctx.save_for_backward(memory, read_indices, read_weights)
        return _sparse_state_read_forward(memory, read_indices, read_weights)

    @staticmethod
    def backward(ctx, grad_readings):
        memory, read_indices, read_weights = ctx.saved_tensors
        grad_memory, grad_weights = _sparse_state_read_backward(
            memory, read_indices, read_weights, grad_readings
        )
        return grad_memory, None, grad_weights

class _SparseStateUpdate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        read_before_update,
    ):
        working_memory = memory.clone()
        readings, final_memory, saved_write_rows, saved_read_rows = (
            _sparse_state_update_forward(
                working_memory,
                write_indices,
                write_weights,
                values,
                beta,
                log_decay,
                read_indices,
                read_weights,
                read_before_update=read_before_update,
                save_selected=True,
            )
        )
        ctx.save_for_backward(
            saved_write_rows,
            saved_read_rows,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
            read_indices,
            read_weights,
        )
        ctx.read_before_update = read_before_update
        ctx.final_shape = tuple(memory.shape)
        ctx.memory_dtype = memory.dtype
        ctx.memory_device = memory.device
        return readings, final_memory

    @staticmethod
    def backward(ctx, grad_readings, grad_final_memory):
        if PROFILE_RANGES:
            with _state_stage("backward"):
                return _SparseStateUpdate._backward(
                    ctx, grad_readings, grad_final_memory
                )
        return _SparseStateUpdate._backward(ctx, grad_readings, grad_final_memory)

    @staticmethod
    def _backward(ctx, grad_readings, grad_final_memory):
        (
            saved_write_rows,
            saved_read_rows,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
            read_indices,
            read_weights,
        ) = ctx.saved_tensors
        if grad_readings is None:
            grad_readings = torch.zeros_like(values)
        if grad_final_memory is None:
            grad_final_memory = torch.zeros(
                ctx.final_shape, device=ctx.memory_device, dtype=ctx.memory_dtype
            )
        gradients = _sparse_state_update_backward(
            grad_readings,
            grad_final_memory,
            saved_write_rows,
            saved_read_rows,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
            read_indices,
            read_weights,
            read_before_update=ctx.read_before_update,
        )
        (
            grad_memory,
            grad_write_weights,
            grad_values,
            grad_beta,
            grad_decay,
            grad_read,
        ) = gradients
        return (
            grad_memory,
            None,
            grad_write_weights,
            grad_values,
            grad_beta,
            grad_decay,
            None,
            grad_read,
            None,
        )

def sparse_state_read(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    needs_grad = torch.is_grad_enabled() and (
        memory.requires_grad or read_weights.requires_grad
    )
    if needs_grad:
        if out is not None:
            raise ValueError("autograd execution does not accept a preallocated output")
        return _SparseStateRead.apply(memory, read_indices, read_weights)
    return _sparse_state_read_forward(memory, read_indices, read_weights, out=out)

def _bounded_state_chunk_size(write_indices, log_decay, slots, requested):
    """Bound exponential factors using actual per-slot decay, not token count.

    This one scalar synchronization is outside the compiled numerical graph.
    It chooses an equivalent schedule and never detaches a numerical operand.
    """
    p, t, w = write_indices.shape
    size = min(requested, t)
    detached = log_decay.detach().float()
    while size > 1:
        n = (t + size - 1) // size
        pad = n * size - t
        addresses = F.pad(write_indices.long(), (0, 0, 0, pad)).reshape(p, n, size * w)
        decay = F.pad(detached.expand(p, t, w), (0, 0, 0, pad)).reshape(p, n, size * w)
        sums = torch.zeros(p, n, slots, device=decay.device).scatter_add(-1, addresses, decay)
        if float(sums.abs().max()) <= 60:
            break
        size = max(1, size // 2)
    return size


def _chunked_state_update(memory, ri, rw, wi, ww, values, beta, log_decay, chunk_size,
             zero_initial_state=False, read_before_update=False):
    p, t, d = values.shape
    slots = memory.shape[1]
    c = min(chunk_size, t)
    n = (t + c - 1) // c
    pad = n * c - t
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32

    def chunks(x):
        return F.pad(x, (0, 0, 0, pad)).reshape(p, n, c, x.shape[-1])

    ri, wi = chunks(ri.long()), chunks(wi.long())
    rw, ww, v, b, g = [chunks(x.to(dtype)) for x in (rw, ww, values, beta, log_decay)]
    slot_decay = torch.zeros(p, n, c, slots, device=values.device, dtype=dtype).scatter(
        -1, wi, g.expand_as(ww))
    cumul = slot_decay.cumsum(-2)
    end = cumul[..., -1:, :]
    write_cumul = cumul.gather(-1, wi)
    read_cumul = (cumul - slot_decay if read_before_update else cumul).gather(-1, ri)
    write_end = end.expand(p, n, c, slots).gather(-1, wi)
    read_end = end.expand(p, n, c, slots).gather(-1, ri)
    dot_dtype = torch.bfloat16 if values.dtype == torch.bfloat16 else dtype

    def dense(indices, weights):
        return torch.zeros(p, n, c, slots, device=values.device, dtype=dot_dtype).scatter_add(
            -1, indices, weights.to(dot_dtype))

    # Exponentials and their saved backward tensors stay sparse. Only the
    # matmul operands expand to slot vectors, already in the compute dtype.
    plus = dense(wi, ww * (write_cumul - write_end * .5).exp())
    minus = dense(wi, ww * (write_end * .5 - write_cumul).exp())
    qplus = dense(ri, rw * (read_cumul - read_end * .5).exp())

    def mm(a, b):
        return (a.to(dot_dtype) @ b.to(dot_dtype)).to(dtype)

    strict_upper = torch.ones(c, c, device=values.device, dtype=torch.bool).triu(0)
    coupling = mm(plus, minus.transpose(-1, -2)).masked_fill(strict_upper, 0.)
    attention = mm(qplus, minus.transpose(-1, -2)).tril(-1 if read_before_update else 0)
    eye = torch.eye(c, device=values.device, dtype=dtype).expand(p, n, c, c)
    system = eye + b * coupling
    initial_w = dense(wi, ww * write_cumul.exp())
    initial_q = dense(ri, rw * read_cumul.exp())
    final_w = dense(wi, ww * (write_end - write_cumul).exp())
    state = memory.to(dtype)
    outputs = []
    for i in range(n):
        if i == 0 and zero_initial_state:
            rhs = b[:, i] * v[:, i]
            base = torch.zeros_like(v[:, i])
        else:
            retrieved = mm(initial_w[:, i], state)
            base = mm(initial_q[:, i], state)
            rhs = b[:, i] * (v[:, i] - retrieved)
        delta = torch.linalg.solve_triangular(system[:, i], rhs,
                                             upper=False, unitriangular=True)
        outputs.append(base + mm(attention[:, i], delta))
        update = mm(final_w[:, i].transpose(-1, -2), delta)
        state = update if i == 0 and zero_initial_state else (
            cumul[:, i, -1, :, None].exp() * state + update
        )
    return torch.cat(outputs, dim=1)[:, :t].to(values.dtype), state.to(memory.dtype)


# Static chunks specialize on geometry, read timing and state-gradient ownership.
# Newer Torch supports a per-call cache budget, so unrelated models do not need
# a process-global compiler setting. Older versions retain their default budget.
_chunk_compile_options = {"fullgraph": True, "dynamic": False}
if "recompile_limit" in inspect.signature(torch.compile).parameters:
    _chunk_compile_options["recompile_limit"] = 64
_compiled_chunked_state_update = torch.compile(_chunked_state_update, **_chunk_compile_options)


def _chunked_tokenwise_state_update(memory, ri, rw, wi, ww, values, beta, log_decay, read_before_update=False):
    """Stable limiting schedule for decay too strong for exponential factoring."""
    p, t, d = values.shape
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    state = memory.to(dtype)
    outputs = []
    for token in range(t):
        reads = ri[:, token].long().unsqueeze(-1).expand(p, -1, d)
        if read_before_update:
            outputs.append((state.gather(1, reads) * rw[:, token].to(dtype).unsqueeze(-1)).sum(1))
        addresses = wi[:, token].long().unsqueeze(-1).expand(p, -1, d)
        decayed = state.gather(1, addresses) * log_decay[:, token].to(dtype).exp().unsqueeze(-1)
        weight = ww[:, token].to(dtype).unsqueeze(-1)
        retrieved = (weight * decayed).sum(1)
        delta = beta[:, token].to(dtype) * (values[:, token].to(dtype) - retrieved)
        state = state.scatter(1, addresses, decayed + weight * delta.unsqueeze(1))
        if not read_before_update:
            outputs.append((state.gather(1, reads) * rw[:, token].to(dtype).unsqueeze(-1)).sum(1))
    return torch.stack(outputs, 1).to(values.dtype), state.to(memory.dtype)


def chunked_sparse_state_update(memory, read_indices, read_weights, *,
                                write_indices, write_weights, values, beta,
                                log_decay, chunk_size=256, compiled=False,
                                zero_initial_state=False, read_before_update=False):
    """Ordered reads with ordered cross-token collisions and unique writes.

    Indices must be valid partition-local routes; writes within each token must
    be unique. Gates have shape [P,T,1] and log_decay is nonpositive. FP64 inputs
    provide an exact algebraic test path; BF16 uses Tensor Core matmuls with
    FP32 solves/decay and rounds the returned readings and state to their input
    dtypes. It does not reproduce per-token BF16 storage rounding.

    zero_initial_state is an explicit lifecycle fact, valid only for a zero
    snapshot that does not require gradients. It is never inferred from values.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if zero_initial_state and memory.requires_grad:
        raise ValueError("zero_initial_state cannot discard initial-memory gradients")
    size = _bounded_state_chunk_size(write_indices, log_decay, memory.shape[1], chunk_size)
    if size == 1:
        return _chunked_tokenwise_state_update(memory, read_indices, read_weights, write_indices, write_weights,
                          values, beta, log_decay, read_before_update)
    call = _compiled_chunked_state_update if compiled else _chunked_state_update
    with torch.autocast(values.device.type, enabled=False):
        return call(memory, read_indices, read_weights, write_indices, write_weights,
                    values, beta, log_decay, size, zero_initial_state, read_before_update)


def sparse_state_update(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    read_before_update: bool,
    out: torch.Tensor | None = None,
    chunked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable = (memory, write_weights, values, beta, log_decay, read_weights)
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in differentiable
    )
    if needs_grad and out is not None:
        raise ValueError("autograd execution does not accept a preallocated output")
    # This bounded physical branch is owned by the native K3 provider. It
    # consumes the same certified routes, read timing and state cotangents.
    # Outside its numerical/workspace envelope the ordered scan stays available.
    size = 1
    if chunked and not bool(((log_decay > 0) | (beta < 0) | (beta > 1)).any()):
        size = _bounded_state_chunk_size(write_indices, log_decay, memory.shape[1], 128)
    if size >= 32:
        zero = not memory.requires_grad and not bool(torch.count_nonzero(memory))
        try:
            readings, final = chunked_sparse_state_update(memory, read_indices, read_weights,
                write_indices=write_indices, write_weights=write_weights, values=values,
                beta=beta, log_decay=log_decay, chunk_size=size, compiled=True,
                zero_initial_state=zero, read_before_update=read_before_update)
        except torch._dynamo.exc.FailOnRecompileLimitHit:
            # Cache exhaustion selects the admitted scan base; numerical/compiler
            # errors are not caught, and no eager reference substitutes for native.
            pass
        else:
            if not needs_grad:
                memory.copy_(final)
                final = memory
                if out is not None:
                    out.copy_(readings)
                    readings = out
            return readings, final
    if needs_grad:
        return _SparseStateUpdate.apply(
            memory,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
            read_indices,
            read_weights,
            read_before_update,
        )
    readings, state, _, _ = _sparse_state_update_forward(
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        read_before_update=read_before_update,
        out=out,
    )
    return readings, state

def launch_metadata(value_dim: int) -> dict[str, int]:
    block_d, warps = _launch_parameters(value_dim)
    return {"block_d": block_d, "num_warps": warps, "tokens_per_program": -1}

def _k3_runtime_spec(spec, operands):
    """Re-materialize the recipe-time batch dims from the operand shapes."""
    from dataclasses import replace as _replace

    values = operands.get("values")
    parallel, sequence = None, None
    if values is not None:
        parallel, sequence = int(values.shape[0]), int(values.shape[1])
    elif operands.get("read_addresses") is not None:
        parallel = int(operands["read_addresses"].shape[0])
        sequence = int(operands["read_addresses"].shape[1])
    return _replace(spec, parallel=parallel or spec.parallel, sequence=sequence or spec.sequence)

def sparse_delta_state(
    memory,
    read_addresses,
    read_weights,
    *,
    write_addresses=None,
    write_weights=None,
    values=None,
    beta=None,
    log_decay=None,
    spec,
    schedule=None,
):
    """Canonical K3 sparse-delta state — the native Triton implementation of the
    uniform address-index signature.

    Same operand names and shapes as the NumPy oracle and Torch reference; the
    closed :class:`SparseStateMixerSpec` supplies read timing and accumulation
    policy. Routes are certified from the descriptor; the fused state kernels
    execute the ordered update + read. Returns ``(readings, updated_memory)``.
    """

    if schedule is not None and schedule not in (
        sparse_state_launch_schedule(spec), sparse_state_launch_schedule(spec, "scan")
    ):
        raise ValueError("invalid native K3 physical schedule")
    spec = _k3_runtime_spec(spec, {
        "values": values, "read_addresses": read_addresses,
    })
    routes = CertifiedSparseStateRoutes.certify_trusted(
        spec,
        read_addresses,
        read_weights,
        write_indices=write_addresses,
        write_weights=write_weights,
    )
    backend = TritonSparseStateMixerBackend(spec, schedule=schedule)
    prepared = backend._prepare_generated_routes(
        routes, values=values, beta=beta, log_decay=log_decay,
    )
    state = SparseState(memory=memory, sequence_length=0)
    readings, new_state = backend.execute(state, prepared)
    return readings, new_state.memory

class K3NativeTritonProvider:
    name = "urm_native_sparse_state_mixer_v0"
    family = "k3"
    tier = "native"

    def decline(self, request) -> str | None:

        if not isinstance(request.descriptor, SparseStateMixerSpec):
            return "K3 providers require a closed SparseStateMixerSpec"
        import torch

        if not torch.cuda.is_available():
            return "native K3 requires CUDA"
        if request.launch_config is not None and request.launch_config not in (
            sparse_state_launch_schedule(request.descriptor),
            sparse_state_launch_schedule(request.descriptor, "scan"),
        ):
            return "invalid native K3 physical schedule"
        return None

    def execute(self, request, operands):
        readings, updated = sparse_delta_state(
            operands["memory"],
            operands["read_addresses"],
            operands["read_weights"],
            write_addresses=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
            spec=request.descriptor,
            schedule=request.launch_config,
        )
        return {"readings": readings, "updated_memory": updated}

class TritonSparseStateMixerBackend:
    """Native URM dispatch; no comparator package is imported or consulted."""

    name = NATIVE_SPARSE_STATE_MIXER_NAME

    def __init__(self, spec: SparseStateMixerSpec, *, schedule=None) -> None:
        self.spec = spec
        self.schedule = schedule or sparse_state_launch_schedule(spec)
        status = self.support_status(spec)
        status.require()

    @staticmethod
    def support_status(spec: SparseStateMixerSpec) -> SparseStateSupportStatus:
        if not native_dependencies_available():
            return SparseStateSupportStatus.no(
                "missing_dependency", "PyTorch and Triton are required"
            )
        import torch

        if not torch.cuda.is_available():
            return SparseStateSupportStatus.no(
                "unsupported_hardware", "CUDA is unavailable"
            )
        return sparse_state_spec_status(
            spec,
            device_type="cuda",
            compute_capability=torch.cuda.get_device_capability(),
        )

    def prepare(
        self,
        routes: CertifiedSparseStateRoutes,
        *,
        values: object | None = None,
        beta: object | None = None,
        log_decay: object | None = None,
    ) -> CertifiedSparseStateOperands:
        import torch

        if routes.spec != self.spec:
            raise ValueError("certified routes do not match backend semantics")
        routes.require_intact()
        supplied = (values, beta, log_decay)
        if self.spec.operation is SparseStateOperation.READ_ONLY:
            if any(item is not None for item in supplied):
                raise ValueError("read-only operation does not accept update operands")
            tensors = ()
        else:
            expected_value = (
                self.spec.parallel,
                self.spec.sequence,
                self.spec.value_dim,
            )
            if values is None or tuple(values.shape) != expected_value:
                raise ValueError(f"values must have shape {expected_value}")
            expected_scalar = (self.spec.parallel, self.spec.sequence, 1)
            if beta is None or tuple(beta.shape) != expected_scalar:
                raise ValueError(f"beta must have shape {expected_scalar}")
            if log_decay is None or tuple(log_decay.shape) != expected_scalar:
                raise ValueError(f"log_decay must have shape {expected_scalar}")
            tensors = supplied
            route_device = routes.read_indices.device
            if any(
                not tensor.is_contiguous()
                or tensor.device != route_device
                or _dtype_name(tensor) != self.spec.dtype.value
                for tensor in tensors
            ):
                raise ValueError(
                    "update operands must be contiguous and match route device/dtype"
                )
            if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
                raise ValueError("update operands must be finite")
        return CertifiedSparseStateOperands(
            self.spec,
            routes,
            values,
            beta,
            log_decay,
            tuple(tensor._version for tensor in tensors),
            _OPERAND_CERTIFICATE,
        )

    def _prepare_generated_routes(
        self,
        routes: CertifiedSparseStateRoutes,
        *,
        values: object | None = None,
        beta: object | None = None,
        log_decay: object | None = None,
    ) -> CertifiedSparseStateOperands:
        """Internal cheap bridge for operands certified by a native pipeline."""
        if routes.spec != self.spec:
            raise ValueError("generated routes do not match backend semantics")
        routes.require_intact()
        tensors = tuple(
            tensor for tensor in (values, beta, log_decay) if tensor is not None
        )
        return CertifiedSparseStateOperands(
            self.spec,
            routes,
            values,
            beta,
            log_decay,
            tuple(tensor._version for tensor in tensors),
            _OPERAND_CERTIFICATE,
        )

    def _validate_state(self, state: SparseState) -> None:
        import torch

        memory = state.memory
        expected = (
            self.spec.parallel,
            self.spec.slots_per_partition,
            self.spec.value_dim,
        )
        if tuple(memory.shape) != expected:
            raise ValueError(f"state memory must have shape {expected}")
        if not memory.is_cuda or not memory.is_contiguous():
            raise ValueError("state memory must be contiguous CUDA storage")
        if _dtype_name(memory) != self.spec.dtype.value:
            raise ValueError("state memory dtype does not match semantics")
        runtime_status = sparse_state_spec_status(
            self.spec,
            device_type=memory.device.type,
            compute_capability=torch.cuda.get_device_capability(memory.device),
        )
        runtime_status.require()
        if not isinstance(state.sequence_length, int) or state.sequence_length < 0:
            raise ValueError("state sequence length must be a non-negative integer")

    def _validate_out(
        self,
        out: object | None,
        state: SparseState,
        prepared: CertifiedSparseStateOperands,
    ) -> None:
        """Reject unsafe preallocated outputs before importing a kernel wrapper."""
        if out is None:
            return
        import torch

        expected = (self.spec.parallel, self.spec.sequence, self.spec.value_dim)
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected:
            raise ValueError(f"out must be a tensor with shape {expected}")
        if not out.is_cuda or out.device != state.memory.device:
            raise ValueError("out must be on the state CUDA device")
        if out.dtype != state.memory.dtype:
            raise ValueError("out dtype must match state dtype")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        operands = (
            state.memory,
            prepared.routes.read_indices,
            prepared.routes.read_weights,
            prepared.routes.write_indices,
            prepared.routes.write_weights,
            prepared.values,
            prepared.beta,
            prepared.log_decay,
        )
        for operand in operands:
            if operand is not None and torch._C._overlaps(out, operand):
                raise ValueError(
                    "out storage must not overlap state, routes, or operands"
                )

    def execute(
        self,
        state: SparseState,
        prepared: CertifiedSparseStateOperands,
        *,
        out: object | None = None,
    ) -> tuple[object, SparseState]:
        if prepared.spec != self.spec:
            raise ValueError("prepared operands do not match backend semantics")
        prepared.require_intact()
        self._validate_state(state)
        if state.memory.device != prepared.routes.read_indices.device:
            raise ValueError("state and routes must share one CUDA device")
        self._validate_out(out, state, prepared)

        if self.spec.operation is SparseStateOperation.READ_ONLY:
            readings = sparse_state_read(
                state.memory,
                prepared.routes.read_indices,
                prepared.routes.read_weights,
                out=out,
            )
            return readings, state
        readings, memory = sparse_state_update(
            state.memory,
            prepared.routes.write_indices,
            prepared.routes.write_weights,
            prepared.values,
            prepared.beta,
            prepared.log_decay,
            prepared.routes.read_indices,
            prepared.routes.read_weights,
            read_before_update=self.spec.read_timing is SparseReadTiming.BEFORE_UPDATE,
            out=out,
            chunked=(self.schedule["schedule_family"] == "guarded_chunked_decayed_delta"
                     and sparse_state_chunked_eligible(self.spec) and self.spec.sequence >= 32),
        )
        state.memory = memory
        state.sequence_length += self.spec.sequence
        return readings, state

    @staticmethod
    def capability() -> dict[str, object]:
        envelope = FROZEN_V0_ENVELOPE
        return {
            "name": NATIVE_SPARSE_STATE_MIXER_NAME,
            "schema_version": envelope.schema_version,
            "native_urm_lowering": True,
            "maximum_parallel": envelope.maximum_parallel,
            "maximum_sequence": envelope.maximum_sequence,
            "maximum_slots_per_partition": envelope.maximum_slots_per_partition,
            "maximum_value_dim": envelope.maximum_value_dim,
            "maximum_route_width": envelope.maximum_route_width,
            "supported_dtypes": [dtype.value for dtype in envelope.supported_dtypes],
            "supported_index_dtypes": list(envelope.supported_index_dtypes),
            "minimum_compute_capability": list(envelope.minimum_compute_capability),
        }

    def launch_schedule(self) -> dict[str, str | int]:
        return dict(self.schedule)
