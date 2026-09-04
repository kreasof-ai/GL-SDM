"""URM-owned Triton kernels for the frozen SparseStateMixer v0 algebra.

The kernels consume certified partition-local routes. One program owns a
partition/value fragment and traverses tokens sequentially, making ordered
cross-token collisions structural rather than atomic or scheduler dependent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    block_d = min(256, max(16, triton.next_power_of_2(min(value_dim, 256))))
    warps = 8 if block_d >= 256 else 4 if block_d >= 64 else 2
    return block_d, warps


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
        # Ignored by the SAVE_SELECTED specialization; passing an existing
        # tensor avoids an inference-only dummy allocation.
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
    grad_memory_fp32 = grad_final_memory.float().contiguous()
    grad_write_weights_fp32 = torch.zeros_like(write_weights, dtype=torch.float32)
    grad_values_fp32 = torch.empty_like(values, dtype=torch.float32)
    grad_beta_fp32 = torch.zeros_like(beta, dtype=torch.float32)
    grad_log_decay_fp32 = torch.zeros_like(log_decay, dtype=torch.float32)
    grad_read_weights_fp32 = torch.zeros_like(read_weights, dtype=torch.float32)
    grid = (parallel, triton.cdiv(value_dim, block_d))
    _sparse_state_update_backward_kernel[grid](
        grad_memory_fp32,
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
        grad_memory_fp32.to(grad_final_memory.dtype),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable = (memory, write_weights, values, beta, log_decay, read_weights)
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in differentiable
    )
    if needs_grad:
        if out is not None:
            raise ValueError("autograd execution does not accept a preallocated output")
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


__all__ = [
    "launch_metadata",
    "sparse_state_read",
    "sparse_state_update",
]
