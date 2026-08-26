"""Parameterized launches of the experimental fused row-scale epilogue.

These are schedule VARIANTS of the trusted prototype in
``src/urm/compiler/anchors/routed_reduction_epilogue.py``: identical math,
identical fp32 accumulation policy, but the launch configuration (BLOCK_D,
num_warps, num_stages) and the grad-value decomposition/traversal are
explicit parameters so the solver-selected schedules can be measured. The
frozen v1 kernel and the default anchor entry points are untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sched_forward_kernel(
    indices,
    weights,
    values,
    row_scale,
    output,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(0, ROUTE_WIDTH, num_stages=NUM_STAGES):
        source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
        route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
            tl.float32
        )
        gathered = tl.load(
            values + source_row * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += route_weight * gathered
    accumulator *= scale
    tl.store(output + query * VALUE_DIM + dimension_offsets, accumulator, mask=mask)


@triton.jit
def _sched_grad_weights_kernel(
    indices,
    values,
    row_scale,
    grad_output,
    grad_weights,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0)
    query = route // ROUTE_WIDTH
    route_offset = route % ROUTE_WIDTH
    source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((), dtype=tl.float32)
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        source_values = tl.load(
            values + source_row * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(source_values * (scale * output_gradient), axis=0)
    tl.store(grad_weights + route, accumulator)


@triton.jit
def _sched_grad_values_segmented_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-query decomposition; the D axis is segmented across programs."""
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    output_gradient = tl.load(
        grad_output + query * VALUE_DIM + dimension_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + route_base + route_offset)
        route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
        contribution = route_weight * scale * output_gradient
        tl.atomic_add(
            grad_values + source_row * VALUE_DIM + dimension_offsets,
            contribution,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def _sched_grad_values_fullrow_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-query decomposition; one program walks the whole D range."""
    query = tl.program_id(0)
    scale = tl.load(row_scale + query).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        for route_offset in tl.range(ROUTE_WIDTH):
            source_row = tl.load(indices + route_base + route_offset)
            route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
            contribution = route_weight * scale * output_gradient
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                contribution,
                mask=mask,
                sem="relaxed",
            )


@triton.jit
def _sched_grad_values_per_route_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-route decomposition: grid is (Q*ROUTE_WIDTH, cdiv(D, BLOCK_D))."""
    program_id = tl.program_id(0)
    query = program_id // ROUTE_WIDTH
    route_offset = program_id % ROUTE_WIDTH
    dimension_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
    route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(tl.float32)
    output_gradient = tl.load(
        grad_output + query * VALUE_DIM + dimension_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    tl.atomic_add(
        grad_values + source_row * VALUE_DIM + dimension_offsets,
        route_weight * scale * output_gradient,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _sched_grad_row_scale_kernel(
    indices,
    weights,
    values,
    grad_output,
    grad_row_scale,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query = tl.program_id(0)
    accumulator = tl.zeros((), dtype=tl.float32)
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        inner = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for route_offset in tl.range(ROUTE_WIDTH):
            source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
            route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
                tl.float32
            )
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            inner += route_weight * gathered
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(inner * output_gradient, axis=0)
    tl.store(grad_row_scale + query, accumulator)


def forward_launch(point, indices, weights, values, row_scale):
    queries, route_width = indices.shape
    value_dim = values.shape[1]
    output = torch.empty((queries, value_dim), device=values.device, dtype=values.dtype)
    grid = (queries, triton.cdiv(value_dim, point.block_d))
    handle = _sched_forward_kernel[grid](
        indices,
        weights,
        values,
        row_scale,
        output,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=point.block_d,
        NUM_STAGES=point.num_stages,
        num_warps=point.num_warps,
    )
    return output, handle


def backward_launch(point, indices, weights, values, row_scale, grad_output):
    queries, route_width = indices.shape
    sources, value_dim = values.shape
    grad_weights = torch.empty(
        weights.shape, device=weights.device, dtype=torch.float32
    )
    _sched_grad_weights_kernel[(queries * route_width,)](
        indices,
        values,
        row_scale,
        grad_output.contiguous(),
        grad_weights,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=point.block_d,
        num_warps=point.num_warps,
    )

    grad_values = torch.zeros(
        (sources, value_dim), device=values.device, dtype=torch.float32
    )
    if point.grad_values_decomposition == "per_query":
        if point.grad_values_schedule == "segmented":
            _sched_grad_values_segmented_kernel[
                (queries, triton.cdiv(value_dim, point.block_d))
            ](
                indices,
                weights,
                row_scale,
                grad_output.contiguous(),
                grad_values,
                ROUTE_WIDTH=route_width,
                VALUE_DIM=value_dim,
                BLOCK_D=point.block_d,
                num_warps=point.num_warps,
            )
        else:
            _sched_grad_values_fullrow_kernel[(queries,)](
                indices,
                weights,
                row_scale,
                grad_output.contiguous(),
                grad_values,
                ROUTE_WIDTH=route_width,
                VALUE_DIM=value_dim,
                BLOCK_D=point.block_d,
                num_warps=point.num_warps,
            )
    else:  # per_route
        _sched_grad_values_per_route_kernel[
            (queries * route_width, triton.cdiv(value_dim, point.block_d))
        ](
            indices,
            weights,
            row_scale,
            grad_output.contiguous(),
            grad_values,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=point.block_d,
            num_warps=point.num_warps,
        )

    grad_scale = torch.empty(
        row_scale.shape, device=row_scale.device, dtype=torch.float32
    )
    _sched_grad_row_scale_kernel[(queries,)](
        indices,
        weights,
        values,
        grad_output.contiguous(),
        grad_scale,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=point.block_d,
        num_warps=point.num_warps,
    )
    return grad_weights, grad_values, grad_scale


def compile_feedback_for(kernel_handle) -> dict[str, int | None]:
    """Best-effort register/shared-memory metadata from a compiled kernel."""
    feedback: dict[str, int | None] = {
        "registers_per_thread": None,
        "shared_mem_bytes": None,
    }
    try:
        feedback["registers_per_thread"] = int(getattr(kernel_handle, "n_regs", 0))
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        feedback["shared_mem_bytes"] = int(kernel_handle.metadata.shared)
    except (TypeError, ValueError, AttributeError):
        pass
    return feedback


def make_inputs(queries, route_width, sources, value_dim, dtype_name, seed=7):
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    indices = torch.randint(
        0,
        sources,
        (queries, route_width),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    weights = torch.randn(
        (queries, route_width), device="cuda", dtype=dtype, generator=generator
    )
    values = torch.randn(
        (sources, value_dim), device="cuda", dtype=dtype, generator=generator
    )
    row_scale = torch.randn((queries,), device="cuda", dtype=dtype, generator=generator)
    return indices, weights, values, row_scale


__all__ = [
    "backward_launch",
    "compile_feedback_for",
    "forward_launch",
    "make_inputs",
]
