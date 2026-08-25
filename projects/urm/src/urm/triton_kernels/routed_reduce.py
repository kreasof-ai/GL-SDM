"""Triton forward and backward kernels for routed weighted reduction.

Routed-reduction v1 semantic contract (projects/urm/docs/triton-backend.md):

    output[q, d] = sum_k(weights[q, k] * values[indices[q, k], d])

Products and reductions accumulate in fp32; outputs and input gradients are
cast back to the corresponding input dtype. All tensors are rank-2 row-major
contiguous (enforced by ``require_row_major`` upstream), so element offsets are
computed directly from the frozen shapes. The value-gradient kernel uses fp32
atomic additions so duplicate routes remain correct.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.knobs import runtime as _runtime_knobs


@triton.jit
def _routed_reduce_forward_kernel(
    indices,
    weights,
    values,
    output,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
        route_weight = tl.load(
            weights + query * ROUTE_WIDTH + route_offset
        ).to(tl.float32)
        if EVEN_D:
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets
            ).to(tl.float32)
        else:
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets,
                mask=dimension_offsets < VALUE_DIM,
                other=0.0,
            ).to(tl.float32)
        accumulator += route_weight * gathered
    if EVEN_D:
        tl.store(output + query * VALUE_DIM + dimension_offsets, accumulator)
    else:
        tl.store(
            output + query * VALUE_DIM + dimension_offsets,
            accumulator,
            mask=dimension_offsets < VALUE_DIM,
        )


@triton.jit
def _routed_reduce_grad_weights_kernel(
    indices,
    values,
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
        accumulator += tl.sum(source_values * output_gradient, axis=0)
    tl.store(grad_weights + route, accumulator)


@triton.jit
def _routed_reduce_grad_values_kernel(
    indices,
    weights,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0)
    dimension_block = tl.program_id(1)
    query = route // ROUTE_WIDTH
    route_offset = route % ROUTE_WIDTH
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < VALUE_DIM
    source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
    route_weight = tl.load(weights + route).to(tl.float32)
    output_gradient = tl.load(
        grad_output + query * VALUE_DIM + dimension_offsets,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        grad_values + source_row * VALUE_DIM + dimension_offsets,
        route_weight * output_gradient,
        mask=dimension_mask,
        sem="relaxed",
    )


@triton.jit
def _routed_reduce_grad_values_per_query_kernel(
    indices,
    weights,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    if EVEN_D:
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets
        ).to(tl.float32)
    else:
        dimension_mask = dimension_offsets < VALUE_DIM
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + route_base + route_offset)
        route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
        if EVEN_D:
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                route_weight * output_gradient,
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                route_weight * output_gradient,
                mask=dimension_mask,
                sem="relaxed",
            )


def _launch_parameters(value_dim: int, queries: int = 1) -> tuple[int, int]:
    block_d = min(256, max(32, triton.next_power_of_2(min(value_dim, 256))))
    if queries >= 1024:
        num_warps = 2
    else:
        num_warps = 4
    return block_d, num_warps


def _backward_launch_parameters(value_dim: int) -> tuple[int, int]:
    npd = triton.next_power_of_2(value_dim)
    if value_dim >= 1024:
        return max(32, min(512, npd)), 8
    if value_dim >= 128:
        return max(32, min(256, npd)), 4
    return max(32, min(256, npd)), 4


_DIRECT_LAUNCH_CACHE: dict[tuple[object, ...], object] = {}


def _launch(
    jit_fn: object,
    grid: tuple[int, ...],
    tensors: tuple[torch.Tensor, ...],
    constexprs: dict[str, int],
    num_warps: int,
) -> None:
    if _runtime_knobs.launch_enter_hook is not None or _runtime_knobs.launch_exit_hook is not None:
        jit_fn[grid](*tensors, **constexprs, num_warps=num_warps)
        return
    launch_grid = grid + (1,) * (3 - len(grid))
    device = tensors[-1].device.index if tensors[-1].device.index is not None else torch.cuda.current_device()
    key = (
        jit_fn,
        device,
        launch_grid,
        num_warps,
        tuple(constexprs.items()),
        tuple(tensor.dtype for tensor in tensors),
        tuple(tensor.data_ptr() % 16 == 0 for tensor in tensors),
    )
    runner = _DIRECT_LAUNCH_CACHE.get(key)
    if runner is None:
        with torch.cuda.device(device):
            kernel = jit_fn.warmup(*tensors, grid=launch_grid, **constexprs, num_warps=num_warps)
            runner = kernel[launch_grid]
        _DIRECT_LAUNCH_CACHE[key] = runner
    runner(*tensors, *constexprs.values())


def _forward(indices: torch.Tensor, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    queries, route_width = indices.shape
    value_dim = values.shape[1]
    block_d, num_warps = _launch_parameters(value_dim, queries)
    output = torch.empty(
        (queries, value_dim), device=values.device, dtype=values.dtype
    )
    grid = (queries, triton.cdiv(value_dim, block_d))
    _launch(
        _routed_reduce_forward_kernel,
        grid,
        (indices, weights, values, output),
        {
            "ROUTE_WIDTH": route_width,
            "VALUE_DIM": value_dim,
            "BLOCK_D": block_d,
            "EVEN_D": value_dim % block_d == 0,
        },
        num_warps,
    )
    return output


def _backward(
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries, route_width = indices.shape
    sources, value_dim = values.shape
    block_d, num_warps = _backward_launch_parameters(value_dim)
    grad_output = grad_output.contiguous()
    grad_weights_fp32 = torch.empty(
        weights.shape, device=weights.device, dtype=torch.float32
    )
    grad_values_fp32 = torch.zeros(
        (sources, value_dim), device=values.device, dtype=torch.float32
    )

    route_grid = (queries * route_width,)
    _launch(
        _routed_reduce_grad_weights_kernel,
        route_grid,
        (indices, values, grad_output, grad_weights_fp32),
        {
            "ROUTE_WIDTH": route_width,
            "VALUE_DIM": value_dim,
            "BLOCK_D": block_d,
        },
        num_warps,
    )

    if queries >= 1024:
        value_grid = (queries, triton.cdiv(value_dim, block_d))
        _launch(
            _routed_reduce_grad_values_per_query_kernel,
            value_grid,
            (indices, weights, grad_output, grad_values_fp32),
            {
                "ROUTE_WIDTH": route_width,
                "VALUE_DIM": value_dim,
                "BLOCK_D": block_d,
                "EVEN_D": value_dim % block_d == 0,
            },
            num_warps,
        )
    else:
        value_grid = (queries * route_width, triton.cdiv(value_dim, block_d))
        _launch(
            _routed_reduce_grad_values_kernel,
            value_grid,
            (indices, weights, grad_output, grad_values_fp32),
            {
                "ROUTE_WIDTH": route_width,
                "VALUE_DIM": value_dim,
                "BLOCK_D": block_d,
            },
            num_warps,
        )
    return grad_weights_fp32.to(weights.dtype), grad_values_fp32.to(values.dtype)


class _RoutedReduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, values):
        ctx.save_for_backward(indices, weights, values)
        return _forward(indices, weights, values)

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights, values = ctx.saved_tensors
        grad_weights, grad_values = _backward(
            indices, weights, values, grad_output
        )
        return None, grad_weights, grad_values


def routed_reduce(
    indices: torch.Tensor, weights: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    if (
        torch.is_grad_enabled()
        and (indices.requires_grad or weights.requires_grad or values.requires_grad)
    ):
        return _RoutedReduction.apply(indices, weights, values)
    return _forward(indices, weights, values)


def launch_metadata(route_width: int, value_dim: int, queries: int = 1) -> dict[str, int]:
    block_d, num_warps = _launch_parameters(value_dim, queries)
    return {
        "routes_per_program": route_width,
        "block_d": block_d,
        "num_warps": num_warps,
    }
