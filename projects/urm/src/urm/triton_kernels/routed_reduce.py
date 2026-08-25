"""Triton forward and backward kernels for routed weighted reduction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _routed_reduce_forward_kernel(
    indices,
    weights,
    values,
    output,
    index_stride_q,
    index_stride_k,
    weight_stride_q,
    weight_stride_k,
    value_stride_s,
    value_stride_d,
    output_stride_q,
    output_stride_d,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < VALUE_DIM
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(
            indices + query * index_stride_q + route_offset * index_stride_k
        )
        route_weight = tl.load(
            weights + query * weight_stride_q + route_offset * weight_stride_k
        ).to(tl.float32)
        gathered = tl.load(
            values
            + source_row * value_stride_s
            + dimension_offsets * value_stride_d,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += route_weight * gathered
    tl.store(
        output + query * output_stride_q + dimension_offsets * output_stride_d,
        accumulator,
        mask=dimension_mask,
    )


@triton.jit
def _routed_reduce_grad_weights_kernel(
    indices,
    values,
    grad_output,
    grad_weights,
    index_stride_q,
    index_stride_k,
    value_stride_s,
    value_stride_d,
    grad_output_stride_q,
    grad_output_stride_d,
    grad_weight_stride_q,
    grad_weight_stride_k,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0)
    query = route // ROUTE_WIDTH
    route_offset = route % ROUTE_WIDTH
    source_row = tl.load(
        indices + query * index_stride_q + route_offset * index_stride_k
    )
    accumulator = tl.zeros((1,), dtype=tl.float32)
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        source_values = tl.load(
            values + source_row * value_stride_s + dimension_offsets * value_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            grad_output
            + query * grad_output_stride_q
            + dimension_offsets * grad_output_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(source_values * output_gradient, axis=0)
    tl.store(
        grad_weights
        + query * grad_weight_stride_q
        + route_offset * grad_weight_stride_k,
        accumulator,
    )


@triton.jit
def _routed_reduce_grad_values_kernel(
    indices,
    weights,
    grad_output,
    grad_values,
    index_stride_q,
    index_stride_k,
    weight_stride_q,
    weight_stride_k,
    grad_output_stride_q,
    grad_output_stride_d,
    grad_value_stride_s,
    grad_value_stride_d,
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
    source_row = tl.load(
        indices + query * index_stride_q + route_offset * index_stride_k
    )
    route_weight = tl.load(
        weights + query * weight_stride_q + route_offset * weight_stride_k
    ).to(tl.float32)
    output_gradient = tl.load(
        grad_output
        + query * grad_output_stride_q
        + dimension_offsets * grad_output_stride_d,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        grad_values
        + source_row * grad_value_stride_s
        + dimension_offsets * grad_value_stride_d,
        route_weight * output_gradient,
        mask=dimension_mask,
    )


def _launch_parameters(value_dim: int) -> tuple[int, int]:
    block_d = min(256, max(32, triton.next_power_of_2(min(value_dim, 256))))
    num_warps = 4 if block_d <= 128 else 8
    return block_d, num_warps


def _forward(indices: torch.Tensor, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    queries, route_width = indices.shape
    value_dim = values.shape[1]
    block_d, num_warps = _launch_parameters(value_dim)
    output = torch.empty(
        (queries, value_dim), device=values.device, dtype=values.dtype
    )
    grid = (queries, triton.cdiv(value_dim, block_d))
    _routed_reduce_forward_kernel[grid](
        indices,
        weights,
        values,
        output,
        indices.stride(0),
        indices.stride(1),
        weights.stride(0),
        weights.stride(1),
        values.stride(0),
        values.stride(1),
        output.stride(0),
        output.stride(1),
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
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
    block_d, num_warps = _launch_parameters(value_dim)
    grad_output = grad_output.contiguous()
    grad_weights_fp32 = torch.empty(
        weights.shape, device=weights.device, dtype=torch.float32
    )
    grad_values_fp32 = torch.zeros(
        (sources, value_dim), device=values.device, dtype=torch.float32
    )

    route_grid = (queries * route_width,)
    _routed_reduce_grad_weights_kernel[route_grid](
        indices,
        values,
        grad_output,
        grad_weights_fp32,
        indices.stride(0),
        indices.stride(1),
        values.stride(0),
        values.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_weights_fp32.stride(0),
        grad_weights_fp32.stride(1),
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )

    value_grid = (queries * route_width, triton.cdiv(value_dim, block_d))
    _routed_reduce_grad_values_kernel[value_grid](
        indices,
        weights,
        grad_output,
        grad_values_fp32,
        indices.stride(0),
        indices.stride(1),
        weights.stride(0),
        weights.stride(1),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_values_fp32.stride(0),
        grad_values_fp32.stride(1),
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
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
    return _RoutedReduction.apply(indices, weights, values)


def launch_metadata(route_width: int, value_dim: int) -> dict[str, int]:
    block_d, num_warps = _launch_parameters(value_dim)
    return {
        "routes_per_program": route_width,
        "block_d": block_d,
        "num_warps": num_warps,
    }
