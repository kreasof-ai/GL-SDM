"""Experimental fused row-scale epilogue for routed weighted reduction.

Semantic expression (after the verified rewrite
``fold_row_scale_into_routed_reduction_epilogue``):

    output[q, d] = row_scale[q] * sum_k weights[q, k] * values[indices[q, k], d]

The per-row scale executes inside the routed-reduction mainloop's epilogue:
the fp32 accumulator is scaled while the tile is still resident, so ``base``
is never materialized as an externally visible tensor.

Contract notes:

- v1 is untouched: this module is a separate experimental anchor selected only
  when the planner requests the typed ``FINAL_SCALE_CONVERT`` visitor.
- Forward equivalence is proven against explicit references in tests.
- Backward covers ALL inputs - weights, values AND row_scale. The row-scale
  gradient needs un-scaled tiles, which are recomputed here (per the rewrite's
  ``recompute`` saved-state obligation) instead of materializing base.
- Accumulation stays fp32; outputs and gradients cast to input dtypes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION = 1


@triton.jit
def _rrs_forward_kernel(
    indices,
    weights,
    values,
    row_scale,
    output,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
        route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
            tl.float32
        )
        if EVEN_D:
            gathered = tl.load(values + source_row * VALUE_DIM + dimension_offsets).to(
                tl.float32
            )
        else:
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets,
                mask=dimension_offsets < VALUE_DIM,
                other=0.0,
            ).to(tl.float32)
        accumulator += route_weight * gathered
    # Typed epilogue: scale while the tile is resident; base never escapes.
    accumulator *= scale
    if EVEN_D:
        tl.store(output + query * VALUE_DIM + dimension_offsets, accumulator)
    else:
        tl.store(
            output + query * VALUE_DIM + dimension_offsets,
            accumulator,
            mask=dimension_offsets < VALUE_DIM,
        )


@triton.jit
def _rrs_grad_weights_kernel(
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
def _rrs_grad_values_per_query_kernel(
    indices,
    weights,
    row_scale,
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
    dimension_mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    if EVEN_D:
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets
        ).to(tl.float32)
    else:
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + route_base + route_offset)
        route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
        contribution = route_weight * scale * output_gradient
        if EVEN_D:
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                contribution,
                sem="relaxed",
            )
        else:
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                contribution,
                mask=dimension_mask,
                sem="relaxed",
            )


@triton.jit
def _rrs_grad_row_scale_kernel(
    indices,
    weights,
    values,
    grad_output,
    grad_row_scale,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """d(row_scale[q]) = sum_d grad_out[q, d] * base[q, d].

    base tiles are recomputed from (indices, weights, values) exactly as in
    the forward mainloop; they are never read back from memory.
    """
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


def _forward_launch(value_dim: int, queries: int) -> tuple[int, int]:
    """Mirror of the tuned v1 forward heuristic; owned by this prototype so
    future v1 retuning cannot silently change experimental numbers."""
    block_d = min(256, max(32, triton.next_power_of_2(min(value_dim, 256))))
    num_warps = 2 if queries >= 1024 else 4
    return block_d, num_warps


def _grad_launch(value_dim: int) -> tuple[int, int]:
    npd = triton.next_power_of_2(value_dim)
    if value_dim >= 128:
        return max(32, min(256, npd)), 4
    return max(32, min(256, npd)), 4


def _forward(
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    queries, route_width = indices.shape
    value_dim = values.shape[1]
    block_d, num_warps = _forward_launch(value_dim, queries)
    output = torch.empty((queries, value_dim), device=values.device, dtype=values.dtype)
    grid = (queries, triton.cdiv(value_dim, block_d))
    _rrs_forward_kernel[grid](
        indices,
        weights,
        values,
        row_scale,
        output,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=block_d,
        EVEN_D=value_dim % block_d == 0,
        num_warps=num_warps,
    )
    return output


class _RoutedReduceRowScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, values, row_scale):
        ctx.save_for_backward(indices, weights, values, row_scale)
        return _forward(indices, weights, values, row_scale)

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights, values, row_scale = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        queries, route_width = indices.shape
        sources, value_dim = values.shape

        gw_block_d, gw_warps = _grad_launch(value_dim)
        grad_weights_fp32 = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        _rrs_grad_weights_kernel[(queries * route_width,)](
            indices,
            values,
            row_scale,
            grad_output,
            grad_weights_fp32,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=gw_block_d,
            num_warps=gw_warps,
        )

        gv_block_d, gv_warps = _grad_launch(value_dim)
        grad_values_fp32 = torch.zeros(
            (sources, value_dim), device=values.device, dtype=torch.float32
        )
        _rrs_grad_values_per_query_kernel[
            (queries, triton.cdiv(value_dim, gv_block_d))
        ](
            indices,
            weights,
            row_scale,
            grad_output,
            grad_values_fp32,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=gv_block_d,
            EVEN_D=value_dim % gv_block_d == 0,
            num_warps=gv_warps,
        )

        ds_block_d, ds_warps = _grad_launch(value_dim)
        grad_scale_fp32 = torch.empty(
            row_scale.shape, device=row_scale.device, dtype=torch.float32
        )
        _rrs_grad_row_scale_kernel[(queries,)](
            indices,
            weights,
            values,
            grad_output,
            grad_scale_fp32,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=ds_block_d,
            num_warps=ds_warps,
        )

        return (
            None,
            grad_weights_fp32.to(weights.dtype),
            grad_values_fp32.to(values.dtype),
            grad_scale_fp32.to(row_scale.dtype),
        )


def routed_reduce_row_scale(
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
) -> torch.Tensor:
    """Fused routed reduction with a typed row-scale epilogue."""
    if torch.is_grad_enabled() and (
        indices.requires_grad
        or weights.requires_grad
        or values.requires_grad
        or row_scale.requires_grad
    ):
        return _RoutedReduceRowScale.apply(indices, weights, values, row_scale)
    return _forward(indices, weights, values, row_scale)


def routed_reduce_row_scale_metadata(
    route_width: int, value_dim: int
) -> dict[str, int]:
    block_d, num_warps = _forward_launch(value_dim, queries=1)
    return {
        "anchor_version": ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION,
        "routes_per_program": route_width,
        "block_d": block_d,
        "num_warps": num_warps,
        "epilogue": "row_scale",
    }
