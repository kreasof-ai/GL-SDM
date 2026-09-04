"""URM-owned Triton lowering for constrained pairwise sparse route selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_float_key(value):
    """Map finite fp32 values to monotonically ordered unsigned integers."""
    bits = value.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000).to(tl.uint64)


@triton.jit
def _sparse_route_forward_kernel(
    scores,
    addresses,
    weights,
    score_stride,
    route_stride,
    HALF: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_ROUTE: tl.constexpr,
    BLOCK_PAIR: tl.constexpr,
):
    row = tl.program_id(0)
    half_offsets = tl.arange(0, BLOCK_HALF)
    valid_half = half_offsets < HALF
    left = tl.load(
        scores + row * score_stride + half_offsets,
        mask=valid_half,
        other=-float("inf"),
    ).to(tl.float32)
    right = tl.load(
        scores + row * score_stride + HALF + half_offsets,
        mask=valid_half,
        other=-float("inf"),
    ).to(tl.float32)

    left_encoded = (_ordered_float_key(left) << 32) | half_offsets.to(tl.uint64)
    right_encoded = (_ordered_float_key(right) << 32) | half_offsets.to(tl.uint64)
    left_top = tl.topk(left_encoded, BLOCK_ROUTE)
    right_top = tl.topk(right_encoded, BLOCK_ROUTE)
    left_indices = (left_top & 0xFFFFFFFF).to(tl.int32)
    right_indices = (right_top & 0xFFFFFFFF).to(tl.int32)

    left_grid = tl.reshape(left_indices, (BLOCK_ROUTE, 1))
    right_grid = tl.reshape(right_indices, (1, BLOCK_ROUTE))
    pair_addresses = left_grid * HALF + right_grid
    pair_scores = (
        (
            tl.load(scores + row * score_stride + left_grid).to(tl.float32)
            + tl.load(scores + row * score_stride + HALF + right_grid).to(tl.float32)
        )
        .to(scores.dtype.element_ty)
        .to(tl.float32)
    )
    pair_offsets = tl.arange(0, BLOCK_PAIR)
    pair_addresses = tl.reshape(pair_addresses, (BLOCK_PAIR,))
    pair_scores = tl.reshape(pair_scores, (BLOCK_PAIR,))
    pair_valid = (pair_offsets // BLOCK_ROUTE < WIDTH) & (
        pair_offsets % BLOCK_ROUTE < WIDTH
    )
    pair_scores = tl.where(pair_valid, pair_scores, -float("inf"))
    pair_encoded = (_ordered_float_key(pair_scores) << 32) | pair_addresses.to(
        tl.uint64
    )
    selected = tl.topk(pair_encoded, BLOCK_ROUTE)
    selected_addresses = (selected & 0xFFFFFFFF).to(tl.int32)

    route_offsets = tl.arange(0, BLOCK_ROUTE)
    canonical_input = tl.where(route_offsets < WIDTH, selected_addresses, 0x7FFFFFFF)
    canonical_addresses = tl.sort(canonical_input, descending=False)
    selected_scores = (
        (
            tl.load(
                scores + row * score_stride + canonical_addresses // HALF,
                mask=route_offsets < WIDTH,
                other=-float("inf"),
            ).to(tl.float32)
            + tl.load(
                scores + row * score_stride + HALF + canonical_addresses % HALF,
                mask=route_offsets < WIDTH,
                other=-float("inf"),
            ).to(tl.float32)
        )
        .to(scores.dtype.element_ty)
        .to(tl.float32)
    )
    maximum = tl.max(selected_scores, axis=0)
    exponentials = tl.exp(selected_scores - maximum)
    denominator = tl.sum(exponentials, axis=0)
    normalized = exponentials / denominator
    route_ptr = row * route_stride + route_offsets
    tl.store(addresses + route_ptr, canonical_addresses, mask=route_offsets < WIDTH)
    tl.store(weights + route_ptr, normalized, mask=route_offsets < WIDTH)


@triton.jit
def _sparse_route_backward_kernel(
    addresses,
    weights,
    grad_weights,
    grad_scores,
    score_stride,
    route_stride,
    HALF: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_ROUTE: tl.constexpr,
):
    row = tl.program_id(0)
    route_offsets = tl.arange(0, BLOCK_ROUTE)
    route_mask = route_offsets < WIDTH
    route_ptr = row * route_stride + route_offsets
    route_addresses = tl.load(addresses + route_ptr, mask=route_mask, other=-1)
    route_weights = tl.load(weights + route_ptr, mask=route_mask, other=0.0).to(
        tl.float32
    )
    incoming = tl.load(grad_weights + route_ptr, mask=route_mask, other=0.0).to(
        tl.float32
    )
    score_gradient = route_weights * (
        incoming - tl.sum(incoming * route_weights, axis=0)
    )

    score_offsets = tl.arange(0, BLOCK_HALF)
    score_grid = tl.reshape(score_offsets, (BLOCK_HALF, 1))
    address_grid = tl.reshape(route_addresses, (1, BLOCK_ROUTE))
    gradient_grid = tl.reshape(score_gradient, (1, BLOCK_ROUTE))
    valid_grid = tl.reshape(route_mask, (1, BLOCK_ROUTE))
    left = tl.sum(
        tl.where(valid_grid & (address_grid // HALF == score_grid), gradient_grid, 0.0),
        axis=1,
    )
    right = tl.sum(
        tl.where(valid_grid & (address_grid % HALF == score_grid), gradient_grid, 0.0),
        axis=1,
    )
    score_mask = score_offsets < HALF
    tl.store(
        grad_scores + row * score_stride + score_offsets,
        left,
        mask=score_mask,
    )
    tl.store(
        grad_scores + row * score_stride + HALF + score_offsets,
        right,
        mask=score_mask,
    )


def _blocks(half: int, width: int) -> tuple[int, int, int]:
    block_half = triton.next_power_of_2(half)
    # Triton 3.4's bitonic top-k implementation does not accept k=1.
    block_route = max(2, triton.next_power_of_2(width))
    return block_half, block_route, block_route * block_route


def _route_forward(
    scores: torch.Tensor,
    source_extent: int,
    width: int,
    *,
    index_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = round(source_extent**0.5)
    block_half, block_route, block_pair = _blocks(half, width)
    addresses = torch.empty(
        (*scores.shape[:-1], width), device=scores.device, dtype=index_dtype
    )
    weights = torch.empty(
        (*scores.shape[:-1], width), device=scores.device, dtype=scores.dtype
    )
    rows = scores.shape[0] * scores.shape[1]
    _sparse_route_forward_kernel[(rows,)](
        scores,
        addresses,
        weights,
        scores.stride(1),
        addresses.stride(1),
        HALF=half,
        WIDTH=width,
        BLOCK_HALF=block_half,
        BLOCK_ROUTE=block_route,
        BLOCK_PAIR=block_pair,
        num_warps=8 if block_pair >= 1024 else 4,
        num_stages=2,
    )
    return addresses, weights


class _SparseRouteSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, source_extent, width, index_dtype):
        addresses, weights = _route_forward(
            scores, source_extent, width, index_dtype=index_dtype
        )
        ctx.save_for_backward(addresses, weights)
        ctx.source_extent = source_extent
        ctx.width = width
        return addresses, weights

    @staticmethod
    def backward(ctx, _grad_addresses, grad_weights):
        addresses, weights = ctx.saved_tensors
        if grad_weights is None:
            return (
                torch.zeros(
                    (*weights.shape[:-1], 2 * round(ctx.source_extent**0.5)),
                    device=weights.device,
                    dtype=weights.dtype,
                ),
                None,
                None,
                None,
            )
        half = round(ctx.source_extent**0.5)
        block_half, block_route, _ = _blocks(half, ctx.width)
        grad_scores = torch.empty(
            (*weights.shape[:-1], 2 * half),
            device=weights.device,
            dtype=weights.dtype,
        )
        rows = weights.shape[0] * weights.shape[1]
        _sparse_route_backward_kernel[(rows,)](
            addresses,
            weights,
            grad_weights.contiguous(),
            grad_scores,
            grad_scores.stride(1),
            addresses.stride(1),
            HALF=half,
            WIDTH=ctx.width,
            BLOCK_HALF=block_half,
            BLOCK_ROUTE=block_route,
            num_warps=4,
            num_stages=2,
        )
        return grad_scores, None, None, None


def sparse_route_selection(
    scores: torch.Tensor,
    source_extent: int,
    width: int,
    *,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate canonical partition-local addresses and normalized weights."""
    if torch.is_grad_enabled() and scores.requires_grad:
        return _SparseRouteSelection.apply(scores, source_extent, width, index_dtype)
    return _route_forward(scores, source_extent, width, index_dtype=index_dtype)


__all__ = ["sparse_route_selection"]
