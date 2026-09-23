"""Fused K1 selected-logit softmax/value reduction kernels."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(
        SCORES,
        VALUES,
        OUTPUT,
        LENGTH: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_LENGTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.arange(0, BLOCK_LENGTH)
        channel = tl.arange(0, BLOCK_WIDTH)
        slot_mask = slot < LENGTH
        channel_mask = channel < WIDTH
        logits = tl.load(SCORES + row * LENGTH + slot, slot_mask, other=-float("inf")).to(
            tl.float32
        )
        maximum = tl.max(logits, 0)
        weights = tl.exp(logits - maximum)
        weights = tl.where(slot_mask, weights, 0.0)
        probabilities = weights / tl.sum(weights, 0)
        values = tl.load(
            VALUES + row * LENGTH * WIDTH + slot[:, None] * WIDTH + channel[None, :],
            slot_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        output = tl.sum(values * probabilities[:, None], 0)
        tl.store(OUTPUT + row * WIDTH + channel, output, channel_mask)

    @triton.jit
    def backward_kernel(
        SCORES,
        VALUES,
        GRAD_OUTPUT,
        GRAD_SCORES,
        GRAD_VALUES,
        LENGTH: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_LENGTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.arange(0, BLOCK_LENGTH)
        channel = tl.arange(0, BLOCK_WIDTH)
        slot_mask = slot < LENGTH
        channel_mask = channel < WIDTH
        logits = tl.load(SCORES + row * LENGTH + slot, slot_mask, other=-float("inf")).to(
            tl.float32
        )
        maximum = tl.max(logits, 0)
        weights = tl.exp(logits - maximum)
        weights = tl.where(slot_mask, weights, 0.0)
        probabilities = weights / tl.sum(weights, 0)
        values = tl.load(
            VALUES + row * LENGTH * WIDTH + slot[:, None] * WIDTH + channel[None, :],
            slot_mask[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            GRAD_OUTPUT + row * WIDTH + channel, channel_mask, other=0.0
        ).to(tl.float32)
        output = tl.sum(values * probabilities[:, None], 0)
        grad_scores = probabilities * tl.sum(
            (values - output[None, :]) * grad_output[None, :], 1
        )
        grad_values = probabilities[:, None] * grad_output[None, :]
        tl.store(GRAD_SCORES + row * LENGTH + slot, grad_scores, slot_mask)
        tl.store(
            GRAD_VALUES + row * LENGTH * WIDTH + slot[:, None] * WIDTH + channel[None, :],
            grad_values,
            slot_mask[:, None] & channel_mask[None, :],
        )

    return triton, forward_kernel, backward_kernel


@lru_cache(maxsize=1)
def _autograd_function():
    import torch

    @staticmethod
    def forward(ctx, scores, values):
        triton, forward_kernel, _ = _kernels()
        rows, length = scores.shape
        width = values.shape[-1]
        output = torch.empty((rows, width), device=values.device, dtype=values.dtype)
        forward_kernel[(rows,)](
            scores,
            values,
            output,
            length,
            width,
            triton.next_power_of_2(length),
            triton.next_power_of_2(width),
            num_warps=4,
        )
        ctx.save_for_backward(scores, values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        import torch

        triton, _, backward_kernel = _kernels()
        scores, values = ctx.saved_tensors
        rows, length = scores.shape
        width = values.shape[-1]
        grad_scores = torch.empty_like(scores)
        grad_values = torch.empty_like(values)
        backward_kernel[(rows,)](
            scores,
            values,
            grad_output.contiguous(),
            grad_scores,
            grad_values,
            length,
            width,
            triton.next_power_of_2(length),
            triton.next_power_of_2(width),
            num_warps=4,
        )
        return grad_scores, grad_values

    return type(
        "_SelectedSoftmaxReadFunction",
        (torch.autograd.Function,),
        {"forward": forward, "backward": backward},
    )


def selected_softmax_read(scores, values):
    """Apply row-wise softmax(scores) @ values with first-order gradients."""
    if not scores.is_cuda or not values.is_cuda:
        raise RuntimeError("selected softmax read requires CUDA tensors")
    if scores.dtype != values.dtype or scores.dtype != __import__("torch").float32:
        raise TypeError("selected softmax read currently supports float32 scores and values")
    if scores.ndim != 2 or values.ndim != 3 or values.shape[:2] != scores.shape:
        raise ValueError("expected scores [rows, selected] and values [rows, selected, width]")
    rows, length = scores.shape
    width = values.shape[-1]
    if min(rows, length, width) <= 0 or length > 128 or width > 512:
        raise ValueError("selected softmax read supports lengths <= 128 and widths <= 512")
    return _autograd_function().apply(scores.contiguous(), values.contiguous())


__all__ = ["selected_softmax_read"]
