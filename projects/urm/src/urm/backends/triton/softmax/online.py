"""Native tiled online-softmax attention with recomputed Triton backward."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_forward_tiled(
    Q,
    K,
    V,
    MASK,
    BIAS,
    OUTPUT,
    LOGSUMEXP,
    B: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    MASK_SB: tl.constexpr,
    MASK_SH: tl.constexpr,
    MASK_SQ: tl.constexpr,
    MASK_SK: tl.constexpr,
    BIAS_SB: tl.constexpr,
    BIAS_SH: tl.constexpr,
    BIAS_SQ: tl.constexpr,
    BIAS_SK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    query = tl.load(
        Q
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
        + key_dims[None, :],
        query_valid[:, None] & (key_dims[None, :] < D),
        other=0.0,
    )
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

    # Causal early termination: key blocks beyond this query block's diagonal are
    # fully masked, so skip them. The last visible key index for the last query in
    # this block is (query_start + BLOCK_M - 1) + (TK - TQ); the loop bound is the
    # first key block past it.
    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_limit = tl.minimum(last_visible + 1, TK)
        key_blocks = tl.cdiv(key_limit, BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)
    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        key = tl.load(
            K
            + ((batch * TK + keys[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(
            query,
            tl.trans(key),
            input_precision="ieee" if INPUT_FP32 else "tf32",
        ) * SCALE
        score_valid = query_valid[:, None] & key_valid[None, :]
        scores = tl.where(score_valid, scores, float("-inf"))
        if HAS_MASK:
            mask_offset = (
                batch * MASK_SB
                + query_head * MASK_SH
                + query_offsets[:, None] * MASK_SQ
                + keys[None, :] * MASK_SK
            )
            if MASK_IS_BOOL:
                mask_values = tl.load(
                    MASK + mask_offset, score_valid, other=0
                )
            else:
                mask_values = tl.load(
                    MASK + mask_offset, score_valid, other=0.0
                )
                scores += mask_values.to(tl.float32)
        if HAS_BIAS:
            bias_offset = (
                batch * BIAS_SB
                + query_head * BIAS_SH
                + query_offsets[:, None] * BIAS_SQ
                + keys[None, :] * BIAS_SK
            )
            bias_values = tl.load(
                BIAS + bias_offset, score_valid, other=0.0
            )
            scores += bias_values.to(tl.float32)
        if HAS_MASK and MASK_IS_BOOL:
            scores = tl.where(mask_values, scores, float("-inf"))
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            scores = tl.where(visible, scores, float("-inf"))
        scores = tl.where(score_valid, scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.where(
            running_sum > 0.0, tl.exp(running_max - next_max), 0.0
        )
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp(scores - safe_max[:, None])
        )
        values = tl.load(
            V
            + ((batch * TK + keys[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        weighted_values = tl.dot(
            probabilities.to(values.dtype),
            values,
            input_precision="ieee" if INPUT_FP32 else "tf32",
        )
        running_value = running_value * old_scale[:, None] + weighted_values
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = next_max

    output = tl.where(
        running_sum[:, None] > 0.0,
        running_value / tl.maximum(running_sum[:, None], 1.0e-30),
        0.0,
    )
    output_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset,
        output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )
    logsumexp = tl.where(
        running_sum > 0.0,
        running_max + tl.log(tl.maximum(running_sum, 1.0e-30)),
        float("-inf"),
    )
    tl.store(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        logsumexp,
        query_valid,
    )


@triton.jit
def _online_softmax_backward_tiled(
    Q,
    K,
    V,
    MASK,
    BIAS,
    OUTPUT,
    LOGSUMEXP,
    GRAD_OUTPUT,
    GRAD_Q,
    GRAD_K,
    GRAD_V,
    GRAD_MASK,
    GRAD_BIAS,
    B: tl.constexpr,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    MASK_SB: tl.constexpr,
    MASK_SH: tl.constexpr,
    MASK_SQ: tl.constexpr,
    MASK_SK: tl.constexpr,
    BIAS_SB: tl.constexpr,
    BIAS_SH: tl.constexpr,
    BIAS_SQ: tl.constexpr,
    BIAS_SK: tl.constexpr,
    GRAD_MASK_SB: tl.constexpr,
    GRAD_MASK_SH: tl.constexpr,
    GRAD_MASK_SQ: tl.constexpr,
    GRAD_MASK_SK: tl.constexpr,
    GRAD_BIAS_SB: tl.constexpr,
    GRAD_BIAS_SH: tl.constexpr,
    GRAD_BIAS_SQ: tl.constexpr,
    GRAD_BIAS_SK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    NEED_GRAD_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NEED_GRAD_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    query = tl.load(
        Q
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
        + key_dims[None, :],
        query_valid[:, None] & (key_dims[None, :] < D),
        other=0.0,
    )
    grad_output = tl.load(
        GRAD_OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    output = tl.load(
        OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    logsumexp = tl.load(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        query_valid,
        other=float("-inf"),
    )
    row_valid = logsumexp != float("-inf")
    safe_logsumexp = tl.where(row_valid, logsumexp, 0.0)
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    # Causal early termination (same as the forward): skip fully-masked key blocks.
    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_blocks = tl.cdiv(tl.minimum(last_visible + 1, TK), BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)
    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        key = tl.load(
            K
            + ((batch * TK + keys[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        value = tl.load(
            V
            + ((batch * TK + keys[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        scores = tl.dot(
            query,
            tl.trans(key),
            input_precision="ieee" if INPUT_FP32 else "tf32",
        ) * SCALE
        score_valid = query_valid[:, None] & key_valid[None, :]
        scores = tl.where(score_valid, scores, float("-inf"))
        if HAS_MASK:
            mask_offset = (
                batch * MASK_SB
                + query_head * MASK_SH
                + query_offsets[:, None] * MASK_SQ
                + keys[None, :] * MASK_SK
            )
            if MASK_IS_BOOL:
                mask_values = tl.load(
                    MASK + mask_offset, score_valid, other=0
                )
            else:
                mask_values = tl.load(
                    MASK + mask_offset, score_valid, other=0.0
                )
                scores += mask_values.to(tl.float32)
        if HAS_BIAS:
            bias_offset = (
                batch * BIAS_SB
                + query_head * BIAS_SH
                + query_offsets[:, None] * BIAS_SQ
                + keys[None, :] * BIAS_SK
            )
            bias_values = tl.load(
                BIAS + bias_offset, score_valid, other=0.0
            )
            scores += bias_values.to(tl.float32)
        if HAS_MASK and MASK_IS_BOOL:
            scores = tl.where(mask_values, scores, float("-inf"))
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            scores = tl.where(visible, scores, float("-inf"))
        scores = tl.where(score_valid, scores, float("-inf"))
        probabilities = tl.where(
            scores == float("-inf"),
            0.0,
            tl.exp(scores - safe_logsumexp[:, None]),
        )

        grad_probabilities = tl.dot(
            grad_output,
            tl.trans(value),
            input_precision="ieee" if INPUT_FP32 else "tf32",
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_scores = tl.where(row_valid[:, None] & score_valid, grad_scores, 0.0)
        grad_scores_native = grad_scores.to(query.dtype)
        grad_query += tl.dot(
            grad_scores_native,
            key,
            input_precision="ieee" if INPUT_FP32 else "tf32",
        ) * SCALE
        grad_key = tl.dot(
            tl.trans(grad_scores_native),
            query,
            input_precision="ieee" if INPUT_FP32 else "tf32",
        ) * SCALE
        grad_value = tl.dot(
            tl.trans(probabilities.to(query.dtype)),
            grad_output,
            input_precision="ieee" if INPUT_FP32 else "tf32",
        )
        grad_key_offset = (
            (batch * TK + keys[:, None]) * HK + key_head
        ) * D + key_dims[None, :]
        grad_value_offset = (
            (batch * TK + keys[:, None]) * HK + key_head
        ) * DV + value_dims[None, :]
        tl.atomic_add(
            GRAD_K + grad_key_offset,
            grad_key,
            key_valid[:, None] & (key_dims[None, :] < D),
            sem="relaxed",
        )
        tl.atomic_add(
            GRAD_V + grad_value_offset,
            grad_value,
            key_valid[:, None] & (value_dims[None, :] < DV),
            sem="relaxed",
        )
        if NEED_GRAD_MASK:
            grad_mask_offset = (
                batch * GRAD_MASK_SB
                + query_head * GRAD_MASK_SH
                + query_offsets[:, None] * GRAD_MASK_SQ
                + keys[None, :] * GRAD_MASK_SK
            )
            tl.atomic_add(
                GRAD_MASK + grad_mask_offset,
                grad_scores,
                score_valid,
                sem="relaxed",
            )
        if NEED_GRAD_BIAS:
            grad_bias_offset = (
                batch * GRAD_BIAS_SB
                + query_head * GRAD_BIAS_SH
                + query_offsets[:, None] * GRAD_BIAS_SQ
                + keys[None, :] * GRAD_BIAS_SK
            )
            tl.atomic_add(
                GRAD_BIAS + grad_bias_offset,
                grad_scores,
                score_valid,
                sem="relaxed",
            )

    grad_query_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * D + key_dims[None, :]
    tl.store(
        GRAD_Q + grad_query_offset,
        grad_query,
        query_valid[:, None] & (key_dims[None, :] < D),
    )


def _broadcast_strides_4d(tensor: Any | None, target: tuple[int, int, int, int]):
    if tensor is None:
        return (0, 0, 0, 0)
    if tensor.ndim not in (2, 3, 4):
        raise ValueError("attention_mask and score_bias must have rank 2, 3, or 4")
    if tensor.ndim == 2:
        shape = (1, 1, *tensor.shape)
        strides = (0, 0, *tensor.stride())
    elif tensor.ndim == 3:
        shape = (tensor.shape[0], 1, *tensor.shape[1:])
        strides = (tensor.stride(0), 0, *tensor.stride()[1:])
    else:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
    effective = []
    for size, wanted, stride in zip(shape, target, strides, strict=True):
        if size not in (1, wanted):
            raise ValueError(
                "attention_mask and score_bias must broadcast to [B,Hq,Tq,Tk]"
            )
        effective.append(0 if size == 1 else stride)
    return tuple(effective)


def _gradient_strides_4d(tensor: Any | None):
    """Strides for accumulating a gradient that broadcast in the forward pass.

    Forward reads broadcast singleton dimensions with a zero stride. The
    gradient of a broadcast dimension must reduce (sum) every broadcast
    contribution back into that single element, so its write stride must also
    be zero. Preserving the raw stride would index past the allocation and
    scatter contributions instead of accumulating them.
    """
    if tensor is None:
        return (0, 0, 0, 0)
    if tensor.ndim == 2:
        shape = (1, 1, *tensor.shape)
        strides = (0, 0, *tensor.stride())
    elif tensor.ndim == 3:
        shape = (tensor.shape[0], 1, *tensor.shape[1:])
        strides = (tensor.stride(0), 0, *tensor.stride()[1:])
    else:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
    return tuple(0 if size == 1 else stride for size, stride in zip(shape, strides, strict=True))


def execute_online_softmax(
    query: Any,
    key: Any,
    value: Any,
    *,
    attention_mask: Any | None,
    score_bias: Any | None,
    causal: bool,
    scale: float,
) -> Any:
    """Run native K1 without allocating a query-by-key score tensor."""
    if query.device.type != "cuda":
        raise ValueError("URM-native K1 requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("K1 query/key/value use BTHD rank-4 layout")
    batch, query_length, query_heads, key_dim = query.shape
    batch_k, key_length, key_heads, key_dim_k = key.shape
    value_dim = value.shape[-1]
    if min(batch, query_length, query_heads, key_dim, key_length, key_heads, value_dim) <= 0:
        raise ValueError("K1 dimensions must be positive")
    if key_dim > 128 or value_dim > 128:
        raise ValueError("native K1 currently supports key and value widths up to 128")
    if (batch, key_dim) != (batch_k, key_dim_k) or value.shape[:3] != (
        batch,
        key_length,
        key_heads,
    ):
        raise ValueError("K1 query/key/value dimensions do not agree")
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (query.dtype == key.dtype == value.dtype) or not query.is_floating_point():
        raise ValueError("K1 query/key/value must use one floating-point dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native K1 supports float16, bfloat16, and float32")
    if not (query.device == key.device == value.device):
        raise ValueError("K1 query/key/value must share one CUDA device")
    if attention_mask is not None:
        if attention_mask.device != query.device:
            raise ValueError("attention_mask must share the query device")
        if attention_mask.dtype is not torch.bool and not attention_mask.is_floating_point():
            raise ValueError("attention_mask must be boolean or floating point")
        if attention_mask.is_floating_point() and attention_mask.dtype is not torch.float32:
            raise ValueError("native K1 additive attention_mask gradients require float32")
    if score_bias is not None:
        if score_bias.device != query.device or score_bias.dtype is not torch.float32:
            raise ValueError("native K1 score_bias must be float32 on the query device")
    target = (batch, query_heads, query_length, key_length)
    mask_strides = _broadcast_strides_4d(attention_mask, target)
    bias_strides = _broadcast_strides_4d(score_bias, target)
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    mask = attention_mask if attention_mask is not None else torch.empty((0,), device=query.device)
    bias = score_bias if score_bias is not None else torch.empty((0,), device=query.device)
    # Block sizes tuned against the competitive fused-attention comparator (SDPA).
    # block_m=128 is the key lever: the previous block_m=16 underutilized the
    # tensor cores (12x slower); 128 brings the kernel within ~1.7x. Larger query
    # tiles amortize the online-softmax state and improve matmul efficiency.
    block_m = 128 if query_length >= 128 else max(16, triton.next_power_of_2(query_length))
    block_n = 64
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    num_warps = 8 if block_m >= 128 else 4
    num_stages = 3

    class _OnlineSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, mask_tensor, bias_tensor):
            output = torch.empty(
                (batch, query_length, query_heads, value_dim),
                device=q.device,
                dtype=q.dtype,
            )
            logsumexp = torch.empty(
                (batch, query_heads, query_length), device=q.device, dtype=torch.float32
            )
            _online_softmax_forward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                mask_tensor,
                bias_tensor,
                output,
                logsumexp,
                batch,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                *mask_strides,
                *bias_strides,
                attention_mask is not None,
                attention_mask is not None and attention_mask.dtype is torch.bool,
                score_bias is not None,
                causal,
                scale,
                q.dtype is torch.float32,
                block_m,
                block_n,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            ctx.save_for_backward(q, k, v, mask_tensor, bias_tensor, output, logsumexp)
            ctx.needs_mask_grad = (
                attention_mask is not None
                and attention_mask.is_floating_point()
                and attention_mask.requires_grad
            )
            ctx.needs_bias_grad = score_bias is not None and score_bias.requires_grad
            ctx.has_mask = attention_mask is not None
            ctx.mask_is_bool = attention_mask is not None and attention_mask.dtype is torch.bool
            ctx.has_bias = score_bias is not None
            ctx.causal = causal
            ctx.scale = scale
            ctx.mask_strides = mask_strides
            ctx.bias_strides = bias_strides
            ctx.mask_shape = None if attention_mask is None else tuple(attention_mask.shape)
            ctx.bias_shape = None if score_bias is None else tuple(score_bias.shape)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, mask_tensor, bias_tensor, output, logsumexp = ctx.saved_tensors
            if grad_output is None:
                return None, None, None, None, None
            grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            grad_k = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            grad_v = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            grad_mask = (
                torch.zeros(ctx.mask_shape, device=q.device, dtype=torch.float32)
                if ctx.needs_mask_grad
                else torch.empty((0,), device=q.device)
            )
            grad_bias = (
                torch.zeros(ctx.bias_shape, device=q.device, dtype=torch.float32)
                if ctx.needs_bias_grad
                else torch.empty((0,), device=q.device)
            )
            grad_mask_strides = _gradient_strides_4d(grad_mask if ctx.needs_mask_grad else None)
            grad_bias_strides = _gradient_strides_4d(grad_bias if ctx.needs_bias_grad else None)
            _online_softmax_backward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                mask_tensor,
                bias_tensor,
                output,
                logsumexp,
                grad_output.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                grad_mask,
                grad_bias,
                batch,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                *ctx.mask_strides,
                *ctx.bias_strides,
                *grad_mask_strides,
                *grad_bias_strides,
                ctx.has_mask,
                ctx.mask_is_bool,
                ctx.needs_mask_grad,
                ctx.has_bias,
                ctx.needs_bias_grad,
                ctx.causal,
                ctx.scale,
                q.dtype is torch.float32,
                block_m,
                block_n,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return (
                grad_q.to(q.dtype),
                grad_k.to(k.dtype),
                grad_v.to(v.dtype),
                grad_mask if ctx.needs_mask_grad else None,
                grad_bias if ctx.needs_bias_grad else None,
            )

    return _OnlineSoftmax.apply(query, key, value, mask, bias)


__all__ = ["execute_online_softmax"]
