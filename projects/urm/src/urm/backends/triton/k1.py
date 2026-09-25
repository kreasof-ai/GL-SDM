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
    LOG2E: tl.constexpr = 1.4426950408889634
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

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
        ) * (SCALE * LOG2E)
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
                scores += mask_values.to(tl.float32) * LOG2E
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
            scores += bias_values.to(tl.float32) * LOG2E
        if HAS_MASK and MASK_IS_BOOL:
            scores = tl.where(mask_values, scores, float("-inf"))
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            scores = tl.where(visible, scores, float("-inf"))
        scores = tl.where(score_valid, scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.where(
            running_sum > 0.0, tl.exp2(running_max - next_max), 0.0
        )
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp2(scores - safe_max[:, None])
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
        running_max + tl.log2(tl.maximum(running_sum, 1.0e-30)),
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
            tl.exp2(scores * 1.4426950408889634 - safe_logsumexp[:, None]),
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


@triton.jit
def _online_softmax_backward_delta(
    OUTPUT,
    GRAD_OUTPUT,
    DELTA,
    B: tl.constexpr,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Compute the row quantity DELTA = sum(dO * O) once (FlashAttention-2 style).

    One program per (batch, query-head, query block); DELTA is shared by the dQ
    pass and the key-parallel dK/dV pass, so it is computed once here rather than
    recomputed in both.
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    output = tl.load(
        OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    grad_output = tl.load(
        GRAD_OUTPUT
        + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    tl.store(
        DELTA + (batch * HQ + query_head) * TQ + query_offsets,
        delta,
        query_valid,
    )


@triton.jit
def _online_softmax_backward_dq(
    Q,
    K,
    V,
    MASK,
    BIAS,
    OUTPUT,
    LOGSUMEXP,
    GRAD_OUTPUT,
    DELTA,
    GRAD_Q,
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
    """Query-parallel grad_q pass (no dK/dV atomics; those use the KV-tiled pass).

    This pass owns each (query-head, query-row) pair exactly once, so it is also
    where the additive score-bias / attention-mask cotangents are accumulated:
    the probabilities it recomputes already carry the bias/mask, and the score
    cotangent ``grad_scores`` it derives is precisely ``grad_bias``/``grad_mask``
    (broadcast dims reduce through relaxed atomics, matching the single-pass
    kernel's policy).
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
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
        + tl.arange(0, BLOCK_V)[None, :],
        query_valid[:, None] & (tl.arange(0, BLOCK_V)[None, :] < DV),
        other=0.0,
    )
    logsumexp = tl.load(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        query_valid,
        other=float("-inf"),
    )
    delta = tl.load(
        DELTA + (batch * HQ + query_head) * TQ + query_offsets,
        query_valid,
        other=0.0,
    )
    row_valid = logsumexp != float("-inf")
    safe_logsumexp = tl.where(row_valid, logsumexp, 0.0)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
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
            + tl.arange(0, BLOCK_V)[None, :],
            key_valid[:, None] & (tl.arange(0, BLOCK_V)[None, :] < DV),
            other=0.0,
        )
        scores = tl.dot(
            query, tl.trans(key), input_precision="ieee" if INPUT_FP32 else "tf32"
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
            tl.exp2(scores * 1.4426950408889634 - safe_logsumexp[:, None]),
        )
        grad_probabilities = tl.dot(
            grad_output, tl.trans(value), input_precision="ieee" if INPUT_FP32 else "tf32"
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_scores = tl.where(row_valid[:, None] & score_valid, grad_scores, 0.0)
        grad_query += tl.dot(
            grad_scores.to(query.dtype), key, input_precision="ieee" if INPUT_FP32 else "tf32"
        ) * SCALE
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


@triton.jit
def _online_softmax_backward_kv_tiled(
    Q,
    K,
    V,
    MASK,
    BIAS,
    OUTPUT,
    LOGSUMEXP,
    GRAD_OUTPUT,
    DELTA,
    GRAD_K,
    GRAD_V,
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
    """Key-parallel grad_k/grad_v pass (FlashAttention-2 style, no atomics).

    One program per (batch, kv-head, key block); each owns its grad_k/grad_v tile
    and loops over the query blocks that attend to it, so no cross-program
    accumulation (atomics) is needed. This is the proven two-pass backward: the
    query-parallel pass computes grad_q, this key-parallel pass computes
    grad_k/grad_v. The bias/mask only shift the recomputed probabilities here;
    their cotangents are accumulated by the query-parallel pass (which owns each
    score exactly once), so this pass takes no grad_mask/grad_bias operands.
    """
    batch = tl.program_id(0)
    key_head = tl.program_id(1)
    key_start = tl.program_id(2) * BLOCK_N
    key_offsets = key_start + tl.arange(0, BLOCK_N)
    query_offsets = tl.arange(0, BLOCK_M)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    key_valid = key_offsets < TK
    key = tl.load(
        K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D + key_dims[None, :],
        key_valid[:, None] & (key_dims[None, :] < D), other=0.0,
    )
    value = tl.load(
        V + ((batch * TK + key_offsets[:, None]) * HK + key_head) * DV + value_dims[None, :],
        key_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
    )
    grad_key = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    grad_value = tl.zeros((BLOCK_N, BLOCK_V), tl.float32)
    if CAUSAL:
        first_query = key_start - (TK - TQ)
        first_query_block = tl.maximum(first_query // BLOCK_M, 0)
        num_query_blocks = tl.cdiv(TQ, BLOCK_M)
    else:
        first_query_block = 0
        num_query_blocks = tl.cdiv(TQ, BLOCK_M)
    for query_block in range(first_query_block, num_query_blocks):
        query_start = query_block * BLOCK_M
        qoff = query_start + query_offsets
        query_valid = qoff < TQ
        for qhead_in_group in range(HQ // HK):
            query_head = key_head * (HQ // HK) + qhead_in_group
            query = tl.load(
                Q + ((batch * TQ + qoff[:, None]) * HQ + query_head) * D + key_dims[None, :],
                query_valid[:, None] & (key_dims[None, :] < D), other=0.0,
            )
            grad_output = tl.load(
                GRAD_OUTPUT + ((batch * TQ + qoff[:, None]) * HQ + query_head) * DV + value_dims[None, :],
                query_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
            )
            logsumexp = tl.load(
                LOGSUMEXP + (batch * HQ + query_head) * TQ + qoff, query_valid, other=float("-inf")
            )
            delta = tl.load(
                DELTA + (batch * HQ + query_head) * TQ + qoff, query_valid, other=0.0
            )
            scores = tl.dot(
                key, tl.trans(query), input_precision="ieee" if INPUT_FP32 else "tf32"
            ) * SCALE
            score_valid = key_valid[:, None] & query_valid[None, :]
            if CAUSAL:
                visible = key_offsets[:, None] <= qoff[None, :] + TK - TQ
                score_valid = score_valid & visible
            scores = tl.where(score_valid, scores, float("-inf"))
            if HAS_MASK:
                mask_offset = (
                    batch * MASK_SB
                    + query_head * MASK_SH
                    + qoff[None, :] * MASK_SQ
                    + key_offsets[:, None] * MASK_SK
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
                    + qoff[None, :] * BIAS_SQ
                    + key_offsets[:, None] * BIAS_SK
                )
                bias_values = tl.load(
                    BIAS + bias_offset, score_valid, other=0.0
                )
                scores += bias_values.to(tl.float32)
            if HAS_MASK and MASK_IS_BOOL:
                scores = tl.where(mask_values, scores, float("-inf"))
            scores = tl.where(score_valid, scores, float("-inf"))
            probabilities = tl.where(
                scores == float("-inf"), 0.0, tl.exp2(scores * 1.4426950408889634 - logsumexp[None, :])
            )
            grad_probabilities = tl.dot(
                value, tl.trans(grad_output), input_precision="ieee" if INPUT_FP32 else "tf32"
            )
            grad_scores = probabilities * (grad_probabilities - delta[None, :])
            grad_scores = tl.where(score_valid, grad_scores, 0.0)
            grad_key += tl.dot(
                grad_scores.to(query.dtype), query, input_precision="ieee" if INPUT_FP32 else "tf32"
            ) * SCALE
            grad_value += tl.dot(
                probabilities.to(grad_output.dtype), grad_output,
                input_precision="ieee" if INPUT_FP32 else "tf32",
            )
    tl.store(
        GRAD_K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D + key_dims[None, :],
        grad_key, key_valid[:, None] & (key_dims[None, :] < D),
    )
    tl.store(
        GRAD_V + ((batch * TK + key_offsets[:, None]) * HK + key_head) * DV + value_dims[None, :],
        grad_value, key_valid[:, None] & (value_dims[None, :] < DV),
    )


@triton.jit
def _online_softmax_decode_kernel(
    Q,
    K,
    V,
    OUTPUT,
    S: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One fused single-query online-softmax decode step against the KV cache.

    Follows the ATMA decode-kernel pattern: one program owns one
    ``(batch, query_head)`` pair and streams the persistent KV cache once in
    ``BLOCK_N`` chunks, accumulating the online softmax (running max, running
    sum, running value) in fp32 so the ``[1, S]`` score tensor is never
    materialized. There is no autograd graph, no per-call spec construction,
    and a fixed launch shape, so the step is CUDA-graph capturable.

    Decode semantics: the single query token sits at the latest position
    ``S - 1``, so causal attention sees the full history (all ``S`` keys).
    GQA-aware: ``key_head = query_head // (HQ // HK)``.
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    key_head = query_head // (HQ // HK)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, BLOCK_N)
    query = tl.load(
        Q + (batch * HQ + query_head) * D + key_dims,
        key_dims < D,
        other=0.0,
    ).to(tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    running_max = float("-inf")
    running_sum = 0.0
    running_value = tl.zeros((BLOCK_V,), tl.float32)
    for key_start in range(0, S, BLOCK_N):
        keys = key_start + key_offsets
        key_valid = keys < S
        key = tl.load(
            K + ((batch * S + keys[:, None]) * HK + key_head) * D + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * (SCALE * LOG2E)
        scores = tl.where(key_valid, scores, float("-inf"))
        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.where(
            running_sum > 0.0, tl.exp2(running_max - next_max), 0.0
        )
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp2(scores - safe_max)
        )
        values = tl.load(
            V + ((batch * S + keys[:, None]) * HK + key_head) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        ).to(tl.float32)
        running_value = (
            running_value * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = next_max
    output = tl.where(
        running_sum > 0.0,
        running_value / tl.maximum(running_sum, 1.0e-30),
        0.0,
    )
    tl.store(
        OUTPUT + (batch * HQ + query_head) * DV + value_dims,
        output,
        value_dims < DV,
    )


@triton.jit
def _indexed_k1_forward_tiled(
    Q,
    K,
    V,
    GATHER,
    OUTPUT,
    LOGSUMEXP,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Indexed K1 gather-attend forward: softmax over a per-query gathered set.

    One program per ``(batch, query_head, query_block)``. The gather_indices
    operand ``GATHER`` is ``[B, HK, TQ, W]`` (per KV head, int32; -1 = padding →
    masked). Each query row owns W source positions; the kernel loops over the W
    slots, gathering the K/V row at each query's position (a per-row gather, no
    [BLOCK_M, W, D] tile — SMEM stays at the dense kernel's [BLOCK_M, BLOCK_D]
    tiles) and accumulating the online softmax in fp32. The route (indices) is
    external; there is no causal mask here — visibility is carried by the
    indices (-1 padding).
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
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
    LOG2E: tl.constexpr = 1.4426950408889634
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    gather_base = GATHER + ((batch * HK + key_head) * TQ + query_offsets) * W
    for w in range(0, W):
        pos = tl.load(gather_base + w, query_valid, other=-1)
        slot_valid = query_valid & (pos >= 0)
        safe_pos = tl.where(pos >= 0, pos, 0)
        key = tl.load(
            K
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            slot_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query.to(tl.float32) * key, axis=1) * (SCALE * LOG2E)
        scores = tl.where(slot_valid, scores, float("-inf"))
        next_max = tl.maximum(running_max, scores)
        old_scale = tl.where(
            running_sum > 0.0, tl.exp2(running_max - next_max), 0.0
        )
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp2(scores - safe_max)
        )
        value = tl.load(
            V
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            slot_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        ).to(tl.float32)
        running_value = (
            running_value * old_scale[:, None]
            + probabilities[:, None] * value
        )
        running_sum = running_sum * old_scale + probabilities
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
        running_max + tl.log2(tl.maximum(running_sum, 1.0e-30)),
        float("-inf"),
    )
    tl.store(
        LOGSUMEXP + (batch * HQ + query_head) * TQ + query_offsets,
        logsumexp,
        query_valid,
    )


@triton.jit
def _indexed_k1_backward_tiled(
    Q,
    K,
    V,
    GATHER,
    OUTPUT,
    LOGSUMEXP,
    GRAD_OUTPUT,
    GRAD_Q,
    GRAD_K,
    GRAD_V,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Indexed K1 gather-attend backward: dq plus a gather-scatter for dk/dv.

    One program per ``(batch, query_head, query_block)`` — it owns each query
    row exactly once, so dq is a plain store. The K/V cotangent at a gathered
    source position accumulates over every query that routed to it; that scatter
    reduces through relaxed ``tl.atomic_add`` (the K3 native backward policy, so
    cross-program accumulation order is not guaranteed). There is no cotangent
    for the integer route (gather_indices). delta = sum(dO·O) and the forward
    logsumexp recombine the softmax probabilities; the loop re-gathers the same
    K/V rows the forward read.
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    query_offsets = query_start + tl.arange(0, BLOCK_M)
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
    row_valid = query_valid & (logsumexp != float("-inf"))
    safe_logsumexp = tl.where(logsumexp != float("-inf"), logsumexp, 0.0)
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    gather_base = GATHER + ((batch * HK + key_head) * TQ + query_offsets) * W
    for w in range(0, W):
        pos = tl.load(gather_base + w, query_valid, other=-1)
        slot_valid = row_valid & (pos >= 0)
        safe_pos = tl.where(pos >= 0, pos, 0)
        key = tl.load(
            K
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :],
            slot_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            V
            + ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :],
            slot_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        ).to(tl.float32)
        query_f = query.to(tl.float32)
        grad_output_f = grad_output.to(tl.float32)
        scores = tl.sum(query_f * key, axis=1) * SCALE
        scores = tl.where(slot_valid, scores, float("-inf"))
        probabilities = tl.where(
            scores == float("-inf"),
            0.0,
            tl.exp2(scores * 1.4426950408889634 - safe_logsumexp),
        )
        grad_probabilities = tl.sum(grad_output_f * value, axis=1)
        grad_scores = probabilities * (grad_probabilities - delta)
        grad_scores = tl.where(slot_valid, grad_scores, 0.0)
        grad_query += grad_scores[:, None] * key * SCALE
        # Gather-scatter: the source at safe_pos accumulates this query's
        # contribution; multiple queries share a source, so reduce via atomics.
        grad_key = grad_scores[:, None] * query_f * SCALE
        grad_value = probabilities[:, None] * grad_output_f
        grad_key_offset = (
            ((batch * TK + safe_pos[:, None]) * HK + key_head) * D
            + key_dims[None, :]
        )
        grad_value_offset = (
            ((batch * TK + safe_pos[:, None]) * HK + key_head) * DV
            + value_dims[None, :]
        )
        tl.atomic_add(
            GRAD_K + grad_key_offset,
            grad_key,
            slot_valid[:, None] & (key_dims[None, :] < D),
            sem="relaxed",
        )
        tl.atomic_add(
            GRAD_V + grad_value_offset,
            grad_value,
            slot_valid[:, None] & (value_dims[None, :] < DV),
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


@triton.jit
def _k1_softmax_probs_kernel(
    Q,
    K,
    P,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    STRICT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Materialize one head's causal/strict-causal softmax probability matrix.

    One program per ``(batch * head, query block)``; Q/K are already expanded to
    a shared head count (GQA pre-expanded by the caller) in BTHD layout. The
    reduction mirrors the canonical core exactly: row max subtraction, exp, then
    division by the clipped row sum. ``STRICT`` excludes the diagonal (s < t);
    ``CAUSAL`` keeps it (s <= t). Fully masked rows produce zero probabilities.
    """
    head_batch = tl.program_id(0)
    query_start = tl.program_id(1) * BLOCK_M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    query_valid = query_offsets < TQ
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    for key_start in range(0, tl.cdiv(TK, BLOCK_N)):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        query = tl.load(
            Q + (head_batch * TQ + query_offsets[:, None]) * D + key_dims[None, :],
            query_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        key = tl.load(
            K + (head_batch * TK + keys[:, None]) * D + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * SCALE
        score_valid = query_valid[:, None] & key_valid[None, :]
        if STRICT:
            visible = keys[None, :] < query_offsets[:, None] + (TK - TQ)
        elif CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + (TK - TQ)
        else:
            visible = score_valid
        scores = tl.where(score_valid & visible, scores, float("-inf"))
        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.where(running_sum > 0.0, tl.exp(running_max - next_max), 0.0)
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        probabilities = tl.where(
            scores == float("-inf"), 0.0, tl.exp(scores - safe_max[:, None])
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = next_max
    safe_max = tl.where(running_max == float("-inf"), 0.0, running_max)
    denom = tl.maximum(running_sum, 1.0e-30)
    for key_start in range(0, tl.cdiv(TK, BLOCK_N)):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        query = tl.load(
            Q + (head_batch * TQ + query_offsets[:, None]) * D + key_dims[None, :],
            query_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        key = tl.load(
            K + (head_batch * TK + keys[:, None]) * D + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * SCALE
        score_valid = query_valid[:, None] & key_valid[None, :]
        if STRICT:
            visible = keys[None, :] < query_offsets[:, None] + (TK - TQ)
        elif CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + (TK - TQ)
        else:
            visible = score_valid
        scores = tl.where(score_valid & visible, scores, float("-inf"))
        probabilities = tl.where(
            scores == float("-inf"),
            0.0,
            tl.exp(scores - safe_max[:, None]) / denom[:, None],
        )
        store_offset = (head_batch * TQ + query_offsets[:, None]) * TK + keys[None, :]
        tl.store(P + store_offset, probabilities, score_valid)


@triton.jit
def _positive_feature_forward_tiled(
    Q,
    K,
    V,
    OUTPUT,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """KATA positive-feature attention: grouped squared scores, L1-normalized.

    scores[t, s] = sum_m (q_m . k_m / sqrt(group_dim))^2 over the ``NUM_GROUPS``
    head-dim groups, causal-masked to zero, then normalized by the row sum (L1,
    not softmax). One program per ``(batch * head, query block)``; Q/K/V are
    pre-expanded to a shared head count. Scores are non-negative, so no max
    subtraction is needed: accumulate the row sum and weighted value, divide once.
    """
    head_batch = tl.program_id(0)
    query_start = tl.program_id(1) * BLOCK_M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    group_dims = tl.arange(0, BLOCK_G)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
    key_blocks = tl.cdiv(tl.minimum(last_visible + 1, TK), BLOCK_N)
    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for group in range(NUM_GROUPS):
            dims = group * GROUP_DIM + group_dims
            dim_valid = group_dims < GROUP_DIM
            query = tl.load(
                Q + (head_batch * TQ + query_offsets[:, None]) * D + dims[None, :],
                query_valid[:, None] & dim_valid[None, :],
                other=0.0,
            )
            key = tl.load(
                K + (head_batch * TK + keys[:, None]) * D + dims[None, :],
                key_valid[:, None] & dim_valid[None, :],
                other=0.0,
            )
            group_scores = tl.dot(
                query, tl.trans(key), input_precision="ieee"
            ) * (GROUP_DIM ** -0.5)
            scores += group_scores * group_scores
        visible = keys[None, :] <= query_offsets[:, None] + (TK - TQ)
        score_valid = query_valid[:, None] & key_valid[None, :] & visible
        scores = tl.where(score_valid, scores, 0.0)
        values = tl.load(
            V + (head_batch * TK + keys[:, None]) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        running_value += tl.dot(scores.to(values.dtype), values, input_precision="ieee")
        running_sum += tl.sum(scores, axis=1)
    output = running_value / tl.maximum(running_sum[:, None], 1.0e-12)
    output_offset = (head_batch * TQ + query_offsets[:, None]) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset,
        output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )


@triton.jit
def _thresholded_attend_forward_tiled(
    Q,
    K,
    V,
    OUTPUT,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """TDA thresholded attention (one branch): L2-normalized Q/K, squared-relu.

    scores[t, s] = relu((qn . kn)[t, s] - beta * sqrt(2 log(t+1) / D))^2 on the
    causal-visible keys (else zero), then the unnormalized weighted value sum.
    One program per ``(batch * head, query block)``; Q/K/V are pre-expanded to a
    shared head count. There is no softmax denominator (unnormalized reduction).
    """
    head_batch = tl.program_id(0)
    query_start = tl.program_id(1) * BLOCK_M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
    query = tl.load(
        Q + (head_batch * TQ + query_offsets[:, None]) * D + key_dims[None, :],
        query_valid[:, None] & (key_dims[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    query = query / tl.sqrt(tl.sum(query * query, axis=1))[:, None]
    positions = (query_offsets + 1 + (TK - TQ)).to(tl.float32)
    threshold = BETA * tl.sqrt(2.0 * tl.log(positions) / D)
    running_value = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
    key_blocks = tl.cdiv(tl.minimum(last_visible + 1, TK), BLOCK_N)
    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        key = tl.load(
            K + (head_batch * TK + keys[:, None]) * D + key_dims[None, :],
            key_valid[:, None] & (key_dims[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        key = key / tl.sqrt(tl.sum(key * key, axis=1))[:, None]
        scores = tl.dot(query, tl.trans(key), input_precision="ieee")
        visible = keys[None, :] <= query_offsets[:, None] + (TK - TQ)
        score_valid = query_valid[:, None] & key_valid[None, :] & visible
        rectified = tl.maximum(scores - threshold[:, None], 0.0)
        weights = tl.where(score_valid, rectified * rectified, 0.0)
        values = tl.load(
            V + (head_batch * TK + keys[:, None]) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        running_value += tl.dot(weights.to(values.dtype), values, input_precision="ieee")
    output_offset = (head_batch * TQ + query_offsets[:, None]) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset,
        running_value,
        query_valid[:, None] & (value_dims[None, :] < DV),
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
    # The forward kernel loads the full channel axis in one tile (block_d covers
    # key_dim), so the A10G's 101KB SMEM caps head widths at 64 (D=128 needs 114KB,
    # measured). Decline wider heads loudly — never launch into a CUDA OOM.
    if key_dim > 64 or value_dim > 64:
        raise ValueError(
            f"native K1 online softmax supports head widths <= 64 on this device "
            f"(SMEM limit); got key_dim={key_dim}, value_dim={value_dim} — "
            f"the reference tier owns wider heads"
        )
    mask = attention_mask if attention_mask is not None else torch.empty((0,), device=query.device)
    bias = score_bias if score_bias is not None else torch.empty((0,), device=query.device)
    # The forward kernel holds [BLOCK_M, BLOCK_D] q + [BLOCK_N, BLOCK_D] k + [BLOCK_N,
    # BLOCK_V] v + [BLOCK_M, BLOCK_N] score fp32 tiles; at block_m=128 with 3-stage
    # pipelining that exceeds the A10G's 101KB SMEM (measured 131072 required at
    # T>=128, D=64). block_m=64 with stages=2 stays under the limit (measured ~74KB)
    # without a measurable throughput cost on this GPU.
    block_m = 64 if query_length >= 64 else max(16, triton.next_power_of_2(query_length))
    block_n = 64
    # block_d/block_v must cover the full head width (the kernel loads the whole channel
    # axis in one tile for tl.dot — they are NOT chunked). The A10G's SMEM therefore caps
    # the native K1 forward at head_dim <= 64 (D=128 needs 114KB > 101KB, measured); the
    # provider declines wider heads (they stay reference-tier).
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    num_warps = 8 if block_m >= 128 else 4
    num_stages = 2
    # The backward kernels hold [BLOCK_M, BLOCK_N/D/V] fp32 tiles; at block_m=128 with
    # 3-stage pipelining they exceed the A10G's 101KB SMEM limit (measured: 131072
    # required). The backward is memory-bound and the pipelining buys little — run the
    # backward at num_stages=1 and cap its block_m at 64, keeping SMEM ≈ 64×64×4 ×
    # (few tiles) comfortably under the limit at any head_dim/length.
    bwd_block_m = min(block_m, 64)
    bwd_block_n = block_n
    bwd_num_warps = num_warps if bwd_block_m >= 64 else 4
    bwd_num_stages = 1

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
            # The two-pass backward (query-parallel dq + key-parallel dk/dv, FA-2 style)
            # is the SMEM-safe path: the single-pass kernel holds too many fp32 tiles
            # for the A10G's 101KB SMEM at head_dim=64 (measured 131072 required). The
            # two-pass kernels run at block_m<=64, num_stages=1 — under the limit.
            use_two_pass = True
            grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            if use_two_pass:
                grad_k = torch.empty(k.shape, device=k.device, dtype=torch.float32)
                grad_v = torch.empty(v.shape, device=v.device, dtype=torch.float32)
            else:
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
            if use_two_pass:
                delta = torch.empty(
                    (batch, query_heads, query_length), device=q.device, dtype=torch.float32
                )
                grad_output_c = grad_output.contiguous()
                _online_softmax_backward_delta[
                    (batch, query_heads, triton.cdiv(query_length, bwd_block_m))
                ](
                    output, grad_output_c, delta,
                    batch, query_length, query_heads, value_dim,
                    bwd_block_m, block_v,
                    num_warps=4, num_stages=1,
                )
                _online_softmax_backward_dq[
                    (batch, query_heads, triton.cdiv(query_length, bwd_block_m))
                ](
                    q, k, v, mask_tensor, bias_tensor, output, logsumexp,
                    grad_output_c, delta, grad_q, grad_mask, grad_bias,
                    batch, query_length, key_length, query_heads, key_heads,
                    key_dim, value_dim,
                    *ctx.mask_strides, *ctx.bias_strides,
                    *grad_mask_strides, *grad_bias_strides,
                    ctx.has_mask, ctx.mask_is_bool, ctx.needs_mask_grad,
                    ctx.has_bias, ctx.needs_bias_grad,
                    ctx.causal, ctx.scale, q.dtype is torch.float32,
                    bwd_block_m, bwd_block_n, block_d, block_v,
                    num_warps=bwd_num_warps, num_stages=bwd_num_stages,
                )
                _online_softmax_backward_kv_tiled[
                    (batch, key_heads, triton.cdiv(key_length, bwd_block_n))
                ](
                    q, k, v, mask_tensor, bias_tensor, output, logsumexp,
                    grad_output_c, delta, grad_k, grad_v,
                    query_length, key_length, query_heads, key_heads,
                    key_dim, value_dim,
                    *ctx.mask_strides, *ctx.bias_strides,
                    ctx.has_mask, ctx.mask_is_bool, ctx.has_bias,
                    ctx.causal, ctx.scale, q.dtype is torch.float32,
                    bwd_block_m, bwd_block_n, block_d, block_v,
                    num_warps=bwd_num_warps, num_stages=bwd_num_stages,
                )
                return (
                    grad_q.to(q.dtype),
                    grad_k.to(k.dtype),
                    grad_v.to(v.dtype),
                    grad_mask if ctx.needs_mask_grad else None,
                    grad_bias if ctx.needs_bias_grad else None,
                )
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
                bwd_block_n,
                block_d,
                block_v,
                num_warps=bwd_num_warps,
                num_stages=bwd_num_stages,
            )
            return (
                grad_q.to(q.dtype),
                grad_k.to(k.dtype),
                grad_v.to(v.dtype),
                grad_mask if ctx.needs_mask_grad else None,
                grad_bias if ctx.needs_bias_grad else None,
            )

    return _OnlineSoftmax.apply(query, key, value, mask, bias)


def _check_k1_operands(query: Any, key: Any, value: Any, *, op: str) -> tuple:
    """Shared BTHD operand validation for the operation-specific K1 kernels."""
    if query.device.type != "cuda":
        raise ValueError(f"URM-native K1 {op} requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"K1 {op} query/key/value use BTHD rank-4 layout")
    batch, query_length, query_heads, key_dim = query.shape
    _, key_length, key_heads, _ = key.shape
    value_dim = value.shape[-1]
    if min(batch, query_length, query_heads, key_dim, key_length, key_heads, value_dim) <= 0:
        raise ValueError(f"K1 {op} dimensions must be positive")
    if key_dim > 128 or value_dim > 128:
        raise ValueError(f"native K1 {op} supports key and value widths up to 128")
    if not (query.dtype == key.dtype == value.dtype) or not query.is_floating_point():
        raise ValueError(f"K1 {op} query/key/value must use one floating-point dtype")
    if not (query.device == key.device == value.device):
        raise ValueError(f"K1 {op} query/key/value must share one CUDA device")
    return batch, query_length, query_heads, key_dim, key_length, value_dim


def execute_softmax_probs(
    query: Any,
    key: Any,
    *,
    causal: bool,
    strict: bool,
    scale: float,
) -> Any:
    """Materialize the causal/strict-causal softmax probability matrix P.

    ``query``/``key`` are BTHD rank-4 and must already share one head count (the
    caller pre-expands any GQA group). Returns ``P`` in ``[B, H, Tq, Tk]`` fp32,
    matching the canonical ``attention_probs`` reduction. Used by the positional
    and delta-transform K1 compositions, which need P explicitly.
    """
    value = query
    batch, query_length, query_heads, key_dim, key_length, _ = _check_k1_operands(
        query, key, value, op="softmax_probs"
    )
    if query.shape[2] != key.shape[2]:
        raise ValueError("softmax_probs expects pre-expanded (shared) head counts")
    query_c, key_c = query.contiguous(), key.contiguous()
    probs = torch.zeros(
        (batch, query_heads, query_length, key_length),
        device=query.device,
        dtype=torch.float32,
    )
    block_m = min(64, max(16, triton.next_power_of_2(query_length)))
    block_n = min(64, max(16, triton.next_power_of_2(key_length)))
    block_d = max(16, triton.next_power_of_2(key_dim))
    q_flat = query_c.permute(0, 2, 1, 3).contiguous()
    k_flat = key_c.permute(0, 2, 1, 3).contiguous()
    _k1_softmax_probs_kernel[
        (batch * query_heads, triton.cdiv(query_length, block_m))
    ](
        q_flat,
        k_flat,
        probs,
        query_length,
        key_length,
        key_dim,
        causal,
        strict,
        scale,
        block_m,
        block_n,
        block_d,
        num_warps=4,
        num_stages=2,
    )
    return probs


def execute_positive_feature(
    query: Any,
    key: Any,
    value: Any,
    *,
    num_groups: int,
) -> Any:
    """KATA positive-feature attention: grouped squared scores, L1-normalized.

    ``query``/``key``/``value`` are BTHD rank-4 with a shared head count (the
    caller pre-expands any GQA group). Always causal, mirroring the canonical
    core. Returns the output in the input dtype.
    """
    batch, query_length, query_heads, key_dim, key_length, value_dim = _check_k1_operands(
        query, key, value, op="positive_feature"
    )
    if query.shape[2] != key.shape[2]:
        raise ValueError("positive_feature expects pre-expanded (shared) head counts")
    if query.shape[1] != key.shape[1]:
        raise ValueError("positive_feature is self-attention (Tq must equal Tk)")
    if key_dim % num_groups:
        raise ValueError("positive_feature head dim must be divisible by num_groups")
    group_dim = key_dim // num_groups
    query_c, key_c, value_c = query.contiguous(), key.contiguous(), value.contiguous()
    output = torch.empty(
        (batch, query_length, query_heads, value_dim),
        device=query.device,
        dtype=query.dtype,
    )
    block_m = min(64, max(16, triton.next_power_of_2(query_length)))
    block_n = min(64, max(16, triton.next_power_of_2(key_length)))
    block_g = max(16, triton.next_power_of_2(group_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    q_flat = query_c.permute(0, 2, 1, 3).contiguous()
    k_flat = key_c.permute(0, 2, 1, 3).contiguous()
    v_flat = value_c.permute(0, 2, 1, 3).contiguous()
    out_flat = torch.empty(
        (batch, query_heads, query_length, value_dim),
        device=query.device,
        dtype=torch.float32,
    )
    _positive_feature_forward_tiled[
        (batch * query_heads, triton.cdiv(query_length, block_m))
    ](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        query_length,
        key_length,
        key_dim,
        value_dim,
        num_groups,
        group_dim,
        block_m,
        block_n,
        block_g,
        block_v,
        num_warps=4,
        num_stages=2,
    )
    output = out_flat.permute(0, 2, 1, 3).to(query.dtype)
    return output


def execute_thresholded_attend(
    query: Any,
    key: Any,
    value: Any,
    *,
    beta: float,
) -> Any:
    """TDA thresholded attention (one branch): L2-normalized squared-relu scores.

    ``query``/``key``/``value`` are BTHD rank-4 with a shared head count (the
    caller pre-expands any GQA group). Always causal and unnormalized, mirroring
    the canonical core. Returns the output in fp32.
    """
    batch, query_length, query_heads, key_dim, key_length, value_dim = _check_k1_operands(
        query, key, value, op="thresholded"
    )
    if query.shape[2] != key.shape[2]:
        raise ValueError("thresholded expects pre-expanded (shared) head counts")
    query_c, key_c, value_c = query.contiguous(), key.contiguous(), value.contiguous()
    block_m = min(64, max(16, triton.next_power_of_2(query_length)))
    block_n = min(64, max(16, triton.next_power_of_2(key_length)))
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    q_flat = query_c.permute(0, 2, 1, 3).contiguous()
    k_flat = key_c.permute(0, 2, 1, 3).contiguous()
    v_flat = value_c.permute(0, 2, 1, 3).contiguous()
    out_flat = torch.empty(
        (batch, query_heads, query_length, value_dim),
        device=query.device,
        dtype=torch.float32,
    )
    _thresholded_attend_forward_tiled[
        (batch * query_heads, triton.cdiv(query_length, block_m))
    ](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        query_length,
        key_length,
        key_dim,
        value_dim,
        float(beta),
        block_m,
        block_n,
        block_d,
        block_v,
        num_warps=4,
        num_stages=2,
    )
    return out_flat.permute(0, 2, 1, 3)


@torch.no_grad()
def execute_online_softmax_decode(
    query: Any,
    key: Any,
    value: Any,
    *,
    scale: float,
    causal: bool = True,
) -> Any:
    """Run one fused single-query K1 decode step against a persistent KV cache.

    This is the decode-path counterpart to :func:`execute_online_softmax`: the
    single query token (``query`` ``[B, HQ, K]``) attends to the full KV cache
    (``key``/``value`` ``[B, S, HK, K]``/``[B, S, HK, DV]``, BTHD) with the
    online-softmax accumulation, so the ``[1, S]`` score tensor is never
    materialized. The cache is read only — the caller appends the new token to
    it before or after this call. Runs under ``torch.no_grad()`` with no
    autograd-class construction, no host sync, and a fixed launch shape, so the
    step is CUDA-graph capturable.

    Decode semantics: the query token sits at the latest position ``S - 1``,
    so ``causal=True`` attends to the full history (all ``S`` keys); this
    matches the native kernel's qlen=1 causal output exactly. ``causal`` is
    accepted for interface symmetry — at the latest position causal and
    non-causal decode are identical (the full cache is visible either way).

    Returns the output ``[B, HQ, DV]`` in the input dtype.
    """
    if query.device.type != "cuda":
        raise ValueError("native K1 decode requires CUDA tensors")
    if query.ndim != 3 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "K1 decode query uses [B,HQ,K], key/value use [B,S,HK,K]/[B,S,HK,DV]"
        )
    batch, query_heads, key_dim = query.shape
    batch_k, key_length, key_heads, key_dim_k = key.shape
    value_dim = value.shape[-1]
    if min(batch, query_heads, key_dim, key_length, key_heads, value_dim) <= 0:
        raise ValueError("K1 decode dimensions must be positive")
    if key_dim > 128 or value_dim > 128:
        raise ValueError("native K1 decode supports key and value widths up to 128")
    if (batch, key_dim) != (batch_k, key_dim_k) or value.shape[:3] != (
        batch,
        key_length,
        key_heads,
    ):
        raise ValueError("K1 decode query/key/value dimensions do not agree")
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (query.dtype == key.dtype == value.dtype) or not query.is_floating_point():
        raise ValueError("K1 decode query/key/value must use one floating-point dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native K1 decode supports float16, bfloat16, and float32")
    if not (query.device == key.device == value.device):
        raise ValueError("K1 decode query/key/value must share one CUDA device")
    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    output = torch.empty(
        (batch, query_heads, value_dim), device=query.device, dtype=query.dtype
    )
    block_n = 64
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    _online_softmax_decode_kernel[(batch, query_heads)](
        query_c,
        key_c,
        value_c,
        output,
        key_length,
        query_heads,
        key_heads,
        key_dim,
        value_dim,
        scale,
        block_n,
        block_d,
        block_v,
        num_warps=4,
        num_stages=3,
    )
    return output


def execute_indexed_k1(
    query: Any,
    key: Any,
    value: Any,
    gather_indices: Any,
    *,
    scale: float,
) -> Any:
    """Run the indexed K1 gather-attend natively (softmax over the gathered set).

    ``query``/``key``/``value`` are BTHD rank-4; ``gather_indices`` is the
    external per-KV-head route ``[B, HK, TQ, W]`` (source positions, -1 = padding
    → masked). W is small (the clients use 2–64); the kernel loops over the W
    slots gathering per-row K/V, so SMEM stays at the dense kernel's tiles and
    the A10G's 101KB limit is met at head_dim <= 64 (same bound as the dense
    native K1). fp32 accumulation throughout; the backward scatters dk/dv via
    relaxed atomics (the K3 native policy).
    """
    if query.device.type != "cuda":
        raise ValueError("URM-native indexed K1 requires CUDA tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("indexed K1 query/key/value use BTHD rank-4 layout")
    batch, query_length, query_heads, key_dim = query.shape
    _, key_length, key_heads, _ = key.shape
    value_dim = value.shape[-1]
    if min(batch, query_length, query_heads, key_dim, key_length, key_heads, value_dim) <= 0:
        raise ValueError("indexed K1 dimensions must be positive")
    if gather_indices.ndim != 4:
        raise ValueError("indexed K1 gather_indices must be [B, HK, TQ, W]")
    if tuple(gather_indices.shape[:3]) != (batch, key_heads, query_length):
        raise ValueError("indexed K1 gather_indices must be [B, HK, TQ, W]")
    width = gather_indices.shape[-1]
    if width <= 0:
        raise ValueError("indexed K1 gather width W must be positive")
    if query_heads % key_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (query.dtype == key.dtype == value.dtype) or not query.is_floating_point():
        raise ValueError("indexed K1 query/key/value must use one floating-point dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native indexed K1 supports float16, bfloat16, and float32")
    if not (query.device == key.device == value.device == gather_indices.device):
        raise ValueError("indexed K1 operands must share one CUDA device")
    # The forward holds [BLOCK_M, BLOCK_D] q + per-slot gathered [BLOCK_M,
    # BLOCK_D] k / [BLOCK_M, BLOCK_V] v tiles; the A10G's 101KB SMEM caps head
    # widths at 64 (the dense native K1 bound). Decline wider heads loudly.
    if key_dim > 64 or value_dim > 64:
        raise ValueError(
            f"native indexed K1 supports head widths <= 64 on this device "
            f"(SMEM limit); got key_dim={key_dim}, value_dim={value_dim} — "
            f"the reference tier owns wider heads"
        )
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    gather = gather_indices.contiguous().to(torch.int32)
    block_m = 64 if query_length >= 64 else max(16, triton.next_power_of_2(query_length))
    block_d = max(16, triton.next_power_of_2(key_dim))
    block_v = max(16, triton.next_power_of_2(value_dim))
    # The per-slot gather keeps only dense-sized tiles live, so the forward runs
    # at block_m<=64, num_stages=1 like the SMEM-safe backward (measured under
    # the A10G's 101KB at head_dim=64).
    num_warps = 4
    num_stages = 1

    class _IndexedK1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, gather_idx):
            output = torch.empty(
                (batch, query_length, query_heads, value_dim),
                device=q.device,
                dtype=q.dtype,
            )
            logsumexp = torch.empty(
                (batch, query_heads, query_length), device=q.device, dtype=torch.float32
            )
            _indexed_k1_forward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                gather_idx,
                output,
                logsumexp,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                width,
                scale,
                q.dtype is torch.float32,
                block_m,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            ctx.save_for_backward(q, k, v, gather_idx, output, logsumexp)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, gather_idx, output, logsumexp = ctx.saved_tensors
            if grad_output is None:
                return None, None, None, None
            grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            # dk/dv accumulate a gather-scatter across queries (relaxed atomics),
            # so they start from zero.
            grad_k = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            grad_v = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            _indexed_k1_backward_tiled[
                (batch, query_heads, triton.cdiv(query_length, block_m))
            ](
                q,
                k,
                v,
                gather_idx,
                output,
                logsumexp,
                grad_output.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                query_length,
                key_length,
                query_heads,
                key_heads,
                key_dim,
                value_dim,
                width,
                scale,
                q.dtype is torch.float32,
                block_m,
                block_d,
                block_v,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return (
                grad_q.to(q.dtype),
                grad_k.to(k.dtype),
                grad_v.to(v.dtype),
                None,  # gather_indices is an integer route; no cotangent
            )

    return _IndexedK1.apply(query, key, value, gather)


__all__ = [
    "execute_online_softmax",
    "execute_online_softmax_decode",
    "execute_indexed_k1",
    "execute_softmax_probs",
    "execute_positive_feature",
    "execute_thresholded_attend",
]


def k1_softmax_attention(
    query,
    key,
    value,
    *,
    descriptor,
    score_bias=None,
    attention_mask=None,
    scale=None,
):
    """Canonical K1 softmax attention — the native Triton implementation of the
    uniform batched signature.

    Same role order, batched shapes (``[B, T, H, D]``), closed
    :class:`K1Descriptor` and output return as the NumPy oracle and Torch
    reference. The descriptor supplies causal/scale policy here — the only
    place that translation happens.
    """
    from urm.ir.program import K1ScaleRule as _SR

    if descriptor.scale_rule is _SR.EXPLICIT_OPERAND:
        if scale is None:
            raise ValueError("K1 explicit_operand scale rule requires a scale value")
        resolved = float(scale)
    else:
        resolved = float(query.shape[-1]) ** -0.5
    return execute_online_softmax(
        query,
        key,
        value,
        attention_mask=attention_mask,
        score_bias=score_bias,
        causal=descriptor.causal,
        scale=resolved,
    )


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class K1NativeTritonProvider:
    name = "urm_native_k1_online_softmax_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ...ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        # Decline what the online-softmax kernel does not execute. The kernel is
        # the plain dot-product score with a softmax reducer (plus causal masking,
        # score_bias and attention_mask); it has no channel-decay score, no
        # threshold/squared-sum reducer, and no indexed gather. Accepting one of
        # those descriptors here would silently run the wrong law.
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native K1 online softmax requires the DOT score law"
        if request.descriptor.reducer_law is not K1ReducerLaw.SOFTMAX:
            return "native K1 online softmax requires the SOFTMAX reducer law"
        if request.descriptor.indexed:
            return "native K1 online softmax does not execute indexed gather"
        import torch

        if not torch.cuda.is_available():
            return "native K1 requires CUDA"
        return None

    def execute(self, request, operands):
        out = k1_softmax_attention(
            operands["query"], operands["key"], operands["value"],
            descriptor=request.descriptor,
            score_bias=operands.get("score_bias"),
            attention_mask=operands.get("attention_mask"),
            scale=None if operands.get("scale") is None else float(operands["scale"]),
        )
        return {"output": out}


class K1NativeIndexedTritonProvider:
    """The native Triton provider for the A2 indexed-K1 gather-attend law.

    Serves ONLY ``descriptor.indexed == True`` (the dense native anchor declines
    those); the gathered source set comes from the external ``gather_indices``
    route. The equation is the same softmax-over-inline-Q·K law, restricted to
    the per-query gathered set — the descriptor's causal field is irrelevant on
    the indexed path (visibility is carried by the -1 padding in the route).
    """

    name = "urm_native_k1_indexed_gather_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ...ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        # This provider owns ONLY the indexed gather-attend: the plain dot score
        # with a softmax reducer over the gathered set. Decline everything else
        # (the dense native anchor serves the non-indexed softmax law).
        if not request.descriptor.indexed:
            return "native indexed K1 serves only indexed gather descriptors"
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native indexed K1 requires the DOT score law"
        if request.descriptor.reducer_law is not K1ReducerLaw.SOFTMAX:
            return "native indexed K1 requires the SOFTMAX reducer law"
        import torch

        if not torch.cuda.is_available():
            return "native indexed K1 requires CUDA"
        return None

    def execute(self, request, operands):
        from urm.ir.program import K1ScaleRule as _SR

        descriptor = request.descriptor
        gather_indices = operands.get("gather_indices")
        if gather_indices is None:
            raise ValueError("indexed K1 requires a gather_indices operand")
        scale_op = operands.get("scale")
        if descriptor.scale_rule is _SR.EXPLICIT_OPERAND:
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5
        out = execute_indexed_k1(
            operands["query"],
            operands["key"],
            operands["value"],
            gather_indices,
            scale=scale,
        )
        return {"output": out}


PROVIDERS = (K1NativeTritonProvider(), K1NativeIndexedTritonProvider())
