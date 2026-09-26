"""Native online-softmax attention (dense K1): tiled forward, two-pass backward, decode."""

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
    # BLOCK_D is the D-chunk tile width (always ≤ the SMEM-safe cap); when D exceeds
    # it, the channel axis is chunked: scores accumulate partial dot products over
    # NUM_D_CHUNKS = cdiv(D, BLOCK_D) chunks. The chunked schedule keeps the q/k tiles
    # at [BLOCK_M/N, BLOCK_D] regardless of the head width.
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    NUM_D_CHUNKS: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ
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
        # Chunked-D score accumulation: scores = Σ_c dot(q_c, k_c^T) over D-chunks.
        # The query chunk is re-loaded per key block (SMEM-bounded; the tile is
        # [BLOCK_M, BLOCK_D] regardless of D).
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            q_chunk = tl.load(
                Q
                + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + d_offs[None, :],
                query_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            k_chunk = tl.load(
                K
                + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            scores += tl.dot(
                q_chunk,
                tl.trans(k_chunk),
                input_precision="ieee" if INPUT_FP32 else "tf32",
            )
        scores = scores * (SCALE * LOG2E)
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
    NUM_D_CHUNKS: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dims = tl.arange(0, BLOCK_D)
    query_valid = query_offsets < TQ
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
    # Chunked-D: the full score (all D-chunks summed) is needed for the softmax
    # probabilities; grad_query accumulates per-chunk via dot(grad_scores, k_chunk).
    # The accumulators are separate register tiles (one per D-chunk), unrolled via
    # tl.static_range — NUM_D_CHUNKS is a constexpr so the unrolling is static.
    grad_query_0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    grad_query_1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_blocks = tl.cdiv(tl.minimum(last_visible + 1, TK), BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)
    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        # Full score: accumulate partial dot products over D-chunks.
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            q_chunk = tl.load(
                Q
                + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + d_offs[None, :],
                query_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            k_chunk = tl.load(
                K
                + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            scores += tl.dot(
                q_chunk, tl.trans(k_chunk),
                input_precision="ieee" if INPUT_FP32 else "tf32",
            )
        scores = scores * SCALE
        value = tl.load(
            V
            + ((batch * TK + keys[:, None]) * HK + key_head) * DV
            + tl.arange(0, BLOCK_V)[None, :],
            key_valid[:, None] & (tl.arange(0, BLOCK_V)[None, :] < DV),
            other=0.0,
        )
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
        # Per-D-chunk grad_query accumulation: dot(grad_scores, k_chunk) per chunk.
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            k_chunk = tl.load(
                K
                + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            contribution = tl.dot(
                grad_scores.to(k_chunk.dtype), k_chunk,
                input_precision="ieee" if INPUT_FP32 else "tf32",
            ) * SCALE
            if d_chunk == 0:
                grad_query_0 += contribution
            elif d_chunk == 1:
                grad_query_1 += contribution
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
    # Store grad_query per D-chunk.
    for d_chunk in tl.static_range(NUM_D_CHUNKS):
        d_offs = d_chunk * BLOCK_D + key_dims
        d_valid = d_offs < D
        grad_query_offset = (
            (batch * TQ + query_offsets[:, None]) * HQ + query_head
        ) * D + d_offs[None, :]
        if d_chunk == 0:
            tl.store(
                GRAD_Q + grad_query_offset,
                grad_query_0,
                query_valid[:, None] & d_valid[None, :],
            )
        elif d_chunk == 1:
            tl.store(
                GRAD_Q + grad_query_offset,
                grad_query_1,
                query_valid[:, None] & d_valid[None, :],
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
    NUM_D_CHUNKS: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D
    key_offsets = key_start + tl.arange(0, BLOCK_N)
    query_offsets = tl.arange(0, BLOCK_M)
    key_dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    key_valid = key_offsets < TK
    value = tl.load(
        V + ((batch * TK + key_offsets[:, None]) * HK + key_head) * DV + value_dims[None, :],
        key_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
    )
    # Chunked-D: grad_key accumulates per-chunk via dot(grad_scores, q_chunk).
    grad_key_0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    grad_key_1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
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
            # Full score: accumulate partial dot products over D-chunks.
            scores = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
            for d_chunk in tl.static_range(NUM_D_CHUNKS):
                d_offs = d_chunk * BLOCK_D + key_dims
                d_valid = d_offs < D
                k_chunk = tl.load(
                    K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D + d_offs[None, :],
                    key_valid[:, None] & d_valid[None, :], other=0.0,
                )
                q_chunk = tl.load(
                    Q + ((batch * TQ + qoff[:, None]) * HQ + query_head) * D + d_offs[None, :],
                    query_valid[:, None] & d_valid[None, :], other=0.0,
                )
                scores += tl.dot(
                    k_chunk, tl.trans(q_chunk),
                    input_precision="ieee" if INPUT_FP32 else "tf32",
                )
            scores = scores * SCALE
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
            # Per-D-chunk grad_key accumulation: dot(grad_scores, q_chunk) per chunk.
            for d_chunk in tl.static_range(NUM_D_CHUNKS):
                d_offs = d_chunk * BLOCK_D + key_dims
                d_valid = d_offs < D
                q_chunk = tl.load(
                    Q + ((batch * TQ + qoff[:, None]) * HQ + query_head) * D + d_offs[None, :],
                    query_valid[:, None] & d_valid[None, :], other=0.0,
                )
                contribution = tl.dot(
                    grad_scores.to(q_chunk.dtype), q_chunk,
                    input_precision="ieee" if INPUT_FP32 else "tf32",
                ) * SCALE
                if d_chunk == 0:
                    grad_key_0 += contribution
                elif d_chunk == 1:
                    grad_key_1 += contribution
            grad_value += tl.dot(
                probabilities.to(grad_output.dtype), grad_output,
                input_precision="ieee" if INPUT_FP32 else "tf32",
            )
    # Store grad_key per D-chunk.
    for d_chunk in tl.static_range(NUM_D_CHUNKS):
        d_offs = d_chunk * BLOCK_D + key_dims
        d_valid = d_offs < D
        if d_chunk == 0:
            tl.store(
                GRAD_K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D + d_offs[None, :],
                grad_key_0, key_valid[:, None] & d_valid[None, :],
            )
        elif d_chunk == 1:
            tl.store(
                GRAD_K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D + d_offs[None, :],
                grad_key_1, key_valid[:, None] & d_valid[None, :],
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
    # The forward kernel holds [BLOCK_M, BLOCK_D_CHUNK] q + [BLOCK_N, BLOCK_D_CHUNK] k
    # + [BLOCK_N, BLOCK_V] v + [BLOCK_M, BLOCK_N] score fp32 tiles. The channel axis is
    # chunked at BLOCK_D_CHUNK = min(next_power_of_2(D), 64), so the SMEM budget is
    # independent of the head width — wide heads (MLA key_dim=80, Tucker >64,
    # differential value_dim=128) run natively. block_m=64 with stages=2 stays under
    # the A10G's 101KB SMEM (measured ~74KB at D_CHUNK=64, BLOCK_V=64).
    block_m = 64 if query_length >= 64 else max(16, triton.next_power_of_2(query_length))
    block_n = 64
    block_d = min(max(16, triton.next_power_of_2(key_dim)), 64)
    # Chunked-D covers key_dim ≤ 128 (2 chunks of 64); wider heads need a deeper
    # unroll (residual work). BLOCK_V is NOT chunked (the value width is the output
    # width); decline wider value dims honestly.
    if key_dim > 128:
        raise ValueError(
            f"native K1 online softmax supports key widths <= 128 on this device "
            f"(chunked-D schedule, 2 chunks of 64); got key_dim={key_dim} — "
            f"the reference tier owns wider heads"
        )
    block_v = max(16, triton.next_power_of_2(value_dim))
    if value_dim > 64:
        raise ValueError(
            f"native K1 online softmax supports value widths <= 64 on this device "
            f"(SMEM limit); got value_dim={value_dim} — "
            f"the reference tier owns wider value heads"
        )
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
            # is the only path: the single-pass kernel holds too many fp32 tiles for
            # the A10G's 101KB SMEM at head_dim=64 (measured 131072 required). The
            # two-pass kernels run at block_m<=64, num_stages=1 — under the limit.
            grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            grad_k = torch.empty(k.shape, device=k.device, dtype=torch.float32)
            grad_v = torch.empty(v.shape, device=v.device, dtype=torch.float32)
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


class K1NativeTritonProvider:
    name = "urm_native_k1_online_softmax_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

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
