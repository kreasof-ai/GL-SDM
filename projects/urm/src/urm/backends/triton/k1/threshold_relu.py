"""Native Triton K1 threshold-ReLU-power reducer (TDA, A13).

The threshold law: out = (ReLU(s − τ))^p @ V with position-dependent threshold
τ_i = β·sqrt(2·log(i+1)/d), causal mask zeroing j>i before the threshold, and NO
normalization denominator. The simplest K1 reducer — no exp, no online rescaling.

One program per (batch, query-head, query-block) loops the key blocks: compute
scores, apply causal mask, threshold, power, accumulate weighted values.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _threshold_relu_forward(
    Q, K, V, OUTPUT,
    TQ: tl.constexpr, TK: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    THRESHOLD_BETA: tl.constexpr,
    RELU_POWER: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
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

    # τ_i = β·sqrt(2·log(i+1)/d) — per query position (1-indexed).
    positions = (query_offsets + 1).to(tl.float32)
    tau = THRESHOLD_BETA * tl.sqrt(2.0 * tl.log(positions) / D)

    output = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_limit = tl.minimum(last_visible + 1, TK)
        key_blocks = tl.cdiv(key_limit, BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)

    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        # Chunked-D score accumulation.
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            q_chunk = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + d_offs[None, :],
                query_valid[:, None] & d_valid[None, :], other=0.0,
            )
            k_chunk = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :], other=0.0,
            )
            scores += tl.dot(q_chunk, tl.trans(k_chunk), input_precision="ieee")
        scores = scores * SCALE
        # Causal mask: zero future scores before the threshold.
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            scores = tl.where(visible, scores, 0.0)
        score_valid = query_valid[:, None] & key_valid[None, :]
        scores = tl.where(score_valid, scores, 0.0)
        # Threshold: ReLU(s − τ)^p, no normalization.
        weights = tl.maximum(scores - tau[:, None], 0.0)
        weights = tl.exp(RELU_POWER * tl.log(tl.maximum(weights, 1e-30)))
        values = tl.load(
            V + ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
        )
        output += tl.dot(weights.to(tl.float32), values.to(tl.float32), input_precision="ieee")

    output_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset, output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )


@triton.jit
def _threshold_relu_backward(
    Q, K, V, GRAD_OUTPUT,
    GRAD_Q, GRAD_K, GRAD_V,
    TQ: tl.constexpr, TK: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    THRESHOLD_BETA: tl.constexpr,
    RELU_POWER: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Backward for the threshold-ReLU-power reducer.

    weights[t,s] = ReLU(s[t,s] − τ[t])^p
    d(weights)/ds = p·ReLU(s−τ)^(p−1)  for s > τ, else 0
    grad_q[t] = Σ_s dW[t,s]·k[s]·scale  where dW = p·ReLU(s−τ)^(p−1)·grad_prob
    grad_k[s] = Σ_t dW[t,s]·q[t]·scale
    grad_v[s] = Σ_t W[t,s]·grad_out[t]
    """
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

    positions = (query_offsets + 1).to(tl.float32)
    tau = THRESHOLD_BETA * tl.sqrt(2.0 * tl.log(positions) / D)

    grad_out = tl.load(
        GRAD_OUTPUT + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
    )

    # grad_q accumulates per D-chunk (chunked-D schedule).
    grad_q_0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    grad_q_1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_limit = tl.minimum(last_visible + 1, TK)
        key_blocks = tl.cdiv(key_limit, BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)

    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        # Full score (chunked-D).
        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            q_chunk = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + d_offs[None, :],
                query_valid[:, None] & d_valid[None, :], other=0.0,
            )
            k_chunk = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :], other=0.0,
            )
            scores += tl.dot(q_chunk, tl.trans(k_chunk), input_precision="ieee")
        scores = scores * SCALE
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            scores = tl.where(visible, scores, 0.0)
        score_valid = query_valid[:, None] & key_valid[None, :]
        scores = tl.where(score_valid, scores, 0.0)

        # weights = ReLU(s − τ)^p; dW/ds = p·ReLU(s−τ)^(p−1)
        relu = tl.maximum(scores - tau[:, None], 0.0)
        weights = tl.exp(RELU_POWER * tl.log(tl.maximum(relu, 1e-30)))
        dW_ds = tl.where(relu > 0.0, RELU_POWER * tl.exp((RELU_POWER - 1.0) * tl.log(tl.maximum(relu, 1e-30))), 0.0)

        values = tl.load(
            V + ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV), other=0.0,
        )

        # grad_v[s] += Σ_t weights[t,s]·grad_out[t]
        grad_v_contrib = tl.dot(
            tl.trans(weights), grad_out.to(tl.float32), input_precision="ieee",
        )
        grad_v_offset = (
            ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :]
        )
        tl.atomic_add(
            GRAD_V + grad_v_offset, grad_v_contrib,
            key_valid[:, None] & (value_dims[None, :] < DV), sem="relaxed",
        )

        # grad_prob[t,s] = grad_out[t]·v[s]  (the dL/dweights)
        grad_prob = tl.dot(grad_out.to(tl.float32), tl.trans(values).to(tl.float32), input_precision="ieee")
        # dL/ds[t,s] = dW/ds[t,s] · grad_prob[t,s]
        grad_scores = dW_ds * grad_prob
        grad_scores = tl.where(score_valid, grad_scores, 0.0)

        # grad_q per D-chunk: dot(grad_scores, k_chunk) * scale
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            k_chunk = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + d_offs[None, :],
                key_valid[:, None] & d_valid[None, :], other=0.0,
            )
            contribution = tl.dot(
                grad_scores, k_chunk.to(tl.float32), input_precision="ieee",
            ) * SCALE
            if d_chunk == 0:
                grad_q_0 += contribution
            elif d_chunk == 1:
                grad_q_1 += contribution

        # grad_k per D-chunk: dot(grad_scores^T, q_chunk) * scale
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_offs = d_chunk * BLOCK_D + key_dims
            d_valid = d_offs < D
            q_chunk = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + d_offs[None, :],
                query_valid[:, None] & d_valid[None, :], other=0.0,
            )
            grad_k_chunk = tl.dot(
                tl.trans(grad_scores), q_chunk.to(tl.float32), input_precision="ieee",
            ) * SCALE
            grad_k_offset = (
                ((batch * TK + keys[:, None]) * HK + key_head) * D + d_offs[None, :]
            )
            tl.atomic_add(
                GRAD_K + grad_k_offset, grad_k_chunk,
                key_valid[:, None] & d_valid[None, :], sem="relaxed",
            )

    # Store grad_q per D-chunk.
    for d_chunk in tl.static_range(NUM_D_CHUNKS):
        d_offs = d_chunk * BLOCK_D + key_dims
        d_valid = d_offs < D
        grad_q_offset = (
            ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D + d_offs[None, :]
        )
        if d_chunk == 0:
            tl.store(GRAD_Q + grad_q_offset, grad_q_0, query_valid[:, None] & d_valid[None, :])
        elif d_chunk == 1:
            tl.store(GRAD_Q + grad_q_offset, grad_q_1, query_valid[:, None] & d_valid[None, :])


class _ThresholdReluK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal, scale, threshold_beta, relu_power):
        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = 64
        block_d = min(max(16, triton.next_power_of_2(D)), 64)
        block_v = max(16, triton.next_power_of_2(DV))
        out = torch.empty(B, T, HQ, DV, dtype=query.dtype, device=query.device)
        grid = (B, HQ, triton.cdiv(T, block_m))
        _threshold_relu_forward[grid](
            query, key, value, out,
            TQ=T, TK=TK, HQ=HQ, HK=HK, D=D, DV=DV,
            CAUSAL=causal, SCALE=scale,
            THRESHOLD_BETA=threshold_beta, RELU_POWER=relu_power,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_d, BLOCK_V=block_v,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(query, key, value)
        ctx.causal = causal
        ctx.scale = scale
        ctx.threshold_beta = threshold_beta
        ctx.relu_power = relu_power
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = 64
        block_d = min(max(16, triton.next_power_of_2(D)), 64)
        block_v = max(16, triton.next_power_of_2(DV))
        grad_q = torch.zeros_like(query)
        grad_k = torch.zeros_like(key)
        grad_v = torch.zeros_like(value)
        grid = (B, HQ, triton.cdiv(T, block_m))
        _threshold_relu_backward[grid](
            query, key, value, grad_output.contiguous(),
            grad_q, grad_k, grad_v,
            TQ=T, TK=TK, HQ=HQ, HK=HK, D=D, DV=DV,
            CAUSAL=ctx.causal, SCALE=ctx.scale,
            THRESHOLD_BETA=ctx.threshold_beta, RELU_POWER=ctx.relu_power,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_d, BLOCK_V=block_v,
            num_warps=4, num_stages=1,
        )
        return grad_q, grad_k, grad_v, None, None, None, None


class K1NativeThresholdReluProvider:
    name = "urm_native_k1_threshold_relu_power_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        if request.descriptor.reducer_law is not K1ReducerLaw.THRESHOLD_RELU_POWER:
            return "native K1 threshold-relu requires the THRESHOLD_RELU_POWER reducer law"
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native K1 threshold-relu requires the DOT score law"
        if request.descriptor.indexed:
            return "native K1 threshold-relu does not execute indexed gather"
        import torch
        if not torch.cuda.is_available():
            return "native K1 requires CUDA"
        return None

    def execute(self, request, operands):
        from ....ir.program import K1ScaleRule as _SR

        desc = request.descriptor
        query, key, value = operands["query"], operands["key"], operands["value"]
        if desc.scale_rule is _SR.EXPLICIT_OPERAND:
            scale = float(operands["scale"])
        else:
            scale = float(query.shape[-1]) ** -0.5

        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = 64
        block_d = min(max(16, triton.next_power_of_2(D)), 64)
        block_v = max(16, triton.next_power_of_2(DV))
        if DV > 64:
            raise ValueError(
                f"native K1 threshold-relu supports value widths <= 64; got {DV}"
            )
        out = _ThresholdReluK1.apply(
            query, key, value,
            desc.causal, scale, desc.threshold_beta, desc.relu_power,
        )
        return {"output": out}


__all__ = ["K1NativeThresholdReluProvider"]
