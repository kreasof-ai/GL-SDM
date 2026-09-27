"""Native Triton K1 squared-sum reducer (KATA, A13).

The squared-sum law: A[t,s] = Σ_g (scale·q_g[t]·k_g[s])² over squared_sum_groups
groups of width E = D/M; output o = (Σ_s A·v_s)/max(Σ_s A, 1) — a positive squared
score with sum normalization, no softmax, no epsilon. Causal masking zeroes future
scores.

One program per (batch, query-head, query-block) loops the key blocks: accumulate
num += A·v and den += ΣA across key blocks, then divide. No online rescaling (no
exp/max) — the scores are non-negative polynomials.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _squared_sum_forward(
    Q, K, V, OUTPUT, DENOM,
    TQ: tl.constexpr, TK: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr,
    M: tl.constexpr,  # squared_sum_groups
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,  # per-group width tile
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    E: tl.constexpr = D // M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    e_dims = tl.arange(0, BLOCK_E)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ

    num = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    den = tl.zeros((BLOCK_M,), tl.float32)

    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_limit = tl.minimum(last_visible + 1, TK)
        key_blocks = tl.cdiv(key_limit, BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)

    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK
        # A[t,s] = Σ_g (scale · q_g·k_g)² — accumulate over groups.
        A = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for g in tl.static_range(M):
            g_base = g * E
            qg = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + g_base + e_dims[None, :],
                query_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            kg = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + g_base + e_dims[None, :],
                key_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            gd = tl.dot(qg, tl.trans(kg), input_precision="ieee") * SCALE
            A += gd * gd  # squared
        # Causal mask: zero future scores.
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            A = tl.where(visible, A, 0.0)
        score_valid = query_valid[:, None] & key_valid[None, :]
        A = tl.where(score_valid, A, 0.0)
        # Accumulate num += A·v and den += Σ_s A.
        values = tl.load(
            V + ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        num += tl.dot(A.to(values.dtype), values, input_precision="ieee")
        den += tl.sum(A, axis=1)

    # o = num / max(den, 1) — 1-safe denominator (pinned law).
    safe_den = tl.maximum(den, 1.0)
    output = num / safe_den[:, None]
    output_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset, output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )
    # Save the denominator for the backward.
    tl.store(
        DENOM + (batch * HQ + query_head) * TQ + query_offsets,
        safe_den, query_valid,
    )


@triton.jit
def _squared_sum_backward(
    Q, K, V, GRAD_OUTPUT, DENOM, FWD_OUT,
    GRAD_Q, GRAD_K, GRAD_V,
    TQ: tl.constexpr, TK: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr,
    M: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Backward for the squared-sum reducer.

    out = num/den where num = A·v, den = max(ΣA, 1), A[t,s] = Σ_g (scale·q_g·k_g)².
    dL/dA[t,s] = (dL/dout[t]·v[s])/den[t] - (dL/dout[t]·num[t])/(den[t]²)  [when den > 1]
    dL/dq_g[t] = 2·scale·Σ_s dL/dA[t,s]·(scale·q_g·k_g)·k_g[s]
    dL/dk_g[s] = 2·scale·Σ_t dL/dA[t,s]·(scale·q_g·k_g)·q_g[t]
    dL/dv[s] = Σ_t dL/dA[t,s]·A[t,s] / den[t]  (wait — dL/dv[s] = Σ_t grad_out[t]·A[t,s]/den[t])
    """
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    query_start = tl.program_id(2) * BLOCK_M
    key_head = query_head // (HQ // HK)
    E: tl.constexpr = D // M
    query_offsets = query_start + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    e_dims = tl.arange(0, BLOCK_E)
    value_dims = tl.arange(0, BLOCK_V)
    query_valid = query_offsets < TQ

    # Load grad_output and denominator for this query block.
    grad_out = tl.load(
        GRAD_OUTPUT + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    den = tl.load(
        DENOM + (batch * HQ + query_head) * TQ + query_offsets,
        query_valid, other=1.0,
    )
    inv_den = 1.0 / den
    # Forward output for the denominator correction term.
    fwd_out = tl.load(
        FWD_OUT + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * DV
        + value_dims[None, :],
        query_valid[:, None] & (value_dims[None, :] < DV),
        other=0.0,
    )
    # correction[t] = Σ_d grad_out[t,d]·out[t,d] / den[t] — only when den > 1.
    # When den ≤ 1 the clamp makes den constant, so no denominator derivative.
    correction = tl.sum(grad_out * fwd_out, axis=1) * inv_den
    correction = tl.where(den > 1.0, correction, 0.0)

    # Accumulate grad_q per group: dL/dq_g[t] = 2·scale²·Σ_s dA[t,s]·(q_g·k_g)·k_g[s]
    # where dA[t,s] = grad_out[t]·v[s]/den[t] - (grad_out[t]·num[t])/den[t]²
    # We compute dA per key block, then accumulate the per-group gradients.
    grad_q = tl.zeros((BLOCK_M, M * BLOCK_E), tl.float32)

    if CAUSAL:
        last_visible = query_start + BLOCK_M - 1 + (TK - TQ)
        key_limit = tl.minimum(last_visible + 1, TK)
        key_blocks = tl.cdiv(key_limit, BLOCK_N)
    else:
        key_blocks = tl.cdiv(TK, BLOCK_N)

    for key_start in range(key_blocks):
        keys = key_start * BLOCK_N + key_offsets
        key_valid = keys < TK

        # Recompute A (accumulated over groups) and load values for this key block.
        values = tl.load(
            V + ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :],
            key_valid[:, None] & (value_dims[None, :] < DV),
            other=0.0,
        )
        A = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for g in tl.static_range(M):
            g_base = g * E
            qg = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + g_base + e_dims[None, :],
                query_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            kg = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + g_base + e_dims[None, :],
                key_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            gd = tl.dot(qg, tl.trans(kg), input_precision="ieee") * SCALE
            A += gd * gd

        # dA[t,s] = (grad_out[t]·v[s])/den[t] − correction[t]
        dA_num = tl.dot(grad_out, tl.trans(values), input_precision="ieee")
        dA = dA_num * inv_den[:, None] - correction[:, None]

        # Causal mask
        if CAUSAL:
            visible = keys[None, :] <= query_offsets[:, None] + TK - TQ
            dA = tl.where(visible, dA, 0.0)
            A = tl.where(visible, A, 0.0)
        score_valid = query_valid[:, None] & key_valid[None, :]
        dA = tl.where(score_valid, dA, 0.0)
        A = tl.where(score_valid, A, 0.0)

        # dL/dv[s,d] = Σ_t A[t,s]·grad_out[t,d]·inv_den[t]
        # Cast both operands to fp32 for the dot (A is fp32, grad_out may be bf16).
        grad_v_contrib = tl.dot(
            tl.trans(A),
            (grad_out * inv_den[:, None]).to(tl.float32),
            input_precision="ieee",
        )
        grad_v_offset = (
            ((batch * TK + keys[:, None]) * HK + key_head) * DV + value_dims[None, :]
        )
        tl.atomic_add(
            GRAD_V + grad_v_offset, grad_v_contrib,
            key_valid[:, None] & (value_dims[None, :] < DV),
            sem="relaxed",
        )

        # Per-group q/k gradients: dL/dq_g = 2·scale·dA·gd_g·k_g, dL/dk_g = 2·scale·dA·gd_g·q_g.
        for g in tl.static_range(M):
            g_base = g * E
            qg = tl.load(
                Q + ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + g_base + e_dims[None, :],
                query_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            kg = tl.load(
                K + ((batch * TK + keys[:, None]) * HK + key_head) * D
                + g_base + e_dims[None, :],
                key_valid[:, None] & (e_dims[None, :] < E),
                other=0.0,
            )
            gd = tl.dot(qg, tl.trans(kg), input_precision="ieee") * SCALE
            coeff = 2.0 * SCALE * dA * gd  # [BLOCK_M, BLOCK_N] fp32
            grad_qg = tl.dot(coeff, kg.to(tl.float32), input_precision="ieee")
            grad_q_offset = (
                ((batch * TQ + query_offsets[:, None]) * HQ + query_head) * D
                + g_base + e_dims[None, :]
            )
            tl.atomic_add(
                GRAD_Q + grad_q_offset, grad_qg,
                query_valid[:, None] & (e_dims[None, :] < E),
                sem="relaxed",
            )
            grad_kg = tl.dot(
                tl.trans(coeff), qg.to(tl.float32), input_precision="ieee"
            )
            grad_k_offset = (
                ((batch * TK + keys[:, None]) * HK + key_head) * D
                + g_base + e_dims[None, :]
            )
            tl.atomic_add(
                GRAD_K + grad_k_offset, grad_kg,
                key_valid[:, None] & (e_dims[None, :] < E),
                sem="relaxed",
            )


class _SquaredSumK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, groups, causal, scale):
        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        E = D // groups
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = 64
        block_e = max(16, triton.next_power_of_2(E))
        block_v = max(16, triton.next_power_of_2(DV))
        out = torch.empty(B, T, HQ, DV, dtype=query.dtype, device=query.device)
        denom = torch.empty(B, HQ, T, dtype=torch.float32, device=query.device)
        grid = (B, HQ, triton.cdiv(T, block_m))
        _squared_sum_forward[grid](
            query, key, value, out, denom,
            TQ=T, TK=TK, HQ=HQ, HK=HK, D=D, DV=DV,
            M=groups, CAUSAL=causal, SCALE=scale,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_E=block_e, BLOCK_V=block_v,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(query, key, value, denom, out)
        ctx.groups = groups
        ctx.causal = causal
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, denom, fwd_out = ctx.saved_tensors
        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        E = D // ctx.groups
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = 64
        block_e = max(16, triton.next_power_of_2(E))
        block_v = max(16, triton.next_power_of_2(DV))
        grad_q = torch.zeros_like(query)
        grad_k = torch.zeros_like(key)
        grad_v = torch.zeros_like(value)
        grid = (B, HQ, triton.cdiv(T, block_m))
        _squared_sum_backward[grid](
            query, key, value, grad_output.contiguous(), denom, fwd_out,
            grad_q, grad_k, grad_v,
            TQ=T, TK=TK, HQ=HQ, HK=HK, D=D, DV=DV,
            M=ctx.groups, CAUSAL=ctx.causal, SCALE=ctx.scale,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_E=block_e, BLOCK_V=block_v,
            num_warps=4, num_stages=1,
        )
        return grad_q, grad_k, grad_v, None, None, None


def squared_sum_attention(
    query: Any, key: Any, value: Any, *,
    groups: int, causal: bool, scale: float,
) -> Any:
    """Native K1 squared-sum attention (KATA)."""
    return _SquaredSumK1.apply(query, key, value, groups, causal, scale)


class K1NativeSquaredSumProvider:
    name = "urm_native_k1_squared_sum_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        if request.descriptor.reducer_law is not K1ReducerLaw.SQUARED_SUM:
            return "native K1 squared-sum requires the SQUARED_SUM reducer law"
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native K1 squared-sum requires the DOT score law"
        if request.descriptor.indexed:
            return "native K1 squared-sum does not execute indexed gather"
        import torch
        if not torch.cuda.is_available():
            return "native K1 requires CUDA"
        return None

    def execute(self, request, operands):
        from ....ir.program import K1ScaleRule as _SR

        desc = request.descriptor
        if desc.scale_rule is _SR.EXPLICIT_OPERAND:
            scale = float(operands["scale"])
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5
        out = squared_sum_attention(
            operands["query"], operands["key"], operands["value"],
            groups=desc.squared_sum_groups or 1,
            causal=desc.causal, scale=scale,
        )
        return {"output": out}


__all__ = ["K1NativeSquaredSumProvider", "squared_sum_attention"]
