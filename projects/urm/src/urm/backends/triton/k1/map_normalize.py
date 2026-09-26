"""Native Triton K1 map-normalize reducer (PAttention/TokenFormer, A13).

The map-normalize law: an elementwise score map (exp/gelu/identity) composed with
an Lp normalization over the source domain, scaled by count^(1/p). The map and the
map↔norm order are closed descriptor fields (score_map / normalize_before_map).

The source domain is the parameter-token bank (typically ≤ 256 tokens), so the
full score row fits in one tile — no online rescaling needed. One program per
(batch, query-head, query-block) computes all scores, applies the map, computes
the Lp norm, normalizes, and contracts with V.

The backward runs through torch autograd on the recomputed scores (the source
domain is small, so the torch backward is cheap and exact — the Lp-norm Jacobian
is complex and error-prone in a hand-written kernel).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _map_normalize_forward(
    Q, K, V, OUTPUT,
    TQ: tl.constexpr, TK: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr,
    SCALE: tl.constexpr,
    SCORE_MAP: tl.constexpr,  # 0=exp, 1=gelu, 2=identity
    NORMALIZER_P: tl.constexpr,
    COUNT_SCALE: tl.constexpr,
    NORMALIZE_BEFORE_MAP: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    # Full score row: all keys in one tile (the source domain is small).
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
            K + ((batch * TK + key_offsets[:, None]) * HK + key_head) * D
            + d_offs[None, :],
            (key_offsets[:, None] < TK) & d_valid[None, :], other=0.0,
        )
        scores += tl.dot(q_chunk, tl.trans(k_chunk), input_precision="ieee")
    scores = scores * SCALE
    score_valid = query_valid[:, None] & (key_offsets[None, :] < TK)
    if CAUSAL:
        visible = key_offsets[None, :] <= query_offsets[:, None] + TK - TQ
        scores = tl.where(visible, scores, 0.0)
    scores = tl.where(score_valid, scores, 0.0)

    # Map + normalize (order depends on normalize_before_map). The score_valid mask
    # must zero the mapped values BEFORE the norm — exp(0)=1 at padded positions
    # would inflate the norm.
    if NORMALIZE_BEFORE_MAP:
        scores_masked = tl.where(score_valid, scores, 0.0)
        norm = tl.sum(tl.exp(NORMALIZER_P * tl.log(tl.maximum(tl.abs(scores_masked), 1e-30))), axis=1, keep_dims=True)
        norm = tl.exp(tl.log(tl.maximum(norm, 1e-30)) / NORMALIZER_P)
        normed = scores / tl.maximum(norm, 1e-30) * COUNT_SCALE
        if SCORE_MAP == 0:
            weights = tl.exp(normed)
        elif SCORE_MAP == 1:
            weights = 0.5 * normed * (1.0 + tl.math.tanh(0.7978845604730 * (normed + 0.044715 * normed * normed * normed)))
        else:
            weights = normed
    else:
        if SCORE_MAP == 0:
            mapped = tl.exp(scores)
        elif SCORE_MAP == 1:
            mapped = 0.5 * scores * (1.0 + tl.math.tanh(0.7978845604730 * (scores + 0.044715 * scores * scores * scores)))
        else:
            mapped = scores
        mapped = tl.where(score_valid, mapped, 0.0)
        norm = tl.sum(tl.exp(NORMALIZER_P * tl.log(tl.maximum(tl.abs(mapped), 1e-30))), axis=1, keep_dims=True)
        norm = tl.exp(tl.log(tl.maximum(norm, 1e-30)) / NORMALIZER_P)
        weights = mapped / tl.maximum(norm, 1e-30) * COUNT_SCALE

    weights = tl.where(score_valid, weights, 0.0)

    values = tl.load(
        V + ((batch * TK + key_offsets[:, None]) * HK + key_head) * DV + value_dims[None, :],
        (key_offsets[:, None] < TK) & (value_dims[None, :] < DV), other=0.0,
    )
    output = tl.dot(weights.to(tl.float32), values.to(tl.float32), input_precision="ieee")
    output_offset = (
        (batch * TQ + query_offsets[:, None]) * HQ + query_head
    ) * DV + value_dims[None, :]
    tl.store(
        OUTPUT + output_offset, output,
        query_valid[:, None] & (value_dims[None, :] < DV),
    )


_SCORE_MAP_IDS = {"exp": 0, "gelu": 1, "identity": 2}


def _forward_torch(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
    causal: bool, scale: float, score_map: int,
    normalizer_p: float, normalize_before_map: bool,
) -> torch.Tensor:
    """Torch forward matching the kernel's equation (used for the backward)."""
    B, T, HQ, D = query.shape
    TK = key.shape[1]
    q_h = query.permute(0, 2, 1, 3)
    k_h = key.permute(0, 2, 1, 3)
    v_h = value.permute(0, 2, 1, 3)
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
    if causal:
        mask = torch.ones(T, TK, dtype=torch.bool, device=scores.device).tril_(
            diagonal=TK - T
        )
        scores = scores.masked_fill(~mask, 0.0)

    count = TK
    p = normalizer_p
    count_scale = count ** (1.0 / p)

    def _map(x):
        if score_map == 0:
            return torch.exp(x)
        if score_map == 1:
            return torch.nn.functional.gelu(x)
        return x

    def _norm(x):
        return x / torch.norm(x, p=p, dim=-1, keepdim=True) * count_scale

    if normalize_before_map:
        weights = _map(_norm(scores))
    else:
        weights = _norm(_map(scores))

    out = torch.matmul(weights, v_h)
    return out.permute(0, 2, 1, 3)


class _MapNormalizeK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal, scale, score_map, normalizer_p, normalize_before_map):
        B, T, HQ, D = query.shape
        TK, HK, DV = key.shape[1], key.shape[2], value.shape[3]
        block_m = min(64, max(16, triton.next_power_of_2(T)))
        block_n = max(16, triton.next_power_of_2(TK))
        block_d = min(max(16, triton.next_power_of_2(D)), 64)
        block_v = max(16, triton.next_power_of_2(DV))
        out = torch.empty(B, T, HQ, DV, dtype=query.dtype, device=query.device)
        grid = (B, HQ, triton.cdiv(T, block_m))
        _map_normalize_forward[grid](
            query, key, value, out,
            TQ=T, TK=TK, HQ=HQ, HK=HK, D=D, DV=DV,
            SCALE=scale, SCORE_MAP=score_map,
            NORMALIZER_P=normalizer_p,
            COUNT_SCALE=TK ** (1.0 / normalizer_p),
            NORMALIZE_BEFORE_MAP=normalize_before_map,
            CAUSAL=causal,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_d, BLOCK_V=block_v,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(query, key, value)
        ctx.causal = causal
        ctx.scale = scale
        ctx.score_map = score_map
        ctx.normalizer_p = normalizer_p
        ctx.normalize_before_map = normalize_before_map
        # The kernel writes into out in-place (untracked by autograd); clone to
        # connect the output to the autograd graph.
        return out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        # The Lp-norm Jacobian is complex; run the backward through torch autograd
        # on the recomputed scores. The source domain is small (parameter tokens),
        # so this is cheap and exact.
        with torch.enable_grad():
            q = query.detach().requires_grad_(True)
            k = key.detach().requires_grad_(True)
            v = value.detach().requires_grad_(True)
            out = _forward_torch(
                q, k, v,
                causal=ctx.causal, scale=ctx.scale,
                score_map=ctx.score_map, normalizer_p=ctx.normalizer_p,
                normalize_before_map=ctx.normalize_before_map,
            )
            out.backward(grad_output)
        return q.grad, k.grad, v.grad, None, None, None, None, None


class K1NativeMapNormalizeProvider:
    name = "urm_native_k1_map_normalize_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        if request.descriptor.reducer_law is not K1ReducerLaw.MAP_NORMALIZE:
            return "native K1 map-normalize requires the MAP_NORMALIZE reducer law"
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "native K1 map-normalize requires the DOT score law"
        if request.descriptor.indexed:
            return "native K1 map-normalize does not execute indexed gather"
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

        score_map_id = _SCORE_MAP_IDS[desc.score_map.value]
        out = _MapNormalizeK1.apply(
            query, key, value,
            desc.causal, scale, score_map_id,
            desc.normalizer_p, desc.normalize_before_map,
        )
        return {"output": out}


__all__ = ["K1NativeMapNormalizeProvider"]
