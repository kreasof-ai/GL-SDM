"""Differentiable PyTorch reference for the K1 softmax-attention equation.

This is the Torch tier of the same K1 contract as the NumPy oracle
(:mod:`urm.backends.numpy.k1`) and the native Triton
schedule (:mod:`urm.backends.triton.k1`). The equation is carried by the
closed :class:`~urm.ir.program.K1Descriptor`: score scale law, causal masking,
grouped head sharing and the all-masked-row-zero policy. Runtime code owns no
part of this equation; it only binds role operands and invokes a provider.
"""

from __future__ import annotations

from typing import Any

from ...ir.program import K1Descriptor, K1ReducerLaw, K1ScaleRule, K1ScoreLaw


def _torch() -> Any:
    import torch

    return torch


def k1_softmax_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    descriptor: K1Descriptor,
    score_bias: Any | None = None,
    attention_mask: Any | None = None,
    scale: float | None = None,
    channel_gate: Any | None = None,
    gather_indices: Any | None = None,
) -> Any:
    """Canonical K1 softmax attention over ``[B, T, H, D]`` operands.

    The score scale follows the descriptor's scale law (never a runtime
    default): the key-dim rule or an explicit operand. Grouped query→KV head
    sharing expands shared KV heads. A fully masked row returns zero.

    The score law is the descriptor's ``score_law``: ``DOT`` is the plain
    dot-product score (an additive term rides ``score_bias``); ``CHANNEL_DECAY``
    contracts per channel with a decay ``exp(P_in − P_jn)`` where ``P`` is the
    prefix cumsum of ``channel_gate`` (the Wall score law).
    """
    torch = _torch()
    if descriptor.scale_rule is K1ScaleRule.EXPLICIT_OPERAND:
        if scale is None:
            raise ValueError("K1 explicit_operand scale rule requires a scale value")
        resolved_scale = float(scale)
    else:
        resolved_scale = float(query.shape[-1]) ** -0.5

    q = query.to(torch.float32).transpose(1, 2)
    k = key.to(torch.float32).transpose(1, 2)
    v = value.to(torch.float32).transpose(1, 2)
    if k.shape[1] != q.shape[1]:  # grouped head map: expand shared KV heads
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    if descriptor.indexed:
        # A2 indexed-K1: gather the per-query source set given by gather_indices
        # [B, Hkv, T, W] (source positions, -1 = padding → masked), then attend
        # over the gathered K/V. The route (indices) is external. gather_indices
        # is per KV head; each query head maps to its shared KV head (GQA).
        if gather_indices is None:
            raise ValueError("indexed K1 requires a gather_indices operand")
        idx = gather_indices.to(torch.long)                              # [B,Hkv,T,W]
        valid = idx >= 0
        idx_safe = idx.clamp(min=0)
        B, Hkv, T, W = idx.shape
        HQ = q.shape[1]
        # Use the UN-expanded k/v ([B,Hkv,S,D]) — gather per KV head, then expand
        # to query heads. (k/v here are pre-expansion only if the head map left
        # them shared; with an equal map Hkv == HQ.)
        kv = key.to(torch.float32).transpose(1, 2)                      # [B,Hkv,S,D]
        vv = value.to(torch.float32).transpose(1, 2)
        S_full = kv.shape[2]
        gathered_k = torch.gather(
            kv.unsqueeze(2).expand(B, Hkv, T, S_full, kv.shape[-1]),
            3, idx_safe.unsqueeze(-1).expand(B, Hkv, T, W, kv.shape[-1]),
        )                                                               # [B,Hkv,T,W,D]
        gathered_v = torch.gather(
            vv.unsqueeze(2).expand(B, Hkv, T, S_full, vv.shape[-1]),
            3, idx_safe.unsqueeze(-1).expand(B, Hkv, T, W, vv.shape[-1]),
        )
        # Expand the gathered KV and the validity mask to the query heads (GQA).
        if HQ != Hkv:
            rep = HQ // Hkv
            gathered_k = gathered_k.repeat_interleave(rep, dim=1)
            gathered_v = gathered_v.repeat_interleave(rep, dim=1)
            valid = valid.repeat_interleave(rep, dim=1)
        # scores over the gathered set: [B,HQ,T,W]
        scores = torch.einsum("bhtd,bhtwd->bhtw", q, gathered_k) * resolved_scale
        scores = scores.masked_fill(~valid, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out = torch.einsum("bhtw,bhtwd->bhtd", probs, gathered_v)
        return out.transpose(1, 2).to(value.dtype)
    if descriptor.score_law is K1ScoreLaw.CHANNEL_DECAY:
        if channel_gate is None:
            raise ValueError("K1 channel_decay score law requires a channel_gate operand")
        # s_ij = (Σ_n q_in k_jn · exp(P_in − P_jn)) · scale, P = cumsum(channel_gate).
        # The channel_gate operand is the natural-log gate. (The pinned Wall source
        # works in base-2 — exp2(cumsum(g_log2)·RCP_LN2) — and applies scale·RCP_LN2;
        # exp(cumsum(g_log2) diff) with the gate passed as g_log2 reproduces the decay,
        # and the caller folds RCP_LN2 into the scale to match the pinned softmax base.)
        # P = prefix cumsum of the gate over TIME (dim 2 in [B,H,T,D]).
        P = channel_gate.to(torch.float32).transpose(1, 2).cumsum(2)   # [B,H,T,D]
        diff = P.unsqueeze(3) - P.unsqueeze(2)                          # [B,H,Ti,Tj,D]
        scores = (
            q.unsqueeze(3) * k.unsqueeze(2) * torch.exp(diff)
        ).sum(-1) * resolved_scale                                       # [B,H,Ti,Tj]
    else:
        scores = torch.matmul(q, k.transpose(-1, -2)) * resolved_scale
    if score_bias is not None:
        scores = scores + score_bias.to(torch.float32)
    # Non-softmax reducers (threshold/squared) zero masked positions; softmax uses -inf.
    mask_value = (
        0.0 if descriptor.reducer_law is not K1ReducerLaw.SOFTMAX else float("-inf")
    )
    if descriptor.causal:
        length_q, length_k = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(length_q, length_k, dtype=torch.bool, device=scores.device).tril_(
            diagonal=length_k - length_q
        )
        scores = scores.masked_fill(~mask, mask_value)
    if attention_mask is not None:
        mask = attention_mask
        if mask.dim() == 2:
            mask = mask.view(1, 1, *mask.shape)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, mask_value)
    if descriptor.reducer_law is K1ReducerLaw.THRESHOLD_RELU_POWER:
        # out = (ReLU(s − τ))^p @ V, no denominator. τ_i = β·sqrt(2·log(i+1)/d);
        # the causal mask zeroes j>i before the threshold. i is the query position.
        d = query.shape[-1]
        i = torch.arange(1, scores.shape[-2] + 1, device=scores.device, dtype=torch.float32)
        tau = descriptor.threshold_beta * torch.sqrt(2.0 * torch.log(i) / d)  # [Ti]
        relu = torch.clamp(scores - tau.view(1, 1, -1, 1), min=0.0)
        weights = relu.pow(descriptor.relu_power)
        return torch.matmul(weights, v).transpose(1, 2).to(value.dtype)
    if descriptor.reducer_law is K1ReducerLaw.SQUARED_SUM:
        # A[t,s] = Σ_g (q_g·k_g)² over squared_sum_groups groups (the head splits
        # into groups of E = D/M). o = (Σ_s A·v_s)/max(Σ_s A, 1) — a positive score
        # with sum normalization, no softmax. Causal mask zeroed future scores above.
        d = query.shape[-1]
        M = descriptor.squared_sum_groups or 1
        E = d // M
        # Recompute the per-group dot from the un-squared q/k (scores holds the
        # full masked dot only for M==1; for M>1 we need the group decomposition).
        qg = q.view(*q.shape[:-1], M, E)                             # [B,H,Ti,M,E]
        kg = k.view(*k.shape[:-1], M, E)                             # [B,H,Tj,M,E]
        gd = torch.einsum("bhime,bhjme->bhijm", qg, kg) * resolved_scale
        A = gd.pow(2.0).sum(-1)                                      # [B,H,Ti,Tj]
        if descriptor.causal:
            length_q, length_k = A.shape[-2], A.shape[-1]
            mask = torch.ones(length_q, length_k, dtype=torch.bool, device=A.device).tril_(
                diagonal=length_k - length_q
            )
            A = A.masked_fill(~mask, 0.0)
        num = torch.matmul(A, v)                                     # [B,H,Ti,D]
        den = A.sum(dim=-1, keepdim=True).clamp(min=1.0)             # 1-safe denominator
        return (num / den).transpose(1, 2).to(value.dtype)
    probs = torch.softmax(scores, dim=-1)
    # Closed all-masked-row policy: a fully masked row returns zero.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v).transpose(1, 2).to(value.dtype)


__all__ = ["k1_softmax_attention", "torch_k1_softmax_attention"]


# Back-compat alias: the canonical name is k1_softmax_attention.
torch_k1_softmax_attention = k1_softmax_attention


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class K1TorchReferenceProvider:
    name = "urm.unified.k1.softmax_reference.v1"
    family = "k1"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        return None

    def execute(self, request, operands):
        out = k1_softmax_attention(
            operands["query"], operands["key"], operands["value"],
            descriptor=request.descriptor,
            score_bias=operands.get("score_bias"),
            attention_mask=operands.get("attention_mask"),
            scale=None if operands.get("scale") is None else float(operands["scale"]),
            channel_gate=operands.get("channel_gate"),
            gather_indices=operands.get("gather_indices"),
        )
        return {"output": out}


class K1SdpaLibraryProvider:
    name = "torch.nn.functional.scaled_dot_product_attention"
    family = "k1"
    tier = "library"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        # torch SDPA executes only the plain dot-product score with a softmax
        # reducer (plus causal/attn_mask/scale and grouped-head sharing). Decline
        # the score/reducer laws and the indexed path it cannot express rather
        # than silently running softmax.
        if request.descriptor.score_law is not K1ScoreLaw.DOT:
            return "SDPA requires the DOT score law"
        if request.descriptor.reducer_law is not K1ReducerLaw.SOFTMAX:
            return "SDPA requires the SOFTMAX reducer law"
        if request.descriptor.indexed:
            return "SDPA does not execute indexed gather"
        return None

    def execute(self, request, operands):
        import torch

        descriptor = request.descriptor
        scale_op = operands.get("scale")
        if descriptor.scale_rule is K1ScaleRule.EXPLICIT_OPERAND:
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5
        q = operands["query"].transpose(1, 2)
        k = operands["key"].transpose(1, 2)
        v = operands["value"].transpose(1, 2)

        def _head_major(mask):
            if mask.dim() == 2:
                return mask.view(1, 1, *mask.shape)
            if mask.dim() == 3:
                return mask.unsqueeze(1)
            return mask

        attn_mask = None
        attention_mask = operands.get("attention_mask")
        score_bias = operands.get("score_bias")
        if attention_mask is not None:
            attn_mask = _head_major(attention_mask)
        if score_bias is not None:
            bias = _head_major(score_bias)
            attn_mask = bias if attn_mask is None else attn_mask & bias
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            is_causal=descriptor.causal and attn_mask is None, scale=scale,
        )
        return {"output": out.transpose(1, 2)}


PROVIDERS = (K1TorchReferenceProvider(), K1SdpaLibraryProvider())
