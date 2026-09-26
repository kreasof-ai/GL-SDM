"""Native Triton K1 channel-decay score law (Wall, A13).

The channel-decay score: s_ij = Σ_n q_in · k_jn · exp(P_in − P_jn), where
P = cumsum(channel_gate) over time. The key identity: exp(P_in − P_jn) =
exp(P_in)/exp(P_jn), so the score is a pre-scaled dot product:
  s_ij = Σ_n (q_in · exp(P_in)) · (k_jn / exp(P_jn)) = Σ_n q'_in · k'_jn.

The kernel computes P (the per-position per-channel gate cumsum) in a pre-pass,
then the standard online-softmax kernel runs on the pre-scaled q'/k' operands.
The pre-scaling is external to the K1 kernel — the channel_decay provider wraps
the online-softmax kernel with the scaling.

The reducer is the standard softmax (the score law is the only difference).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _channel_decay_prescale(
    GATE,  # [B, T, H, D] raw per-channel log gate
    Q_OUT,  # [B, T, H, D] pre-scaled query: q · exp(P)
    K_OUT,  # [B, T, H, D] pre-scaled key: k / exp(P)
    Q_IN, K_IN,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute P = cumsum(gate) over time, then scale q and k.

    One program per (batch, head, channel-block): walks the time axis sequentially,
    accumulating the running cumsum P_t = P_{t-1} + gate_t, and emits the scaled
    operands q'_t = q_t · exp(P_t), k'_t = k_t / exp(P_t).
    """
    batch = tl.program_id(0)
    head = tl.program_id(1)
    d_start = tl.program_id(2) * BLOCK
    d_offs = d_start + tl.arange(0, BLOCK)
    d_valid = d_offs < D

    P = tl.zeros((BLOCK,), tl.float32)
    for t in range(T):
        base = ((batch * T + t) * H + head) * D + d_offs
        g = tl.load(GATE + base, d_valid, other=0.0).to(tl.float32)
        P = P + g
        exp_P = tl.exp(P)
        q = tl.load(Q_IN + base, d_valid, other=0.0).to(tl.float32)
        k = tl.load(K_IN + base, d_valid, other=0.0).to(tl.float32)
        tl.store(Q_OUT + base, q * exp_P, d_valid)
        tl.store(K_OUT + base, k / tl.maximum(exp_P, 1e-30), d_valid)


def channel_decay_prescale(
    query: torch.Tensor, key: torch.Tensor, channel_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-scale q/k by the channel-decay factors: q' = q·exp(P), k' = k/exp(P).

    ``channel_gate`` is the per-channel log gate ``[B, T, H, D]`` (pre-cumsum).
    Returns the pre-scaled query and key at the same shapes.
    """
    B, T, H, D = query.shape
    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)
    BLOCK = min(256, triton.next_power_of_2(D))
    grid = (B, H, triton.cdiv(D, BLOCK))
    _channel_decay_prescale[grid](
        channel_gate.contiguous(), q_out, k_out,
        query.contiguous(), key.contiguous(),
        B=B, T=T, H=H, D=D, BLOCK=BLOCK,
        num_warps=4,
    )
    return q_out, k_out


class K1NativeChannelDecayProvider:
    name = "urm_native_k1_channel_decay_v1"
    family = "k1"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        if request.descriptor.score_law is not K1ScoreLaw.CHANNEL_DECAY:
            return "native K1 channel-decay requires the CHANNEL_DECAY score law"
        if request.descriptor.reducer_law is not K1ReducerLaw.SOFTMAX:
            return "native K1 channel-decay requires the SOFTMAX reducer law"
        if request.descriptor.indexed:
            return "native K1 channel-decay does not execute indexed gather"
        import torch
        if not torch.cuda.is_available():
            return "native K1 requires CUDA"
        return None

    def execute(self, request, operands):
        from .online_softmax import k1_softmax_attention
        from ....ir.program import K1ScaleRule as _SR

        desc = request.descriptor
        query, key, value = operands["query"], operands["key"], operands["value"]
        channel_gate = operands["channel_gate"]
        if desc.scale_rule is _SR.EXPLICIT_OPERAND:
            scale = float(operands["scale"])
        else:
            scale = float(query.shape[-1]) ** -0.5

        # Pre-scale q/k by the channel-decay factors, then run the standard
        # online-softmax kernel on the scaled operands.
        q_scaled, k_scaled = channel_decay_prescale(query, key, channel_gate)

        # The descriptor for the inner call is the same but with DOT score law
        # (the channel-decay is folded into the pre-scaling).
        from ....ir.program import K1Descriptor as _KD, K1ScoreLaw as _SL
        inner_desc = _KD(
            scale_rule=desc.scale_rule,
            head_map=desc.head_map,
            group_size=desc.group_size,
            causal=desc.causal,
            score_bias=desc.score_bias,
            attention_mask=desc.attention_mask,
            score_law=_SL.DOT,  # the decay is in the pre-scaled operands
            reducer_law=desc.reducer_law,
            masked_row=desc.masked_row,
            accumulation_dtype=desc.accumulation_dtype,
        )
        out = k1_softmax_attention(
            q_scaled, k_scaled, value,
            descriptor=inner_desc,
            score_bias=operands.get("score_bias"),
            attention_mask=operands.get("attention_mask"),
            scale=scale,
        )
        return {"output": out}


__all__ = ["K1NativeChannelDecayProvider", "channel_decay_prescale"]
