"""Differentiable PyTorch reference for the K1 softmax-attention equation.

This is the Torch tier of the same K1 contract as the NumPy oracle
(:mod:`urm.backends.providers.k1.numpy`) and the native Triton
schedule (:mod:`urm.backends.providers.k1.triton`). The equation is carried by the
closed :class:`~urm.ir.program.K1Descriptor`: score scale law, causal masking,
grouped head sharing and the all-masked-row-zero policy. Runtime code owns no
part of this equation; it only binds role operands and invokes a provider.
"""

from __future__ import annotations

from typing import Any

from ....ir.program import K1Descriptor, K1ScaleRule


def _torch() -> Any:
    import torch

    return torch


def torch_k1_softmax_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    descriptor: K1Descriptor,
    score_bias: Any | None = None,
    attention_mask: Any | None = None,
    scale: float | None = None,
) -> Any:
    """Canonical K1 softmax attention over ``[B, T, H, D]`` operands.

    The score scale follows the descriptor's scale law (never a runtime
    default): the key-dim rule or an explicit operand. Grouped query→KV head
    sharing expands shared KV heads. A fully masked row returns zero.
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
    scores = torch.matmul(q, k.transpose(-1, -2)) * resolved_scale
    if score_bias is not None:
        scores = scores + score_bias.to(torch.float32)
    if descriptor.causal:
        length_q, length_k = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(length_q, length_k, dtype=torch.bool, device=scores.device).tril_(
            diagonal=length_k - length_q
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    if attention_mask is not None:
        mask = attention_mask
        if mask.dim() == 2:
            mask = mask.view(1, 1, *mask.shape)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # Closed all-masked-row policy: a fully masked row returns zero.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v).transpose(1, 2).to(value.dtype)


__all__ = ["torch_k1_softmax_attention"]
