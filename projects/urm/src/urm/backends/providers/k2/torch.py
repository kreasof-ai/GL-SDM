"""Differentiable PyTorch reference for the K2 linear-delta state law.

This is the Torch tier of the same typed request/result ABI as the NumPy
oracle (:mod:`urm.backends.providers.k2.numpy`) and the native Triton
schedules: NumPy supplies the independent high-precision equation, Torch the
transparent differentiable reference. Both implement the canonical law in
docs/kernels/linear-delta.md — decay precedes retrieval, the canonical read is
after the update, gate scope (none/scalar/head/channel) is a semantic field,
and a normalized variant carries an explicit denominator state. Unsupported
descriptor combinations raise rather than silently degrade.
"""

from __future__ import annotations

from typing import Any

from ....ir.program import K2GateScope, K2ReadTiming, LinearDeltaSpec


def _torch() -> Any:
    import torch

    return torch


def torch_linear_delta_state(
    initial_state: Any,
    keys: Any,
    queries: Any,
    values: Any,
    beta: Any,
    log_decay: Any,
    *,
    spec: LinearDeltaSpec,
    scale: float | None = None,
) -> tuple[Any, Any] | tuple[Any, tuple[Any, Any]]:
    """Run the canonical K2 recurrence over a token sequence.

    Shapes: ``initial_state`` ``[B, H, K, V]``; ``keys``/``queries``
    ``[B, H, T, K]``; ``values`` ``[B, H, T, V]``; ``beta`` ``[B, H, T]`` or
    ``[B, H, T, 1]``; ``log_decay`` ``[B, H, T]`` (scalar/head scope) or
    ``[B, H, T, K]`` (channel scope). All computation runs in float32; the
    result is cast back to the input dtype. Returns ``(out, final_state)`` or,
    for a normalized spec, ``(out, (final_state, final_denominator))``.
    """
    torch = _torch()
    dtype = values.dtype
    m = initial_state.to(torch.float32).clone()
    k = keys.to(torch.float32)
    q = queries.to(torch.float32)
    v = values.to(torch.float32)
    b = beta.to(torch.float32)
    g = log_decay.to(torch.float32)
    # beta/log_decay may arrive as [B, H, T, 1]; squeeze only a trailing
    # singleton that is NOT the token axis (i.e. only when 4-D).
    if b.dim() == 4 and b.shape[-1] == 1:
        b = b.squeeze(-1)
    if g.dim() == 4 and g.shape[-1] == 1:
        g = g.squeeze(-1)

    if spec.gate_scope is K2GateScope.NONE:
        # No decay: G_t = I, expressed as an explicit zero schedule so the
        # recurrence stays a proper autograd chain (never z aliasing m).
        g = torch.zeros(m.shape[0], m.shape[1], v.shape[2], dtype=torch.float32, device=m.device)
    elif spec.gate_scope in (K2GateScope.SCALAR, K2GateScope.HEAD):
        if g is not None and g.dim() != 3:
            raise ValueError("scalar/head decay expects log_decay [B, H, T]")
    elif spec.gate_scope is K2GateScope.CHANNEL:
        if g is not None and g.dim() != 4:
            raise ValueError("channel decay expects log_decay [B, H, T, K]")
    else:  # pragma: no cover - enum is closed
        raise ValueError(f"unsupported gate scope {spec.gate_scope}")

    if spec.scale_rule.value == "one":
        resolved_scale = 1.0
    elif spec.scale_rule.value == "key_dim_rsqrt":
        resolved_scale = k.shape[-1] ** -0.5
    else:  # explicit_operand
        if scale is None:
            raise ValueError("scale_rule=explicit_operand requires a scale value")
        resolved_scale = float(scale)

    norm = None
    if spec.normalized:
        norm = torch.zeros(m.shape[0], m.shape[1], m.shape[2], dtype=torch.float32, device=m.device)

    T = v.shape[2]
    outs = []
    for t in range(T):
        if spec.gate_scope is K2GateScope.CHANNEL:
            decay = torch.exp(g[:, :, t]).unsqueeze(-1)  # [B,H,K,1]
            z = decay * m
            norm_decay = decay.squeeze(-1)  # [B,H,K]
        else:
            decay_k = torch.exp(g[:, :, t]).unsqueeze(-1)  # [B,H,1] key-domain broadcast
            z = decay_k.unsqueeze(-1) * m
            norm_decay = decay_k.expand(-1, -1, m.shape[2])  # [B,H,K] for the denominator state
        if norm is not None and norm_decay is not None:
            norm = norm_decay * norm  # decay the denominator with the state

        # h_t = k_tᵀ Z_t : contract the key over the state's key axis.
        retr = torch.einsum("bhk,bhkv->bhv", k[:, :, t], z)  # [B,H,V]
        if spec.delta:
            delta = b[:, :, t].unsqueeze(-1) * (v[:, :, t] - retr)
        else:
            delta = v[:, :, t]
        m = z + k[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
        if norm is not None:
            norm = norm + k[:, :, t]

        if spec.read_timing is K2ReadTiming.BEFORE_UPDATE:
            read_state = z
            read_norm = norm - k[:, :, t] if norm is not None else None
        else:
            read_state = m
            read_norm = norm
        y = resolved_scale * torch.einsum("bhk,bhkv->bhv", q[:, :, t], read_state)
        if norm is not None:
            denom = (q[:, :, t] * read_norm).sum(-1, keepdim=True).clamp_min(spec.epsilon)
            y = y / denom
        outs.append(y)
    out = torch.stack(outs, dim=2).to(dtype)  # [B,H,T,V]
    if norm is not None:
        return out, (m.to(dtype), norm.to(dtype))
    return out, m.to(dtype)


__all__ = ["torch_linear_delta_state"]
