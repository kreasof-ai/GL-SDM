"""Torch reference provider for the typed DyadicBankedState ordinary operator.

The DyadicBankedState op (generality axis A4) computes the banked dyadic
hierarchical state law — the log-linear mixer. It keeps ``num_levels−1`` additive
K2-matrix states over the token axis with a dyadic carry-cascade lifecycle: slot
``S_m`` holds the aligned size-``2^m`` block of keys ending at the current
position, decayed forward each step by the per-head factor ``exp(g_t)``; the read
combines the per-level contributions ``level_scales[t,l]·q_tᵀ S_{l-1}`` plus the
level-0 diagonal; the write adds the rank-1 ``k_t v_tᵀ`` to slot 0 and promotes
``S_m → S_{m+1}`` (resetting ``S_m``) on the carry bits ``(~t & (t+1)) − 1``.

Each causal pair ``(t, j≤t)`` contributes at exactly one dyadic level (the largest
aligned block containing ``j``); the union over present levels plus the diagonal
is the causal prefix. This is a *bank* of states with a conditional promote/reset
lifecycle — structurally distinct from the single fixed-address K2 matrix, so it
is its own ordinary typed operator, unfused by default. The decay-forward form is
numerically safe for long sequences (no global ``exp(−gcum)`` prefix to overflow).
Verified against the pinned ``naive_log_linear_attn`` and the disjoint
dyadic-block decomposition.
"""

from __future__ import annotations

from typing import Any

from ...ir.program import DyadicBankedState  # noqa: F401  (descriptor type)


def dyadic_banked_state_forward(
    query: Any, key: Any, value: Any, log_decay: Any, level_scales: Any, num_levels: int
) -> Any:
    """Compute the banked dyadic hierarchical state output.

    ``query``/``key``/``value`` are ``[B, T, H, D]`` (the log-linear layer layout);
    ``log_decay`` is the per-head log decay ``g`` ``[B, T, H]``; ``level_scales`` is
    the per-token per-level output scale ``[B, T, H, num_levels]``. Returns the
    output ``[B, T, H, D]``. Runs in float32.

    Slot ``S_m`` (``m = 0 .. num_levels−2``) holds the aligned size-``2^m`` block of
    keys ending at the current position, decayed to the current time. The carry
    cascade uses the pinned rule ``check = (~t & (t+1)) − 1``: when bit ``m`` of
    ``check`` is set, slot ``m`` promotes into slot ``m+1`` and resets.
    """
    torch = __import__("torch")
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    g = log_decay.to(torch.float32)
    ls = level_scales.to(torch.float32)
    B, T, H, D = v.shape
    num_levels = int(num_levels)
    nbank = num_levels - 1
    out = torch.zeros(B, T, H, D, dtype=torch.float32, device=v.device)
    S = [torch.zeros(B, H, D, D, dtype=torch.float32, device=v.device) for _ in range(nbank)]
    for t in range(T):
        # Decay every slot forward by the per-head factor exp(g_t).
        dec = torch.exp(g[:, t, :]).reshape(B, H, 1, 1)          # [B,H,1,1]
        for m in range(nbank):
            S[m] = dec * S[m]
        # Read: level 0 diagonal + levels l>=1 from slot l-1.
        score0 = (q[:, t] * k[:, t]).sum(-1)                     # [B,H]
        out[:, t] += (ls[:, t, :, 0] * score0).unsqueeze(-1) * v[:, t]
        for l in range(1, num_levels):
            contrib = torch.einsum("bhk,bhkv->bhv", q[:, t], S[l - 1])  # [B,H,D]
            out[:, t] += ls[:, t, :, l].unsqueeze(-1) * contrib
        # Write the rank-1 k_t v_tᵀ into slot 0, then carry-cascade.
        S[0] = S[0] + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        check = (~t & (t + 1)) - 1
        for m in range(nbank - 1):
            if check & (1 << m):
                S[m + 1] = S[m + 1] + S[m]
                S[m] = torch.zeros(B, H, D, D, dtype=torch.float32, device=v.device)
    return out


class DyadicBankedStateTorchReferenceProvider:
    name = "urm.unified.dyadic_banked_state.reference.v1"
    family = "dyadic_banked_state"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, DyadicBankedState):
            return "dyadic_banked_state providers require a DyadicBankedState op descriptor"
        if request.accumulation_dtype != "float32":
            return "dyadic_banked_state v1 requires float32 accumulation"
        return None

    def execute(self, request, operands):
        out = dyadic_banked_state_forward(
            operands["query"],
            operands["key"],
            operands["value"],
            operands["log_decay"],
            operands["level_scales"],
            request.descriptor.num_levels,
        )
        return {"output": out}


PROVIDERS = (DyadicBankedStateTorchReferenceProvider(),)

__all__ = ["DyadicBankedStateTorchReferenceProvider", "dyadic_banked_state_forward"]
