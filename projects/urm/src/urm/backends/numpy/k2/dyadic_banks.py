"""Float64 canonical K2 banked dyadic hierarchical state recurrence (A4).

The DyadicBankedState op keeps ``num_levels−1`` additive K2-matrix states over
the token axis with a dyadic carry-cascade lifecycle (the log-linear mixer).
Slot ``S_m`` holds the aligned size-``2^m`` block of keys ending at the current
position, decayed forward each step by the per-head factor ``exp(g_t)``::

    S_m ← exp(g_t) · S_m                                (decay every slot forward)
    o_t = Σ_l level_scales[t,l] · read_l                (read, after the decay)
    S_0 ← S_0 + k_t v_tᵀ                                (rank-1 write into slot 0)
    on carry bit m of (~t & (t+1)) − 1:                 (the dyadic cascade)
        S_{m+1} ← S_{m+1} + S_m ;  S_m ← 0

The level-0 read is the diagonal ``(q_t·k_t)·v_t``; level ``l ≥ 1`` reads the
post-decay slot ``S_{l−1}`` as ``q_tᵀ S_{l−1}``. Each causal pair ``(t, j≤t)``
contributes at exactly one dyadic level (the largest aligned block containing
``j``); the union over present levels plus the diagonal is the causal prefix.

This module is the independent high-precision oracle for that equation: it is
written from the recurrence above, in pure NumPy float64, with no Torch
dependency and no gradient path. It is an oracle for representation and
composition checks, never a performance backend.
"""

from __future__ import annotations

import numpy as np

from ....ir.program import DyadicBankedState  # noqa: F401  (descriptor type)


def _inputs(query, key, value, log_decay, level_scales, num_levels):
    q, k, v, g, ls = (
        np.asarray(x, dtype=np.float64)
        for x in (query, key, value, log_decay, level_scales)
    )
    if v.ndim != 4:
        raise ValueError("query/key/value must be [B, T, H, D] tensors")
    bsz, t, heads, dim = v.shape
    if q.shape != v.shape or k.shape != v.shape:
        raise ValueError("query, key and value must share shape [B, T, H, D]")
    if g.shape != (bsz, t, heads):
        raise ValueError("log_decay must be a [B, T, H] tensor")
    if ls.shape != (bsz, t, heads, num_levels):
        raise ValueError("level_scales must be a [B, T, H, num_levels] tensor")
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    if any(not np.isfinite(x).all() for x in (q, k, v, g, ls)):
        raise ValueError("inputs must be finite")
    return q, k, v, g, ls


def dyadic_banked_state_forward(query, key, value, log_decay, level_scales,
                                num_levels):
    """Compute the banked dyadic hierarchical state output in float64.

    Same operands and shapes as every tier: ``query``/``key``/``value`` are
    ``[B, T, H, D]``, ``log_decay`` is the per-head log decay ``[B, T, H]`` and
    ``level_scales`` the per-token per-level output scales
    ``[B, T, H, num_levels]``. Returns the output ``[B, T, H, D]``. Runs in
    float64 (the oracle tier).
    """
    num_levels = int(num_levels)
    q, k, v, g, ls = _inputs(query, key, value, log_decay, level_scales,
                             num_levels)
    bsz, t, heads, dim = v.shape
    nbank = num_levels - 1
    out = np.zeros_like(v)
    slots = [np.zeros((bsz, heads, dim, dim), dtype=np.float64)
             for _ in range(nbank)]
    for i in range(t):
        decay = np.exp(g[:, i, :])[:, :, None, None]          # [B,H,1,1]
        for m in range(nbank):
            slots[m] = decay * slots[m]
        # Read: the level-0 diagonal, then level l ≥ 1 from post-decay slot l−1.
        score0 = np.sum(q[:, i] * k[:, i], axis=-1)           # [B,H]
        out[:, i] = (ls[:, i, :, 0] * score0)[..., None] * v[:, i]
        for level in range(1, num_levels):
            contrib = np.einsum("bhk,bhkv->bhv", q[:, i], slots[level - 1])
            out[:, i] += ls[:, i, :, level][..., None] * contrib
        # Write the rank-1 k_i v_iᵀ into slot 0, then the carry cascade.
        slots[0] = slots[0] + np.einsum("bhk,bhv->bhkv", k[:, i], v[:, i])
        check = (~i & (i + 1)) - 1
        for m in range(nbank - 1):
            if check & (1 << m):
                slots[m + 1] = slots[m + 1] + slots[m]
                slots[m] = np.zeros((bsz, heads, dim, dim), dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class K2DyadicBankedStateNumpyProvider:
    name = "urm.reference.numpy.k2.dyadic_banked_state.v1"
    family = "k2"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, DyadicBankedState):
            return "K2 NumPy provider requires a DyadicBankedState op descriptor"
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


PROVIDERS = (K2DyadicBankedStateNumpyProvider(),)

__all__ = ["K2DyadicBankedStateNumpyProvider", "dyadic_banked_state_forward"]
