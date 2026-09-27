"""URM-owned Triton kernels for the typed DyadicBankedState ordinary operator (A4).

The banked dyadic hierarchical state law (the log-linear mixer): a bank of
``num_levels−1`` additive K2-matrix states over the token axis with the dyadic
carry-cascade lifecycle. Slot ``S_m`` holds the aligned size-``2^m`` block of keys
ending at the current position, decayed forward each step by the per-head factor
``exp(g_t)``; the read combines the level-0 diagonal with the per-level
contributions ``level_scales[t,l]·q_tᵀ S_{l-1}``; the write adds the rank-1
``k_t v_tᵀ`` to slot 0 and promotes ``S_m → S_{m+1}`` (resetting ``S_m``) on the
carry bits ``(~t & (t+1)) − 1``. Decay-forward form, fp32 accumulation.

Schedule: one program owns one (batch, head) pair and one value-dim block and
walks the token axis in order (forward) or reverse (backward); the bank slots
live in a program-exclusive slice of a fp32 global state buffer, so the ordered
cross-token lifecycle is structural — no atomics in the forward. The backward
carries the slot adjoints in a second program-exclusive buffer and reduces the
per-token operand cotangents (q/k/log_decay/level_scales) across value blocks
with relaxed atomics; value cotangents are program-owned. Parity vs the pinned
torch reference (``urm.backends.torch.dyadic_banked_state``) is gated forward
and cotangent in ``tests/test_native_dyadic_banked_state.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NATIVE_DYADIC_BANKED_STATE_NAME = "urm_native_dyadic_banked_state_v1"


@triton.jit
def _dyadic_banked_forward_kernel(
    query,
    key,
    value,
    log_decay,
    level_scales,
    out,
    state,
    saved,
    H,
    T,
    D,
    LEVELS: tl.constexpr,
    NBANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SAVE_STATES: tl.constexpr,
):
    bh = tl.program_id(0)
    vb = tl.program_id(1)
    b = bh // H
    h = bh % H
    koffs = tl.arange(0, BLOCK_K)
    voffs = vb * BLOCK_D + tl.arange(0, BLOCK_D)
    kmask = koffs < D
    vmask = voffs < D
    mask2 = kmask[:, None] & vmask[None, :]
    state_base = state + bh.to(tl.int64) * NBANK * D * D
    saved_base = saved + bh.to(tl.int64) * T * NBANK * D * D
    for t in tl.range(0, T):
        tok = (b * T + t) * H + h
        dec = tl.exp(tl.load(log_decay + tok).to(tl.float32))
        q_vec = tl.load(query + tok * D + koffs, mask=kmask, other=0.0).to(tl.float32)
        k_vec = tl.load(key + tok * D + koffs, mask=kmask, other=0.0).to(tl.float32)
        v_vec = tl.load(value + tok * D + voffs, mask=vmask, other=0.0).to(tl.float32)
        # Level-0 diagonal: level_scales[t,0]·(q_t·k_t)·v_t.
        ls0 = tl.load(level_scales + tok * LEVELS).to(tl.float32)
        acc = ls0 * tl.sum(q_vec * k_vec, axis=0) * v_vec
        # Decay every slot forward, then read level m+1 from slot m.
        for m in tl.static_range(NBANK):
            sptr = state_base + m * D * D + koffs[:, None] * D + voffs[None, :]
            tile = tl.load(sptr, mask=mask2, other=0.0)
            tile = tile * dec
            tl.store(sptr, tile, mask=mask2)
            if SAVE_STATES:
                # The post-decay slot (what the read saw); the backward needs it
                # for the query/scale cotangents and the log-decay contraction.
                tl.store(
                    saved_base + (t * NBANK + m) * D * D
                    + koffs[:, None] * D + voffs[None, :],
                    tile,
                    mask=mask2,
                )
            ls_m = tl.load(level_scales + tok * LEVELS + (m + 1)).to(tl.float32)
            acc += ls_m * tl.sum(q_vec[:, None] * tile, axis=0)
        tl.store(out + tok * D + voffs, acc, mask=vmask)
        if NBANK > 0:
            # Write the rank-1 k_t v_tᵀ into slot 0, then the carry cascade
            # (pinned rule check = (~t & (t+1)) − 1: bit m promotes S_m → S_{m+1}
            # and resets S_m; ascending m so a multi-bit carry chains).
            wptr = state_base + koffs[:, None] * D + voffs[None, :]
            tile0 = tl.load(wptr, mask=mask2, other=0.0)
            tl.store(wptr, tile0 + k_vec[:, None] * v_vec[None, :], mask=mask2)
            check = (~t & (t + 1)) - 1
            for m in tl.static_range(NBANK - 1):
                pm = ((check >> m) & 1).to(tl.float32)
                lo = state_base + m * D * D + koffs[:, None] * D + voffs[None, :]
                hi = lo + D * D
                t_lo = tl.load(lo, mask=mask2, other=0.0)
                t_hi = tl.load(hi, mask=mask2, other=0.0)
                tl.store(hi, t_hi + pm * t_lo, mask=mask2)
                tl.store(lo, t_lo * (1.0 - pm), mask=mask2)


@triton.jit
def _dyadic_banked_backward_kernel(
    grad_out,
    query,
    key,
    value,
    log_decay,
    level_scales,
    saved,
    grad_state,
    grad_query,
    grad_key,
    grad_value,
    grad_log_decay,
    grad_level_scales,
    H,
    T,
    D,
    LEVELS: tl.constexpr,
    NBANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    vb = tl.program_id(1)
    b = bh // H
    h = bh % H
    koffs = tl.arange(0, BLOCK_K)
    voffs = vb * BLOCK_D + tl.arange(0, BLOCK_D)
    kmask = koffs < D
    vmask = voffs < D
    mask2 = kmask[:, None] & vmask[None, :]
    gs_base = grad_state + bh.to(tl.int64) * NBANK * D * D
    sv_base = saved + bh.to(tl.int64) * T * NBANK * D * D
    # Reverse-time adjoint scan: grad_state holds the adjoint of the post-cascade
    # slots; the op returns no terminal state, so it starts at zero.
    for i in tl.range(0, T):
        t = T - 1 - i
        tok = (b * T + t) * H + h
        dec = tl.exp(tl.load(log_decay + tok).to(tl.float32))
        q_vec = tl.load(query + tok * D + koffs, mask=kmask, other=0.0).to(tl.float32)
        k_vec = tl.load(key + tok * D + koffs, mask=kmask, other=0.0).to(tl.float32)
        v_vec = tl.load(value + tok * D + voffs, mask=vmask, other=0.0).to(tl.float32)
        do = tl.load(grad_out + tok * D + voffs, mask=vmask, other=0.0).to(tl.float32)
        gq = tl.zeros((BLOCK_K,), dtype=tl.float32)
        gk = tl.zeros((BLOCK_K,), dtype=tl.float32)
        # Undo the carry cascade (descending m): the adjoint of
        # S_{m+1} += S_m; S_m = 0 is dS_m = pm·dS_{m+1} (+ (1−pm)·dS_m).
        check = (~t & (t + 1)) - 1
        for i_m in tl.static_range(NBANK - 1):
            m = NBANK - 2 - i_m
            pm = ((check >> m) & 1).to(tl.float32)
            lo = gs_base + m * D * D + koffs[:, None] * D + voffs[None, :]
            hi = lo + D * D
            d_lo = tl.load(lo, mask=mask2, other=0.0)
            d_hi = tl.load(hi, mask=mask2, other=0.0)
            tl.store(lo, pm * d_hi + (1.0 - pm) * d_lo, mask=mask2)
        if NBANK > 0:
            # Undo the rank-1 write S_0 += k_t v_tᵀ: the post-write slot-0
            # adjoint contracts with v (for dk) and k (for dv), and flows on
            # unchanged into the pre-write slot adjoint.
            wptr = gs_base + koffs[:, None] * D + voffs[None, :]
            d_c0 = tl.load(wptr, mask=mask2, other=0.0)
            gk += tl.sum(d_c0 * v_vec[None, :], axis=1)
            gv = tl.sum(d_c0 * k_vec[:, None], axis=0)
        else:
            gv = tl.zeros((BLOCK_D,), dtype=tl.float32)
        # Undo the read (level m+1 read slot m) and the decay. dS_m currently
        # holds the adjoint of the post-decay slot B_m; add the read's outer
        # product, then scale by exp(g_t) into the pre-decay adjoint, and
        # contract ⟨dB_m, B_m⟩ into the log-decay cotangent.
        gg = 0.0
        for m in tl.static_range(NBANK):
            sptr = sv_base + (t * NBANK + m) * D * D + koffs[:, None] * D + voffs[None, :]
            tile = tl.load(sptr, mask=mask2, other=0.0)
            gptr = gs_base + m * D * D + koffs[:, None] * D + voffs[None, :]
            d_b = tl.load(gptr, mask=mask2, other=0.0)
            ls_m = tl.load(level_scales + tok * LEVELS + (m + 1)).to(tl.float32)
            d_b += ls_m * q_vec[:, None] * do[None, :]
            gq += ls_m * tl.sum(tile * do[None, :], axis=1)
            gls_m = tl.sum(tile * q_vec[:, None] * do[None, :])
            tl.atomic_add(
                grad_level_scales + tok * LEVELS + (m + 1), gls_m, sem="relaxed"
            )
            gg += tl.sum(d_b * tile)
            tl.store(gptr, d_b * dec, mask=mask2)
        # Level-0 diagonal cotangents.
        ls0 = tl.load(level_scales + tok * LEVELS).to(tl.float32)
        s0 = tl.sum(q_vec * k_vec, axis=0)
        dov = tl.sum(do * v_vec, axis=0)
        gq += ls0 * dov * k_vec
        gk += ls0 * dov * q_vec
        gv += ls0 * s0 * do
        tl.atomic_add(grad_level_scales + tok * LEVELS, s0 * dov, sem="relaxed")
        if NBANK > 0:
            tl.atomic_add(grad_log_decay + tok, gg, sem="relaxed")
        tl.atomic_add(grad_query + tok * D + koffs, gq, mask=kmask, sem="relaxed")
        tl.atomic_add(grad_key + tok * D + koffs, gk, mask=kmask, sem="relaxed")
        tl.store(grad_value + tok * D + voffs, gv, mask=vmask)


def _launch_parameters(key_dim: int, value_block: int) -> tuple[int, int, int]:
    block_k = max(16, triton.next_power_of_2(key_dim))
    block_d = min(max(16, triton.next_power_of_2(value_block)), 64)
    elems = block_k * block_d
    warps = 8 if elems >= 8192 else 4 if elems >= 1024 else 2
    return block_k, block_d, warps


def _dyadic_banked_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    level_scales: torch.Tensor,
    num_levels: int,
    *,
    save_states: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, D = value.shape
    nbank = num_levels - 1
    block_k, block_d, warps = _launch_parameters(D, D)
    out = torch.empty((B, T, H, D), device=value.device, dtype=torch.float32)
    if nbank:
        state = torch.zeros((B * H, nbank, D, D), device=value.device, dtype=torch.float32)
        saved = (
            torch.empty((B * H, T, nbank, D, D), device=value.device, dtype=torch.float32)
            if save_states
            else state
        )
    else:
        state = saved = out
    grid = (B * H, triton.cdiv(D, block_d))
    _dyadic_banked_forward_kernel[grid](
        query,
        key,
        value,
        log_decay,
        level_scales,
        out,
        state,
        saved,
        H,
        T,
        D,
        LEVELS=num_levels,
        NBANK=nbank,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        SAVE_STATES=save_states and nbank > 0,
        num_warps=warps,
    )
    return out, saved if save_states and nbank > 0 else None


def _dyadic_banked_backward(
    grad_out: torch.Tensor,
    saved: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    level_scales: torch.Tensor,
    num_levels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = value.shape
    nbank = num_levels - 1
    block_k, block_d, warps = _launch_parameters(D, D)
    grad_query = torch.zeros((B, T, H, D), device=value.device, dtype=torch.float32)
    grad_key = torch.zeros((B, T, H, D), device=value.device, dtype=torch.float32)
    grad_value = torch.empty((B, T, H, D), device=value.device, dtype=torch.float32)
    grad_log_decay = torch.zeros((B, T, H), device=value.device, dtype=torch.float32)
    grad_level_scales = torch.zeros(
        (B, T, H, num_levels), device=value.device, dtype=torch.float32
    )
    grad_state = (
        torch.zeros((B * H, nbank, D, D), device=value.device, dtype=torch.float32)
        if nbank
        else grad_value
    )
    grid = (B * H, triton.cdiv(D, block_d))
    _dyadic_banked_backward_kernel[grid](
        grad_out.contiguous(),
        query,
        key,
        value,
        log_decay,
        level_scales,
        saved if nbank else grad_value,
        grad_state,
        grad_query,
        grad_key,
        grad_value,
        grad_log_decay,
        grad_level_scales,
        H,
        T,
        D,
        LEVELS=num_levels,
        NBANK=nbank,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=warps,
    )
    return grad_query, grad_key, grad_value, grad_log_decay, grad_level_scales


class _DyadicBankedState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, log_decay, level_scales, num_levels):
        out, saved = _dyadic_banked_forward(
            query, key, value, log_decay, level_scales, num_levels, save_states=True
        )
        ctx.save_for_backward(saved, query, key, value, log_decay, level_scales)
        ctx.num_levels = num_levels
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved, query, key, value, log_decay, level_scales = ctx.saved_tensors
        gradients = _dyadic_banked_backward(
            grad_out,
            saved,
            query,
            key,
            value,
            log_decay,
            level_scales,
            ctx.num_levels,
        )
        return (*gradients, None)


def dyadic_banked_state(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    level_scales: torch.Tensor,
    num_levels: int,
) -> torch.Tensor:
    """The native banked dyadic hierarchical state output, fp32 accumulation.

    Same operands and shapes as the torch reference
    (:func:`urm.backends.torch.dyadic_banked_state.dyadic_banked_state_forward`):
    ``query``/``key``/``value`` are ``[B, T, H, D]``, ``log_decay`` is
    ``[B, T, H]``, ``level_scales`` is ``[B, T, H, num_levels]``; returns the
    fp32 output ``[B, T, H, D]``. Differentiable w.r.t. every operand.
    """
    operands = tuple(
        tensor.to(torch.float32).contiguous()
        for tensor in (query, key, value, log_decay, level_scales)
    )
    if torch.is_grad_enabled() and any(t.requires_grad for t in operands):
        return _DyadicBankedState.apply(*operands, int(num_levels))
    out, _ = _dyadic_banked_forward(*operands, int(num_levels), save_states=False)
    return out


__all__ = ["dyadic_banked_state"]


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class DyadicBankedStateNativeTritonProvider:
    name = NATIVE_DYADIC_BANKED_STATE_NAME
    family = "dyadic_banked_state"
    tier = "native"

    def decline(self, request) -> str | None:
        from ....ir.program import DyadicBankedState

        if not isinstance(request.descriptor, DyadicBankedState):
            return "dyadic_banked_state providers require a DyadicBankedState op descriptor"
        if request.accumulation_dtype != "float32":
            return "dyadic_banked_state v1 requires float32 accumulation"
        if not torch.cuda.is_available():
            return "native dyadic_banked_state requires CUDA"
        return None

    def execute(self, request, operands):
        out = dyadic_banked_state(
            operands["query"],
            operands["key"],
            operands["value"],
            operands["log_decay"],
            operands["level_scales"],
            request.descriptor.num_levels,
        )
        return {"output": out}


PROVIDERS = (DyadicBankedStateNativeTritonProvider(),)
