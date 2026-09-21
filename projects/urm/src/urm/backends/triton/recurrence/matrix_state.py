"""Fused Triton scan for the matrix-state K2 recurrence family.

This is the native generator for the K2 matrix-state class - the single largest
recurrence group in the catalog. One reusable kernel covers the whole class; the
compiler selects the decay granularity, update rule, and read timing from the
semantic spec, never from an architecture name. The equation per token is the
``urm/ir/recurrence.py`` contract::

    Z_t    = decay(G_t) * M_{t-1}         (decay broadcast by granularity)
    h_t    = k_t^T Z_t                     (retrieved; delta rule only)
    delta_t = beta_t * (v_t - c * h_t)     (c=1 delta, c=0 additive)
    M_t    = Z_t + k_t delta_t^T
    y_t    = scale * q_t^T M_t             (read before or after the update)

Decay granularity (how ``exp(log_decay)`` broadcasts over the ``[K, V]`` state):

- ``none``:          no decay.
- ``head``:          one scalar per head, broadcast over ``[K, V]``.
- ``key_channel``:   one per key channel, broadcast over ``V``.
- ``value_channel``: one per value channel, broadcast over ``K``.

The state ``M`` is a per-head ``[K, V]`` matrix held in fp32. One program owns
one ``(batch, head)`` pair and scans the sequence, so the recurrence is exact
(no chunked approximation). Forward stores the per-token states so the backward
pass runs an exact reverse scan. This covers the delta and additive matrix-state
recurrences (gated-delta, GLA, DeltaNet, and the broader class) natively.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Decay granularity encodings passed to the kernel as constexpr.
_DECAY_NONE = 0
_DECAY_HEAD = 1
_DECAY_KEY_CHANNEL = 2
_DECAY_VALUE_CHANNEL = 3


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def forward_kernel(
        Q,
        K,
        V,
        G,  # log_decay: [B,T,H] (head), [B,T,H,K] (key_channel), [B,T,H,V] (value_channel)
        BETA,  # [B,T,H] or None
        INITIAL,
        OUTPUT,
        STATES,
        FINAL,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        DECAY: tl.constexpr,
        IS_DELTA: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_base = ((batch * H + head) * K_DIM) * V_DIM
        state_offset = state_base + k_index[:, None] * V_DIM + v_index[None, :]
        if HAS_INITIAL:
            state = tl.load(INITIAL + state_offset, kv_mask, other=0.0).to(tl.float32)
        else:
            state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        qk_token_base = batch * (T * H * K_DIM) + head * K_DIM
        v_token_base = batch * (T * H * V_DIM) + head * V_DIM
        gb_token_base = batch * (T * H) + head
        for token in range(T):
            k_t = tl.load(
                K + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(
                V + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)
            q_t = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            # Decay broadcast by granularity.
            if DECAY == 1:  # head: scalar per head
                g_t = tl.load(G + gb_token_base + token * H).to(tl.float32)
                state = tl.exp(g_t) * state
            elif DECAY == 2:  # key_channel: per-K, broadcast over V
                g_t = tl.load(
                    G + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[:, None] * state
            elif DECAY == 3:  # value_channel: per-V, broadcast over K
                g_t = tl.load(
                    G + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[None, :] * state
            # else DECAY == 0: no decay
            if READ_BEFORE:
                output = SCALE * tl.sum(state * q_t[:, None], axis=0)
                tl.store(
                    OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                    output,
                    v_mask,
                )
            if IS_DELTA:
                beta_t = tl.load(BETA + gb_token_base + token * H).to(tl.float32)
                retrieved = tl.sum(state * k_t[:, None], axis=0)
                delta = beta_t * (v_t - retrieved)
            else:
                delta = v_t
            state = state + k_t[:, None] * delta[None, :]
            if not READ_BEFORE:
                output = SCALE * tl.sum(state * q_t[:, None], axis=0)
                tl.store(
                    OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                    output,
                    v_mask,
                )
            tl.store(
                STATES + (token * tl.num_programs(0) + row) * (K_DIM * V_DIM)
                + k_index[:, None] * V_DIM
                + v_index[None, :],
                state,
                kv_mask,
            )
        tl.store(FINAL + state_offset, state, kv_mask)

    @triton.jit
    def backward_kernel(
        Q,
        K,
        V,
        G,
        BETA,
        INITIAL,
        STATES,
        GRAD_OUTPUT,
        GRAD_FINAL,
        GRAD_Q,
        GRAD_K,
        GRAD_V,
        GRAD_G,
        GRAD_BETA,
        GRAD_INITIAL,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        DECAY: tl.constexpr,
        IS_DELTA: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_base = ((batch * H + head) * K_DIM) * V_DIM
        qk_token_base = batch * (T * H * K_DIM) + head * K_DIM
        v_token_base = batch * (T * H * V_DIM) + head * V_DIM
        gb_token_base = batch * (T * H) + head
        if HAS_GRAD_FINAL:
            dstate = tl.load(
                GRAD_FINAL + state_base + k_index[:, None] * V_DIM + v_index[None, :],
                kv_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            dstate = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        for reverse_index in range(T):
            token = T - reverse_index - 1
            k_t = tl.load(
                K + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(
                V + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)
            q_t = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            grad_out = tl.load(
                GRAD_OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                v_mask,
                other=0.0,
            ).to(tl.float32)
            # Post-update state S_t saved by the forward pass.
            state_t = tl.load(
                STATES + (token * tl.num_programs(0) + row) * (K_DIM * V_DIM)
                + k_index[:, None] * V_DIM
                + v_index[None, :],
                kv_mask,
                other=0.0,
            ).to(tl.float32)
            # Pre-update state S_prev: the previous post-update state, or the
            # initial state at token 0.
            if token > 0:
                state_prev = tl.load(
                    STATES + ((token - 1) * tl.num_programs(0) + row) * (K_DIM * V_DIM)
                    + k_index[:, None] * V_DIM
                    + v_index[None, :],
                    kv_mask,
                    other=0.0,
                ).to(tl.float32)
            elif HAS_INITIAL:
                state_prev = tl.load(
                    INITIAL + state_base + k_index[:, None] * V_DIM + v_index[None, :],
                    kv_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                state_prev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
            # Decay gate broadcast by granularity, and S_dec = decay * S_prev.
            if DECAY == 1:
                g_t = tl.load(G + gb_token_base + token * H).to(tl.float32)
                decay = tl.zeros((BLOCK_K, BLOCK_V), tl.float32) + tl.exp(g_t)
            elif DECAY == 2:
                g_k = tl.load(
                    G + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
                ).to(tl.float32)
                decay = tl.zeros((BLOCK_K, BLOCK_V), tl.float32) + tl.exp(g_k)[:, None]
            elif DECAY == 3:
                g_v = tl.load(
                    G + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
                ).to(tl.float32)
                decay = tl.zeros((BLOCK_K, BLOCK_V), tl.float32) + tl.exp(g_v)[None, :]
            else:
                decay = tl.zeros((BLOCK_K, BLOCK_V), tl.float32) + 1.0
            state_dec = decay * state_prev
            # Output gradient depends on read timing. dstate carries dL/dS_t from
            # later tokens; dstate_t adds this token's output contribution.
            if READ_BEFORE:
                # y_t read from S_dec (pre-update), so the output feeds S_dec.
                grad_q = SCALE * tl.sum(state_dec * grad_out[None, :], axis=1)
                dstate_t = dstate
                dsdec_out = SCALE * q_t[:, None] * grad_out[None, :]
            else:
                # y_t read from S_t (post-update), so the output feeds S_t.
                grad_q = SCALE * tl.sum(state_t * grad_out[None, :], axis=1)
                dstate_t = dstate + SCALE * q_t[:, None] * grad_out[None, :]
                dsdec_out = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
            # Reverse the rank-1 update S_t = S_dec + k_t delta^T to get dS_dec.
            if IS_DELTA:
                beta_t = tl.load(BETA + gb_token_base + token * H).to(tl.float32)
                retrieved = tl.sum(state_dec * k_t[:, None], axis=0)
                delta = beta_t * (v_t - retrieved)
                ddelta = tl.sum(dstate_t * k_t[:, None], axis=0)
                grad_k = tl.sum(dstate_t * delta[None, :], axis=1)
                grad_v = beta_t * ddelta
                grad_beta = tl.sum(ddelta * (v_t - retrieved), axis=0)
                dretrieved = -beta_t * ddelta
                grad_k = grad_k + tl.sum(state_dec * dretrieved[None, :], axis=1)
                dsdec = dstate_t + k_t[:, None] * dretrieved[None, :]
            else:
                grad_v = tl.sum(dstate_t * k_t[:, None], axis=0)
                grad_k = tl.sum(dstate_t * v_t[None, :], axis=1)
                grad_beta = 0.0
                dsdec = dstate_t
            # For READ_BEFORE the output also flows into S_dec directly.
            dsdec = dsdec + dsdec_out
            # Reverse the decay S_dec = decay * S_prev.
            if DECAY == 1:
                grad_g = tl.sum(tl.sum(dsdec * state_dec, axis=1), axis=0)
                tl.store(GRAD_G + gb_token_base + token * H, grad_g)
            elif DECAY == 2:
                grad_g_k = tl.sum(dsdec * state_dec, axis=1)
                tl.store(
                    GRAD_G + qk_token_base + token * (H * K_DIM) + k_index,
                    grad_g_k,
                    k_mask,
                )
            elif DECAY == 3:
                grad_g_v = tl.sum(dsdec * state_dec, axis=0)
                tl.store(
                    GRAD_G + v_token_base + token * (H * V_DIM) + v_index,
                    grad_g_v,
                    v_mask,
                )
            dstate = decay * dsdec
            tl.store(
                GRAD_Q + qk_token_base + token * (H * K_DIM) + k_index, grad_q, k_mask
            )
            tl.store(
                GRAD_K + qk_token_base + token * (H * K_DIM) + k_index, grad_k, k_mask
            )
            tl.store(
                GRAD_V + v_token_base + token * (H * V_DIM) + v_index, grad_v, v_mask
            )
            if IS_DELTA:
                tl.store(GRAD_BETA + gb_token_base + token * H, grad_beta)
        if HAS_INITIAL:
            tl.store(
                GRAD_INITIAL + state_base + k_index[:, None] * V_DIM + v_index[None, :],
                dstate,
                kv_mask,
            )

    return triton, forward_kernel, backward_kernel


def execute_matrix_state_recurrence(
    *,
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any | None,
    beta: Any | None,
    initial_state: Any | None,
    scale: float | None,
    decay_granularity: str,
    is_delta: bool,
    read_before: bool,
) -> tuple[Any, Any]:
    """Run the fused matrix-state recurrence selected by semantic fields.

    ``query``/``key`` use ``[B, T, H, K]``, ``value`` uses ``[B, T, H, V]``.
    ``log_decay`` shape depends on ``decay_granularity``: ``[B,T,H]`` for
    ``head``, ``[B,T,H,K]`` for ``key_channel``, ``[B,T,H,V]`` for
    ``value_channel``; it is unused for ``none``. ``beta`` (``[B,T,H]``) is
    required for the delta rule and ignored for the additive rule. Returns
    ``(output, final_state)`` with output ``[B,T,H,V]`` and final state
    ``[B,H,K,V]`` (fp32).
    """
    import torch

    triton, forward_kernel, backward_kernel = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native matrix-state recurrence requires CUDA tensors")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if key.shape != query.shape:
        raise ValueError("key must match query shape [B,T,H,K]")
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("value must use [B,T,H,V] matching query heads")
    decay_code = {
        "none": _DECAY_NONE,
        "head": _DECAY_HEAD,
        "key_channel": _DECAY_KEY_CHANNEL,
        "value_channel": _DECAY_VALUE_CHANNEL,
    }[decay_granularity]
    if is_delta and beta is None:
        raise ValueError("the delta update rule requires beta")
    if decay_code != _DECAY_NONE and log_decay is None:
        raise ValueError("this recurrence requires log_decay")
    resolved_scale = float(scale) if scale is not None else 1.0
    if initial_state is None:
        initial_tensor = torch.zeros(
            (batch, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
        )
        has_initial = False
    else:
        if tuple(initial_state.shape) != (batch, heads, key_dim, value_dim):
            raise ValueError("initial_state must use [B,H,K,V]")
        initial_tensor = initial_state.contiguous()
        has_initial = True
    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    log_decay_c = (
        log_decay.contiguous()
        if log_decay is not None
        else torch.zeros((batch, sequence, heads), device=query.device, dtype=torch.float32)
    )
    beta_c = (
        beta.contiguous()
        if beta is not None
        else torch.zeros((batch, sequence, heads), device=query.device, dtype=torch.float32)
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    grid = (batch * heads,)
    warps = 4

    class _MatrixState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, g, b, initial):
            output = torch.empty(
                (batch, sequence, heads, value_dim), device=q.device, dtype=q.dtype
            )
            states = torch.empty(
                (sequence, batch * heads, key_dim, value_dim),
                device=q.device,
                dtype=torch.float32,
            )
            final = torch.empty(
                (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
            )
            forward_kernel[grid](
                q, k, v, g, b, initial, output, states, final,
                heads, sequence, key_dim, value_dim, resolved_scale,
                decay_code, is_delta, read_before, has_initial,
                block_k, block_v, num_warps=warps,
            )
            ctx.save_for_backward(q, k, v, g, b, initial, states)
            ctx.has_initial = has_initial
            return output, final

        @staticmethod
        def backward(ctx, grad_output, grad_final):
            q, k, v, g, b, initial, states = ctx.saved_tensors
            grad_output = (
                torch.zeros_like(v) if grad_output is None else grad_output.contiguous()
            )
            grad_q = torch.empty_like(q)
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
            grad_g = torch.zeros_like(g)
            grad_b = torch.zeros_like(b)
            grad_initial = torch.empty(
                (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
            )
            grad_final_tensor = (
                torch.zeros(
                    (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
                )
                if grad_final is None
                else grad_final.contiguous().float()
            )
            backward_kernel[grid](
                q, k, v, g, b, initial, states, grad_output, grad_final_tensor,
                grad_q, grad_k, grad_v, grad_g, grad_b, grad_initial,
                heads, sequence, key_dim, value_dim, resolved_scale,
                decay_code, is_delta, read_before, ctx.has_initial,
                grad_final is not None, block_k, block_v, num_warps=warps,
            )
            return (
                grad_q,
                grad_k,
                grad_v,
                grad_g if log_decay is not None else None,
                grad_b if beta is not None else None,
                grad_initial if ctx.has_initial else None,
            )

    return _MatrixState.apply(query_c, key_c, value_c, log_decay_c, beta_c, initial_tensor)


__all__ = ["execute_matrix_state_recurrence"]
