"""Fused Triton scan for the matrix-state K2 recurrence family.

This is the native generator for the K2 matrix-state class - the single largest
recurrence group in the catalog. One reusable kernel covers the whole class; the
compiler selects the decay granularity, update rule, and read timing from the
semantic spec, never from an architecture name. The equation per token mirrors
the NumPy canonical core :func:`urm.oracles.matrix_state.recurrent`::

    Z_t    = decay(G_t) * M_{t-1}        (or Z_t = left_t @ M_{t-1}, factored)
    h_t    = retr_t^T Z_t                (retrieval; delta rule only)
    delta_t = beta_t * (v_t - h_t)       (delta rule)
            | write_gate*v_t - (erase_gate*k_t)^T Z_t   (dual-gate)
            | v_t                                       (additive)
    M_t    = Z_t + k_t delta_t^T
    y_t    = scale * q_t^T M_t           (read before or after the update)
            [/ max(q_t^T z_t, epsilon)   (query/key normalizer)]

The canonical-core options covered here, all in fp32 accumulation:

- decay granularity ``none`` / ``head`` / ``key_channel`` / ``value_channel``
  (how ``exp(log_decay)`` broadcasts over the ``[K, V]`` state), or a factored
  left transition ``Z = left_t @ M`` (generalized-delta IPLR/DPLR).
- delta / additive / dual-gate (``erase_gate``/``write_gate``) update rules, and
  a separate retrieval key (``retrieval_keys``, the comba dual-key delta).
- multi-rank updates: ``R`` sequential rank-1 delta updates within one token
  (``update_keys``/``update_values``/``rank_beta``, gated_delta_product).
- a query/key denominator normalizer (linear-attention form) tracked as a second
  state and read as ``y = scale*(q^T M)/max(q^T z, epsilon)``.
- feature maps (identity / l2_normalize / relu / elu_plus_one) applied to the
  query/key on load. The polynomial quadratic feature bases expand the feature
  dimension; the caller pre-expands the operands (matching the canonical core),
  so the kernel sees the expanded width and needs no polynomial mode.

The state ``M`` is a per-head ``[K, V]`` matrix held in fp32. One program owns
one ``(batch, head)`` pair and scans the sequence, so the recurrence is exact
(no chunked approximation). Forward stores the per-token states (and, for the
normalizer / multi-rank variants, the per-token denominator states and per-rank
pre-update states) so the backward pass runs an exact reverse scan. The reverse
(adjoint) scan covers the full canonical-core envelope: the plain delta/additive
path plus the dual-gate, retrieval-key, normalizer, multi-rank, left-transition,
and non-identity-feature-map configurations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Decay granularity encodings passed to the kernel as constexpr.
_DECAY_NONE = 0
_DECAY_HEAD = 1
_DECAY_KEY_CHANNEL = 2
_DECAY_VALUE_CHANNEL = 3

# Feature map encodings passed to the kernel as constexpr.
_FEATURE_IDENTITY = 0
_FEATURE_L2_NORMALIZE = 1
_FEATURE_RELU = 2
_FEATURE_ELU_PLUS_ONE = 3

_FEATURE_MAPS = {
    "identity": _FEATURE_IDENTITY,
    "l2_normalize": _FEATURE_L2_NORMALIZE,
    "relu": _FEATURE_RELU,
    "elu_plus_one": _FEATURE_ELU_PLUS_ONE,
}


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _apply_feature_map(x, FEATURE_MAP: tl.constexpr):
        """Feature map applied to a [BLOCK_K] vector on load (fp32)."""
        if FEATURE_MAP == 1:  # l2_normalize: x * rsqrt(sum(x^2) + 1e-6)
            return x * tl.rsqrt(tl.sum(x * x, axis=0) + 1e-6)
        elif FEATURE_MAP == 2:  # relu
            return tl.maximum(x, 0.0)
        elif FEATURE_MAP == 3:  # elu_plus_one: where(x > 0, x, expm1(x)) + 1
            return tl.where(x > 0, x, libdevice.expm1(x)) + 1.0
        else:  # identity
            return x

    @triton.jit
    def _apply_feature_map_backward(x, grad, FEATURE_MAP: tl.constexpr):
        """Jacobian-transpose of the load feature map: dL/dx_raw from dL/dx_mapped."""
        if FEATURE_MAP == 1:  # l2_normalize: f = s x, s = rsqrt(sum(x^2) + 1e-6)
            s = tl.rsqrt(tl.sum(x * x, axis=0) + 1e-6)
            return s * grad - (s * s * s) * x * tl.sum(x * grad, axis=0)
        elif FEATURE_MAP == 2:  # relu
            return tl.where(x > 0, grad, 0.0)
        elif FEATURE_MAP == 3:  # elu_plus_one: f' = 1 if x > 0 else exp(x)
            return grad * tl.where(x > 0, 1.0, tl.exp(x))
        else:  # identity
            return grad

    @triton.jit
    def forward_kernel(
        Q,
        K,
        V,
        G,  # log_decay: [B,T,H] (head), [B,T,H,K] (key_channel), [B,T,H,V] (value_channel)
        BETA,  # [B,T,H] or None
        RETR,  # retrieval_keys [B,T,H,K] or None
        ERASE,  # erase_gate [B,T,H,K] or None
        WRITE,  # write_gate [B,T,H,V] or None
        LEFT,  # left_transitions [B,T,H,K,K] or None
        UK,  # update_keys [B,T,R,H,K] or None
        UV,  # update_values [B,T,R,H,V] or None
        RB,  # rank_beta [B,T,R,H] or None
        INITIAL,
        OUTPUT,
        STATES,
        NORM_STATES,  # [T, B*H, K] post-update denominator states (normalizer only)
        RANK_STATES,  # [T, B*H, RANK, K, V] per-rank pre-update states (multi-rank)
        FINAL,
        FINAL_NORM,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        RANK: tl.constexpr,
        SCALE: tl.constexpr,
        EPSILON: tl.constexpr,
        DECAY: tl.constexpr,
        IS_DELTA: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        FEATURE_MAP: tl.constexpr,
        DUAL_GATE: tl.constexpr,
        HAS_RETR: tl.constexpr,
        NORMALIZER: tl.constexpr,
        HAS_LEFT: tl.constexpr,
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
        # Query/key denominator normalizer state [K], tracked alongside M.
        norm = tl.zeros((BLOCK_K,), dtype=tl.float32)
        qk_token_base = batch * (T * H * K_DIM) + head * K_DIM
        v_token_base = batch * (T * H * V_DIM) + head * V_DIM
        gb_token_base = batch * (T * H) + head
        left_token_base = batch * (T * H * K_DIM * K_DIM) + head * (K_DIM * K_DIM)
        uk_token_base = batch * (T * RANK * H * K_DIM) + head * K_DIM
        uv_token_base = batch * (T * RANK * H * V_DIM) + head * V_DIM
        rb_token_base = batch * (T * RANK * H) + head
        for token in range(T):
            k_t = tl.load(
                K + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            k_t = _apply_feature_map(k_t, FEATURE_MAP)
            v_t = tl.load(
                V + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)
            q_t = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            q_t = _apply_feature_map(q_t, FEATURE_MAP)
            # Transition: factored left matrix Z = left_t @ M, or diagonal decay.
            if HAS_LEFT:
                left_t = tl.load(
                    LEFT
                    + left_token_base
                    + token * (H * K_DIM * K_DIM)
                    + k_index[:, None] * K_DIM
                    + k_index[None, :],
                    k_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                state = tl.dot(left_t, state, input_precision="ieee")
                # The canonical core does not combine a left transition with the
                # denominator normalizer; no normalizer decay here.
            elif DECAY == 1:  # head: scalar per head
                g_t = tl.load(G + gb_token_base + token * H).to(tl.float32)
                state = tl.exp(g_t) * state
                if NORMALIZER:
                    norm = tl.exp(g_t) * norm
            elif DECAY == 2:  # key_channel: per-K, broadcast over V
                g_t = tl.load(
                    G + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[:, None] * state
                if NORMALIZER:
                    norm = tl.exp(g_t) * norm
            elif DECAY == 3:  # value_channel: per-V, broadcast over K
                g_t = tl.load(
                    G + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[None, :] * state
                # value_channel decay does not apply to the [K] denominator.
            # else DECAY == 0: no decay
            if READ_BEFORE:
                output = SCALE * tl.sum(state * q_t[:, None], axis=0)
                if NORMALIZER:
                    denom = tl.sum(norm * q_t, axis=0)
                    output = output / tl.maximum(denom, EPSILON)
                tl.store(
                    OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                    output,
                    v_mask,
                )
            if RANK > 0:
                # Multi-rank delta: R sequential rank-1 delta updates within the token.
                for r in tl.static_range(RANK):
                    # Save the pre-update state for the reverse (adjoint) scan.
                    tl.store(
                        RANK_STATES
                        + ((token * tl.num_programs(0) + row) * RANK + r)
                        * (K_DIM * V_DIM)
                        + k_index[:, None] * V_DIM
                        + v_index[None, :],
                        state,
                        kv_mask,
                    )
                    uk_r = tl.load(
                        UK + uk_token_base + token * (RANK * H * K_DIM)
                        + r * (H * K_DIM) + k_index,
                        k_mask,
                        other=0.0,
                    ).to(tl.float32)
                    uv_r = tl.load(
                        UV + uv_token_base + token * (RANK * H * V_DIM)
                        + r * (H * V_DIM) + v_index,
                        v_mask,
                        other=0.0,
                    ).to(tl.float32)
                    rb_r = tl.load(
                        RB + rb_token_base + token * (RANK * H) + r * H
                    ).to(tl.float32)
                    retr_r = tl.sum(state * uk_r[:, None], axis=0)
                    delta_r = rb_r * (uv_r - retr_r)
                    state = state + uk_r[:, None] * delta_r[None, :]
            elif DUAL_GATE:
                # Dual-gate delta: retrieval uses erase*k, the write value uses
                # write*v, and the outer product uses the (feature-mapped) key k.
                erase_t = tl.load(
                    ERASE + qk_token_base + token * (H * K_DIM) + k_index,
                    k_mask,
                    other=0.0,
                ).to(tl.float32)
                write_t = tl.load(
                    WRITE + v_token_base + token * (H * V_DIM) + v_index,
                    v_mask,
                    other=0.0,
                ).to(tl.float32)
                retrieved = tl.sum(state * (erase_t * k_t)[:, None], axis=0)
                delta = write_t * v_t - retrieved
                state = state + k_t[:, None] * delta[None, :]
            elif IS_DELTA:
                beta_t = tl.load(BETA + gb_token_base + token * H).to(tl.float32)
                if HAS_RETR:
                    # Separate retrieval key (comba dual-key); not feature-mapped.
                    retr_t = tl.load(
                        RETR + qk_token_base + token * (H * K_DIM) + k_index,
                        k_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    retr_t = k_t
                retrieved = tl.sum(state * retr_t[:, None], axis=0)
                delta = beta_t * (v_t - retrieved)
                state = state + k_t[:, None] * delta[None, :]
            else:
                delta = v_t
                state = state + k_t[:, None] * delta[None, :]
            if NORMALIZER:
                # The denominator accumulates the (feature-mapped) write key.
                norm = norm + k_t
            if not READ_BEFORE:
                output = SCALE * tl.sum(state * q_t[:, None], axis=0)
                if NORMALIZER:
                    denom = tl.sum(norm * q_t, axis=0)
                    output = output / tl.maximum(denom, EPSILON)
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
            if NORMALIZER:
                # Save the post-update denominator state for the reverse scan.
                tl.store(
                    NORM_STATES + (token * tl.num_programs(0) + row) * K_DIM + k_index,
                    norm,
                    k_mask,
                )
        tl.store(FINAL + state_offset, state, kv_mask)
        if NORMALIZER:
            tl.store(
                FINAL_NORM + (batch * H + head) * K_DIM + k_index, norm, k_mask
            )

    @triton.jit
    def backward_kernel(
        Q,
        K,
        V,
        G,
        BETA,
        RETR,  # retrieval_keys [B,T,H,K] (comba dual-key) or None
        ERASE,  # erase_gate [B,T,H,K] (dual-gate) or None
        WRITE,  # write_gate [B,T,H,V] (dual-gate) or None
        LEFT,  # left_transitions [B,T,H,K,K] (factored) or None
        UK,  # update_keys [B,T,R,H,K] (multi-rank) or None
        UV,  # update_values [B,T,R,H,V] (multi-rank) or None
        RB,  # rank_beta [B,T,R,H] (multi-rank) or None
        INITIAL,
        STATES,
        NORM_STATES,  # [T, B*H, K] post-update denominator states (normalizer)
        RANK_STATES,  # [T, B*H, RANK, K, V] per-rank pre-update states (multi-rank)
        GRAD_OUTPUT,
        GRAD_FINAL,
        GRAD_FINAL_NORM,
        GRAD_Q,
        GRAD_K,
        GRAD_V,
        GRAD_G,
        GRAD_BETA,
        GRAD_RETR,
        GRAD_ERASE,
        GRAD_WRITE,
        GRAD_LEFT,
        GRAD_UK,
        GRAD_UV,
        GRAD_RB,
        GRAD_INITIAL,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        RANK: tl.constexpr,
        SCALE: tl.constexpr,
        EPSILON: tl.constexpr,
        DECAY: tl.constexpr,
        IS_DELTA: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        FEATURE_MAP: tl.constexpr,
        DUAL_GATE: tl.constexpr,
        HAS_RETR: tl.constexpr,
        NORMALIZER: tl.constexpr,
        HAS_LEFT: tl.constexpr,
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
        left_token_base = batch * (T * H * K_DIM * K_DIM) + head * (K_DIM * K_DIM)
        uk_token_base = batch * (T * RANK * H * K_DIM) + head * K_DIM
        uv_token_base = batch * (T * RANK * H * V_DIM) + head * V_DIM
        rb_token_base = batch * (T * RANK * H) + head
        norm_row_base = row * K_DIM
        if HAS_GRAD_FINAL:
            dstate = tl.load(
                GRAD_FINAL + state_base + k_index[:, None] * V_DIM + v_index[None, :],
                kv_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            dstate = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        if NORMALIZER and HAS_GRAD_FINAL:
            dnorm = tl.load(
                GRAD_FINAL_NORM + norm_row_base + k_index, k_mask, other=0.0
            ).to(tl.float32)
        else:
            dnorm = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for reverse_index in range(T):
            token = T - reverse_index - 1
            k_raw = tl.load(
                K + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(
                V + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)
            q_raw = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            # The forward applies the feature map on load; reconstruct the mapped
            # operands and map the resulting gradients back through the Jacobian.
            k_t = _apply_feature_map(k_raw, FEATURE_MAP)
            q_t = _apply_feature_map(q_raw, FEATURE_MAP)
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
            # The factored left transition replaces pointwise decay: Z = left @ M.
            if HAS_LEFT:
                left_t = tl.load(
                    LEFT
                    + left_token_base
                    + token * (H * K_DIM * K_DIM)
                    + k_index[:, None] * K_DIM
                    + k_index[None, :],
                    k_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                state_dec = tl.dot(left_t, state_prev, input_precision="ieee")
            # Denominator normalizer states saved by the forward pass. norm_t is
            # post-update; the pre-update (decayed) norm is norm_dec.
            if NORMALIZER:
                norm_t = tl.load(
                    NORM_STATES + token * tl.num_programs(0) * K_DIM
                    + norm_row_base + k_index,
                    k_mask,
                    other=0.0,
                ).to(tl.float32)
                if token > 0:
                    norm_prev = tl.load(
                        NORM_STATES + (token - 1) * tl.num_programs(0) * K_DIM
                        + norm_row_base + k_index,
                        k_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    norm_prev = tl.zeros((BLOCK_K,), dtype=tl.float32)
                if DECAY == 1:
                    norm_dec = tl.exp(g_t) * norm_prev
                elif DECAY == 2:
                    norm_dec = tl.exp(g_k) * norm_prev
                else:
                    # value_channel decay does not apply to the [K] denominator.
                    norm_dec = norm_prev
            # Output gradient depends on read timing. dstate carries dL/dS_t from
            # later tokens; dstate_t adds this token's output contribution. For the
            # normalizer the read is y = scale*(q^T S)/max(q^T n, eps).
            if READ_BEFORE:
                # y_t read from S_dec (pre-update), so the output feeds S_dec.
                if NORMALIZER:
                    denom = tl.sum(norm_dec * q_t, axis=0)
                    denom_c = tl.maximum(denom, EPSILON)
                    dnum = grad_out / denom_c
                    num = SCALE * tl.sum(state_dec * q_t[:, None], axis=0)
                    active = denom > EPSILON
                    dden = tl.where(
                        active, -tl.sum(grad_out * num, axis=0) / (denom_c * denom_c),
                        0.0,
                    )
                    grad_q = SCALE * tl.sum(state_dec * dnum[None, :], axis=1)
                    grad_q = grad_q + dden * norm_dec
                    dnorm_read = dden * q_t
                    dstate_t = dstate
                    dsdec_out = SCALE * q_t[:, None] * dnum[None, :]
                else:
                    grad_q = SCALE * tl.sum(state_dec * grad_out[None, :], axis=1)
                    dstate_t = dstate
                    dsdec_out = SCALE * q_t[:, None] * grad_out[None, :]
                    dnorm_read = tl.zeros((BLOCK_K,), dtype=tl.float32)
            else:
                # y_t read from S_t (post-update), so the output feeds S_t.
                if NORMALIZER:
                    denom = tl.sum(norm_t * q_t, axis=0)
                    denom_c = tl.maximum(denom, EPSILON)
                    dnum = grad_out / denom_c
                    num = SCALE * tl.sum(state_t * q_t[:, None], axis=0)
                    active = denom > EPSILON
                    dden = tl.where(
                        active, -tl.sum(grad_out * num, axis=0) / (denom_c * denom_c),
                        0.0,
                    )
                    grad_q = SCALE * tl.sum(state_t * dnum[None, :], axis=1)
                    grad_q = grad_q + dden * norm_t
                    dnorm_read = dden * q_t
                    dstate_t = dstate + SCALE * q_t[:, None] * dnum[None, :]
                else:
                    grad_q = SCALE * tl.sum(state_t * grad_out[None, :], axis=1)
                    dstate_t = dstate + SCALE * q_t[:, None] * grad_out[None, :]
                    dnorm_read = tl.zeros((BLOCK_K,), dtype=tl.float32)
                dsdec_out = tl.zeros((BLOCK_K, BLOCK_V), tl.float32)
            # Reverse the update to get dS_dec and the update-operand gradients.
            grad_k = tl.zeros((BLOCK_K,), dtype=tl.float32)
            grad_v = tl.zeros((BLOCK_V,), dtype=tl.float32)
            grad_beta = 0.0
            if RANK > 0:
                # Reverse the R sequential rank-1 delta updates (reverse order).
                dz = dstate_t
                for r in tl.static_range(RANK - 1, -1, -1):
                    # Pre-update state for rank r, saved by the forward pass.
                    z_r = tl.load(
                        RANK_STATES
                        + ((token * tl.num_programs(0) + row) * RANK + r)
                        * (K_DIM * V_DIM)
                        + k_index[:, None] * V_DIM
                        + v_index[None, :],
                        kv_mask,
                        other=0.0,
                    ).to(tl.float32)
                    uk_r = tl.load(
                        UK + uk_token_base + token * (RANK * H * K_DIM)
                        + r * (H * K_DIM) + k_index,
                        k_mask,
                        other=0.0,
                    ).to(tl.float32)
                    uv_r = tl.load(
                        UV + uv_token_base + token * (RANK * H * V_DIM)
                        + r * (H * V_DIM) + v_index,
                        v_mask,
                        other=0.0,
                    ).to(tl.float32)
                    rb_r = tl.load(
                        RB + rb_token_base + token * (RANK * H) + r * H
                    ).to(tl.float32)
                    retr_r = tl.sum(z_r * uk_r[:, None], axis=0)
                    delta_r = rb_r * (uv_r - retr_r)
                    ddelta_r = tl.sum(dz * uk_r[:, None], axis=0)
                    grad_uk_r = tl.sum(dz * delta_r[None, :], axis=1)
                    grad_uv_r = rb_r * ddelta_r
                    grad_rb_r = tl.sum(ddelta_r * (uv_r - retr_r), axis=0)
                    dretr_r = -rb_r * ddelta_r
                    grad_uk_r = grad_uk_r + tl.sum(z_r * dretr_r[None, :], axis=1)
                    dz = dz + uk_r[:, None] * dretr_r[None, :]
                    tl.store(
                        GRAD_UK + uk_token_base + token * (RANK * H * K_DIM)
                        + r * (H * K_DIM) + k_index,
                        grad_uk_r,
                        k_mask,
                    )
                    tl.store(
                        GRAD_UV + uv_token_base + token * (RANK * H * V_DIM)
                        + r * (H * V_DIM) + v_index,
                        grad_uv_r,
                        v_mask,
                    )
                    tl.store(
                        GRAD_RB + rb_token_base + token * (RANK * H) + r * H,
                        grad_rb_r,
                    )
                dsdec = dz
            elif DUAL_GATE:
                # Dual-gate delta: retrieval uses erase*k, the write value uses
                # write*v, and the outer product uses the (feature-mapped) key k.
                erase_t = tl.load(
                    ERASE + qk_token_base + token * (H * K_DIM) + k_index,
                    k_mask,
                    other=0.0,
                ).to(tl.float32)
                write_t = tl.load(
                    WRITE + v_token_base + token * (H * V_DIM) + v_index,
                    v_mask,
                    other=0.0,
                ).to(tl.float32)
                retrieved = tl.sum(state_dec * (erase_t * k_t)[:, None], axis=0)
                delta = write_t * v_t - retrieved
                ddelta = tl.sum(dstate_t * k_t[:, None], axis=0)
                grad_k = tl.sum(dstate_t * delta[None, :], axis=1)
                grad_v = ddelta * write_t
                grad_write = ddelta * v_t
                dretrieved = -ddelta
                dek = tl.sum(state_dec * dretrieved[None, :], axis=1)  # Z @ dretrieved
                grad_k = grad_k + erase_t * dek
                grad_erase = k_t * dek
                dsdec = dstate_t + (erase_t * k_t)[:, None] * dretrieved[None, :]
                tl.store(
                    GRAD_ERASE + qk_token_base + token * (H * K_DIM) + k_index,
                    grad_erase,
                    k_mask,
                )
                tl.store(
                    GRAD_WRITE + v_token_base + token * (H * V_DIM) + v_index,
                    grad_write,
                    v_mask,
                )
            elif IS_DELTA:
                beta_t = tl.load(BETA + gb_token_base + token * H).to(tl.float32)
                if HAS_RETR:
                    # Separate retrieval key (comba dual-key); not feature-mapped.
                    retr_t = tl.load(
                        RETR + qk_token_base + token * (H * K_DIM) + k_index,
                        k_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    retr_t = k_t
                retrieved = tl.sum(state_dec * retr_t[:, None], axis=0)
                delta = beta_t * (v_t - retrieved)
                ddelta = tl.sum(dstate_t * k_t[:, None], axis=0)
                grad_k = tl.sum(dstate_t * delta[None, :], axis=1)
                grad_v = beta_t * ddelta
                grad_beta = tl.sum(ddelta * (v_t - retrieved), axis=0)
                dretrieved = -beta_t * ddelta
                dretr = tl.sum(state_dec * dretrieved[None, :], axis=1)  # Z @ dretr
                if HAS_RETR:
                    # The retrieval gradient goes to the separate key, not k.
                    tl.store(
                        GRAD_RETR + qk_token_base + token * (H * K_DIM) + k_index,
                        dretr,
                        k_mask,
                    )
                else:
                    grad_k = grad_k + dretr
                dsdec = dstate_t + retr_t[:, None] * dretrieved[None, :]
            else:
                grad_v = tl.sum(dstate_t * k_t[:, None], axis=0)
                grad_k = tl.sum(dstate_t * v_t[None, :], axis=1)
                dsdec = dstate_t
            # For READ_BEFORE the output also flows into S_dec directly.
            dsdec = dsdec + dsdec_out
            # Reverse the denominator update norm_t = norm_dec + k_t and its decay.
            if NORMALIZER:
                # The update norm_t = norm_dec + k_t contributes dL/dnorm_t to k_t
                # and to norm_dec. The read contributes dnorm_read: for an
                # after-update read it consumes norm_t (so it also flows through
                # k_t); for a before-update read it consumes the pre-update
                # norm_dec directly (bypassing k_t).
                if READ_BEFORE:
                    grad_k = grad_k + dnorm
                else:
                    grad_k = grad_k + dnorm + dnorm_read
                dnorm_dec = dnorm + dnorm_read
                if DECAY == 1:
                    grad_g_norm = tl.sum(dnorm_dec * norm_dec, axis=0)
                    dnorm = tl.exp(g_t) * dnorm_dec
                elif DECAY == 2:
                    grad_g_norm = dnorm_dec * norm_dec
                    dnorm = tl.exp(g_k) * dnorm_dec
                else:
                    grad_g_norm = tl.zeros((BLOCK_K,), dtype=tl.float32)
                    dnorm = dnorm_dec
            # Reverse the transition. For the factored left transition Z = left @ M
            # the adjoint is dM = left^T @ dZ and dL/dleft = dZ @ M^T; the left
            # gradient flows back through the torch-side factorization. For
            # pointwise decay Z = decay * M the adjoint is dM = decay * dZ.
            if HAS_LEFT:
                grad_left = tl.dot(dsdec, tl.trans(state_prev), input_precision="ieee")
                tl.store(
                    GRAD_LEFT
                    + left_token_base
                    + token * (H * K_DIM * K_DIM)
                    + k_index[:, None] * K_DIM
                    + k_index[None, :],
                    grad_left,
                    k_mask[:, None] & k_mask[None, :],
                )
                dstate = tl.dot(tl.trans(left_t), dsdec, input_precision="ieee")
            else:
                if DECAY == 1:
                    grad_g = tl.sum(tl.sum(dsdec * state_dec, axis=1), axis=0)
                    if NORMALIZER:
                        grad_g = grad_g + grad_g_norm
                    tl.store(GRAD_G + gb_token_base + token * H, grad_g)
                elif DECAY == 2:
                    grad_g_k = tl.sum(dsdec * state_dec, axis=1)
                    if NORMALIZER:
                        grad_g_k = grad_g_k + grad_g_norm
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
            # Map the query/key gradients back through the feature-map Jacobian.
            grad_q = _apply_feature_map_backward(q_raw, grad_q, FEATURE_MAP)
            grad_k = _apply_feature_map_backward(k_raw, grad_k, FEATURE_MAP)
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

    @triton.jit
    def decode_step_kernel(
        Q,
        K,
        V,
        G,  # log_decay [B,H] (head decay)
        BETA,  # [B,H]
        STATE,  # persistent [B,H,K,V], updated in place
        OUTPUT,  # [B,H,V]
        H: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        DECAY: tl.constexpr,
        IS_DELTA: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One fused single-token matrix-state decode step, in place on STATE.

        Follows the ATMA decode-kernel pattern: the persistent per-head [K, V]
        state is read and written once, in place; there is no per-step state
        history allocation, no autograd graph, and no host sync, so the step is
        CUDA-graph capturable. One program owns one (batch, head) pair.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_offset = ((batch * H + head) * K_DIM) * V_DIM + k_index[:, None] * V_DIM + v_index[None, :]
        state = tl.load(STATE + state_offset, kv_mask, other=0.0).to(tl.float32)
        qk_base = (batch * H + head) * K_DIM
        v_base = (batch * H + head) * V_DIM
        gb_base = batch * H + head
        k_t = tl.load(K + qk_base + k_index, k_mask, other=0.0).to(tl.float32)
        v_t = tl.load(V + v_base + v_index, v_mask, other=0.0).to(tl.float32)
        q_t = tl.load(Q + qk_base + k_index, k_mask, other=0.0).to(tl.float32)
        # Decay broadcast by granularity (head / key_channel / value_channel / none).
        if DECAY == 1:  # head: scalar per head
            g_t = tl.load(G + gb_base).to(tl.float32)
            state = tl.exp(g_t) * state
        elif DECAY == 2:  # key_channel: per-K, broadcast over V
            g_t = tl.load(G + qk_base + k_index, k_mask, other=0.0).to(tl.float32)
            state = tl.exp(g_t)[:, None] * state
        elif DECAY == 3:  # value_channel: per-V, broadcast over K
            g_t = tl.load(G + v_base + v_index, v_mask, other=0.0).to(tl.float32)
            state = tl.exp(g_t)[None, :] * state
        if READ_BEFORE:
            output = SCALE * tl.sum(state * q_t[:, None], axis=0)
            tl.store(OUTPUT + v_base + v_index, output, v_mask)
        if IS_DELTA:
            beta_t = tl.load(BETA + gb_base).to(tl.float32)
            retrieved = tl.sum(state * k_t[:, None], axis=0)
            delta = beta_t * (v_t - retrieved)
        else:
            delta = v_t
        state = state + k_t[:, None] * delta[None, :]
        if not READ_BEFORE:
            output = SCALE * tl.sum(state * q_t[:, None], axis=0)
            tl.store(OUTPUT + v_base + v_index, output, v_mask)
        tl.store(STATE + state_offset, state, kv_mask)

    return triton, forward_kernel, backward_kernel, decode_step_kernel


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
    retrieval_keys: Any | None = None,
    erase_gate: Any | None = None,
    write_gate: Any | None = None,
    left_transitions: Any | None = None,
    update_keys: Any | None = None,
    update_values: Any | None = None,
    rank_beta: Any | None = None,
    feature_map: str = "identity",
    normalizer: bool = False,
    epsilon: float = 1e-6,
) -> tuple[Any, Any]:
    """Run the fused matrix-state recurrence selected by semantic fields.

    ``query``/``key`` use ``[B, T, H, K]``, ``value`` uses ``[B, T, H, V]``.
    ``log_decay`` shape depends on ``decay_granularity``: ``[B,T,H]`` for
    ``head``, ``[B,T,H,K]`` for ``key_channel``, ``[B,T,H,V]`` for
    ``value_channel``; it is unused for ``none``. ``beta`` (``[B,T,H]``) is
    required for the delta rule and ignored for the additive rule. Returns
    ``(output, final_state)`` with output ``[B,T,H,V]`` and final state
    ``[B,H,K,V]`` (fp32); when ``normalizer`` is set, returns
    ``(output, final_state, final_normalizer)`` with the denominator state
    ``[B,H,K]``.

    Canonical-core options (mirroring ``urm.oracles.matrix_state.recurrent``):

    - ``retrieval_keys`` (``[B,T,H,K]``): a separate retrieval key for the delta
      rule (comba dual-key); the write outer product still uses ``key``.
    - ``erase_gate`` (``[B,T,H,K]``) / ``write_gate`` (``[B,T,H,V]``): the
      dual-gate delta update ``write*v - (erase*k)^T Z`` (gdn2).
    - ``left_transitions`` (``[B,T,H,K,K]``): a factored left transition
      ``Z = left_t @ M`` replacing diagonal decay (generalized-delta IPLR/DPLR).
    - ``update_keys``/``update_values``/``rank_beta`` (``[B,T,R,H,K]`` /
      ``[B,T,R,H,V]`` / ``[B,T,R,H]``): ``R`` sequential rank-1 delta updates
      within one token (gated_delta_product).
    - ``feature_map``: identity / l2_normalize / relu / elu_plus_one applied to
      the query/key on load. Polynomial quadratic bases are pre-expanded by the
      caller (the kernel sees the expanded feature width).
    - ``normalizer``/``epsilon``: track a query/key denominator state and read
      ``y = scale*(q^T M)/max(q^T z, epsilon)`` (linear-attention form).

    The reverse (adjoint) scan covers all of the configurations above.
    """
    import torch

    triton, forward_kernel, backward_kernel, _ = _kernels()
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
    feature_code = _FEATURE_MAPS[feature_map]
    dual_gate = erase_gate is not None or write_gate is not None
    has_retr = retrieval_keys is not None
    has_left = left_transitions is not None
    multi_rank = update_keys is not None
    if multi_rank:
        ranks = update_keys.shape[2]
        if update_values is None or rank_beta is None:
            raise ValueError("multi-rank updates require update_values and rank_beta")
        if update_values.shape[2] != ranks or rank_beta.shape[2] != ranks:
            raise ValueError("update_keys/update_values/rank_beta must share R")
    else:
        ranks = 0
    if dual_gate and (erase_gate is None or write_gate is None):
        raise ValueError("the dual-gate delta requires both erase_gate and write_gate")
    if dual_gate and (is_delta or multi_rank):
        raise ValueError("the dual-gate delta is exclusive of delta/multi-rank")
    if multi_rank and is_delta:
        raise ValueError("multi-rank updates are exclusive of the plain delta rule")
    if has_retr and not is_delta:
        raise ValueError("retrieval_keys require the delta update rule")
    if has_left and decay_code != _DECAY_NONE:
        raise ValueError("left_transitions replace pointwise decay")
    if has_left and normalizer:
        raise ValueError("the canonical core does not combine left transitions and a normalizer")
    if is_delta and not dual_gate and beta is None:
        raise ValueError("the delta update rule requires beta")
    if decay_code != _DECAY_NONE and log_decay is None:
        raise ValueError("this recurrence requires log_decay")
    resolved_scale = float(scale) if scale is not None else 1.0
    resolved_epsilon = float(epsilon)
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

    def _optional(tensor, shape, name):
        if tensor is None:
            # A minimal dummy: the kernel never dereferences an operand whose
            # constexpr flag is off, so a single element suffices (avoids
            # allocating a full [B,T,H,K,K] transition / [B,T,R,H,*] update).
            return torch.zeros(1, device=query.device, dtype=torch.float32)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must use shape {tuple(shape)}")
        return tensor.contiguous()

    retr_c = _optional(retrieval_keys, (batch, sequence, heads, key_dim), "retrieval_keys")
    erase_c = _optional(erase_gate, (batch, sequence, heads, key_dim), "erase_gate")
    write_c = _optional(write_gate, (batch, sequence, heads, value_dim), "write_gate")
    left_c = _optional(
        left_transitions, (batch, sequence, heads, key_dim, key_dim), "left_transitions"
    )
    uk_c = _optional(
        update_keys, (batch, sequence, max(ranks, 1), heads, key_dim), "update_keys"
    )
    uv_c = _optional(
        update_values, (batch, sequence, max(ranks, 1), heads, value_dim), "update_values"
    )
    rb_c = _optional(
        rank_beta, (batch, sequence, max(ranks, 1), heads), "rank_beta"
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    if has_left:
        # tl.dot requires the block dims to be at least 16; the transition and
        # state are zero-padded, so the valid region is computed exactly.
        block_k = max(block_k, 16)
        block_v = max(block_v, 16)
    grid = (batch * heads,)
    warps = 4

    # The reverse (adjoint) scan covers the full canonical-core envelope: the
    # plain delta/additive path plus the dual-gate, retrieval-key, normalizer,
    # multi-rank, left-transition, and non-identity-feature-map configurations.
    class _MatrixState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, g, b, retr, erase, write, left, uk, uv, rb, initial):
            output = torch.empty(
                (batch, sequence, heads, value_dim), device=q.device, dtype=q.dtype
            )
            states = torch.empty(
                (sequence, batch * heads, key_dim, value_dim),
                device=q.device,
                dtype=torch.float32,
            )
            norm_states = (
                torch.empty(
                    (sequence, batch * heads, key_dim),
                    device=q.device,
                    dtype=torch.float32,
                )
                if normalizer
                else states  # unused placeholder when the normalizer is off
            )
            rank_states = (
                torch.empty(
                    (sequence, batch * heads, ranks, key_dim, value_dim),
                    device=q.device,
                    dtype=torch.float32,
                )
                if multi_rank
                else states  # unused placeholder when multi-rank is off
            )
            final = torch.empty(
                (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
            )
            final_norm = (
                torch.empty((batch, heads, key_dim), device=q.device, dtype=torch.float32)
                if normalizer
                else final  # unused placeholder when the normalizer is off
            )
            forward_kernel[grid](
                q, k, v, g, b, retr, erase, write, left, uk, uv, rb,
                initial, output, states, norm_states, rank_states, final, final_norm,
                heads, sequence, key_dim, value_dim, ranks, resolved_scale,
                resolved_epsilon,
                decay_code, is_delta, read_before, has_initial,
                feature_code, dual_gate, has_retr, normalizer, has_left,
                block_k, block_v, num_warps=warps,
            )
            ctx.save_for_backward(
                q, k, v, g, b, retr, erase, write, left, uk, uv, rb,
                initial, states, norm_states, rank_states,
            )
            ctx.has_initial = has_initial
            if normalizer:
                return output, final, final_norm
            return output, final

        @staticmethod
        def backward(ctx, grad_output, grad_final, grad_final_norm=None):
            (q, k, v, g, b, retr, erase, write, left, uk, uv, rb,
             initial, states, norm_states, rank_states) = ctx.saved_tensors
            grad_output = (
                torch.zeros_like(v) if grad_output is None else grad_output.contiguous()
            )
            grad_q = torch.empty_like(q)
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
            grad_g = torch.zeros_like(g)
            grad_b = torch.zeros_like(b)
            grad_retr = (
                torch.zeros_like(retr) if has_retr else retr
            )
            grad_erase = torch.zeros_like(erase) if dual_gate else erase
            grad_write = torch.zeros_like(write) if dual_gate else write
            grad_left = (
                torch.zeros(
                    (batch, sequence, heads, key_dim, key_dim),
                    device=q.device,
                    dtype=torch.float32,
                )
                if has_left
                else left
            )
            grad_uk = torch.zeros_like(uk) if multi_rank else uk
            grad_uv = torch.zeros_like(uv) if multi_rank else uv
            grad_rb = torch.zeros_like(rb) if multi_rank else rb
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
            grad_final_norm_tensor = (
                torch.zeros(
                    (batch, heads, key_dim), device=q.device, dtype=torch.float32
                )
                if grad_final_norm is None
                else grad_final_norm.contiguous().float()
            )
            backward_kernel[grid](
                q, k, v, g, b, retr, erase, write, left, uk, uv, rb,
                initial, states, norm_states, rank_states,
                grad_output, grad_final_tensor, grad_final_norm_tensor,
                grad_q, grad_k, grad_v, grad_g, grad_b,
                grad_retr, grad_erase, grad_write, grad_left,
                grad_uk, grad_uv, grad_rb, grad_initial,
                heads, sequence, key_dim, value_dim, ranks, resolved_scale,
                resolved_epsilon,
                decay_code, is_delta, read_before, ctx.has_initial,
                grad_final is not None,
                feature_code, dual_gate, has_retr, normalizer, has_left,
                block_k, block_v, num_warps=warps,
            )
            return (
                grad_q,
                grad_k,
                grad_v,
                grad_g if log_decay is not None else None,
                grad_b if beta is not None else None,
                grad_retr if has_retr else None,
                grad_erase if dual_gate else None,
                grad_write if dual_gate else None,
                grad_left if has_left else None,
                grad_uk if multi_rank else None,
                grad_uv if multi_rank else None,
                grad_rb if multi_rank else None,
                grad_initial if ctx.has_initial else None,
            )

    return _MatrixState.apply(
        query_c, key_c, value_c, log_decay_c, beta_c,
        retr_c, erase_c, write_c, left_c, uk_c, uv_c, rb_c, initial_tensor,
    )


def execute_matrix_state_decode_step(
    *,
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any | None,
    beta: Any | None,
    state: Any,
    scale: float | None,
    decay_granularity: str,
    is_delta: bool,
    read_before: bool,
) -> Any:
    """Run one fused single-token matrix-state decode step, in place on ``state``.

    This is the decode-path counterpart to :func:`execute_matrix_state_recurrence`:
    the persistent per-head ``[B, H, K, V]`` state is read and written once, in
    place, under ``torch.no_grad()`` with no per-step state-history allocation
    and no autograd graph, so the step is CUDA-graph capturable. ``query``/``key``
    use ``[B, H, K]``, ``value`` uses ``[B, H, V]``, ``state`` is the persistent
    ``[B, H, K, V]`` fp32 tensor updated in place. Returns the output ``[B, H, V]``.
    """
    import torch

    triton, _, _, decode_step_kernel = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native matrix-state decode step requires CUDA tensors")
    batch, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if key.shape != query.shape:
        raise ValueError("key must match query shape [B,H,K]")
    if value.shape[:2] != (batch, heads):
        raise ValueError("value must use [B,H,V] matching query heads")
    if tuple(state.shape) != (batch, heads, key_dim, value_dim):
        raise ValueError("state must use [B,H,K,V]")
    if state.dtype is not torch.float32:
        raise ValueError("the persistent matrix state is fp32")
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
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    log_decay_c = (
        log_decay.contiguous()
        if log_decay is not None
        else torch.zeros((batch, heads), device=query.device, dtype=torch.float32)
    )
    beta_c = (
        beta.contiguous()
        if beta is not None
        else torch.zeros((batch, heads), device=query.device, dtype=torch.float32)
    )
    output = torch.empty((batch, heads, value_dim), device=query.device, dtype=torch.float32)
    with torch.no_grad():
        decode_step_kernel[(batch * heads,)](
            query.contiguous(), key.contiguous(), value.contiguous(),
            log_decay_c, beta_c, state, output,
            heads, key_dim, value_dim, resolved_scale,
            decay_code, is_delta, read_before,
            block_k, block_v, num_warps=4,
        )
    return output


__all__ = ["execute_matrix_state_recurrence", "execute_matrix_state_decode_step"]
