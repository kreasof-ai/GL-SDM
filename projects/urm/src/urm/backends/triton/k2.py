"""Fused Triton scan for the matrix-state K2 recurrence family.

This is the native generator for the K2 matrix-state class - the single largest
recurrence group in the catalog. One reusable kernel covers the whole class; the
compiler selects the decay granularity, update rule, and read timing from the
semantic spec, never from an architecture name. The equation per token mirrors
the NumPy canonical core :func:`urm.backends.numpy.k2.recurrent`::

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

_DECAY_NONE = 0
_DECAY_HEAD = 1
_DECAY_KEY_CHANNEL = 2
_DECAY_VALUE_CHANNEL = 3

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
        if FEATURE_MAP == 1:
            return x * tl.rsqrt(tl.sum(x * x, axis=0) + 1e-6)
        elif FEATURE_MAP == 2:
            return tl.maximum(x, 0.0)
        elif FEATURE_MAP == 3:
            return tl.where(x > 0, x, libdevice.expm1(x)) + 1.0
        else:
            return x

    @triton.jit
    def _apply_feature_map_backward(x, grad, FEATURE_MAP: tl.constexpr):
        """Jacobian-transpose of the load feature map: dL/dx_raw from dL/dx_mapped."""
        if FEATURE_MAP == 1:
            s = tl.rsqrt(tl.sum(x * x, axis=0) + 1e-6)
            return s * grad - (s * s * s) * x * tl.sum(x * grad, axis=0)
        elif FEATURE_MAP == 2:
            return tl.where(x > 0, grad, 0.0)
        elif FEATURE_MAP == 3:
            return grad * tl.where(x > 0, 1.0, tl.exp(x))
        else:
            return grad

    @triton.jit
    def forward_kernel(
        Q,
        K,
        V,
        G,
        BETA,
        RETR,
        ERASE,
        WRITE,
        LEFT,
        UK,
        UV,
        RB,
        INITIAL,
        OUTPUT,
        STATES,
        NORM_STATES,
        RANK_STATES,
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
            elif DECAY == 1:
                g_t = tl.load(G + gb_token_base + token * H).to(tl.float32)
                state = tl.exp(g_t) * state
                if NORMALIZER:
                    norm = tl.exp(g_t) * norm
            elif DECAY == 2:
                g_t = tl.load(
                    G + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[:, None] * state
                if NORMALIZER:
                    norm = tl.exp(g_t) * norm
            elif DECAY == 3:
                g_t = tl.load(
                    G + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
                ).to(tl.float32)
                state = tl.exp(g_t)[None, :] * state
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
                for r in tl.static_range(RANK):
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
        RETR,
        ERASE,
        WRITE,
        LEFT,
        UK,
        UV,
        RB,
        INITIAL,
        STATES,
        NORM_STATES,
        RANK_STATES,
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
            k_t = _apply_feature_map(k_raw, FEATURE_MAP)
            q_t = _apply_feature_map(q_raw, FEATURE_MAP)
            grad_out = tl.load(
                GRAD_OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                v_mask,
                other=0.0,
            ).to(tl.float32)
            state_t = tl.load(
                STATES + (token * tl.num_programs(0) + row) * (K_DIM * V_DIM)
                + k_index[:, None] * V_DIM
                + v_index[None, :],
                kv_mask,
                other=0.0,
            ).to(tl.float32)
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
                    norm_dec = norm_prev
            if READ_BEFORE:
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
            grad_k = tl.zeros((BLOCK_K,), dtype=tl.float32)
            grad_v = tl.zeros((BLOCK_V,), dtype=tl.float32)
            grad_beta = 0.0
            if RANK > 0:
                dz = dstate_t
                for r in tl.static_range(RANK - 1, -1, -1):
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
                dek = tl.sum(state_dec * dretrieved[None, :], axis=1)
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
                dretr = tl.sum(state_dec * dretrieved[None, :], axis=1)
                if HAS_RETR:
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
            dsdec = dsdec + dsdec_out
            if NORMALIZER:
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
        G,
        BETA,
        STATE,
        OUTPUT,
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
        if DECAY == 1:
            g_t = tl.load(G + gb_base).to(tl.float32)
            state = tl.exp(g_t) * state
        elif DECAY == 2:
            g_t = tl.load(G + qk_base + k_index, k_mask, other=0.0).to(tl.float32)
            state = tl.exp(g_t)[:, None] * state
        elif DECAY == 3:
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

    Canonical-core options (mirroring ``urm.backends.numpy.k2.recurrent``):

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
        block_k = max(block_k, 16)
        block_v = max(block_v, 16)
    grid = (batch * heads,)
    warps = 4

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
                else states
            )
            rank_states = (
                torch.empty(
                    (sequence, batch * heads, ranks, key_dim, value_dim),
                    device=q.device,
                    dtype=torch.float32,
                )
                if multi_rank
                else states
            )
            final = torch.empty(
                (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
            )
            final_norm = (
                torch.empty((batch, heads, key_dim), device=q.device, dtype=torch.float32)
                if normalizer
                else final
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


def linear_delta_state(
    initial_state,
    keys,
    queries,
    values,
    beta,
    log_decay,
    *,
    spec,
    scale=None,
):
    """Canonical K2 linear-delta state law — the native Triton implementation of
    the uniform batched signature.

    Same role order, batched shapes (``[B, H, T, *]``), closed
    :class:`LinearDeltaSpec` and ``(output, final_state)`` return as the NumPy
    oracle and Torch reference. The descriptor is decomposed into the native
    kernel's semantic knobs here — the only place that translation happens.
    """
    if spec.normalized:
        # The native kernel's normalizer reads y/max(q·z, ε); the pinned law is
        # y/((scale·q·norm)+ε). The provider declines normalized specs before this
        # point; this guard keeps direct callers honest too.
        raise ValueError(
            "native K2 does not implement the normalized variant (denominator law differs)"
        )
    if spec.gate_scope.value == "elementwise":
        raise ValueError("native K2 does not implement the elementwise gate scope")
    gate = spec.gate_scope.value
    granularity = {
        "none": "none",
        "scalar": "head",
        "head": "head",
        "channel": "key_channel",
    }[gate]
    # Resolve the scale rule exactly as the Torch reference does (urm/backends/torch/
    # k2.py): ``one`` → 1.0, ``key_dim_rsqrt`` → K**-0.5, ``explicit_operand`` → the
    # scale operand is required. The kernel itself only consumes a resolved float;
    # leaving this to the kernel's None→1.0 default would silently run key_dim_rsqrt
    # laws (GLA, DeltaNet, GDN, …) at 8× the pinned read scale — the reference tier
    # owns the rule, the native tier must mirror it, never bypass it.
    if spec.scale_rule.value == "one":
        scale = 1.0
    elif spec.scale_rule.value == "key_dim_rsqrt":
        scale = float(keys.shape[-1]) ** -0.5
    else:  # explicit_operand
        if scale is None:
            raise ValueError("scale_rule=explicit_operand requires a scale value")
        scale = float(scale)
    # The native kernel consumes [B, T, H, *]; the canonical operands arrive
    # [B, H, T, *]. Transpose to the kernel layout, run, transpose back.
    import torch

    q = queries.transpose(1, 2).contiguous()
    k = keys.transpose(1, 2).contiguous()
    v = values.transpose(1, 2).contiguous()
    g = None if granularity == "none" else log_decay.transpose(1, 2).contiguous()
    b = beta.transpose(1, 2).contiguous() if spec.delta else None
    # The native K2 recurrence is ALWAYS invoked as an opaque custom op (the ATMA
    # custom-op pattern): dynamo treats it as a single fused node with no trace into the
    # kernel closure, so a torch.compile'd model compiles without a graph break. The
    # equation is identical — the wrapper only changes how the compiler sees the call.
    from .k2_op import k2_recurrence

    out, final = k2_recurrence(
        q, k, v, g, b, initial_state,
        scale=scale,
        decay_granularity=granularity,
        is_delta=spec.delta,
        read_before=spec.read_timing.value == "before_update",
    )
    return out.transpose(1, 2), final


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class _K2NativeBase:
    """Native K2 anchors run the fused Triton matrix-state recurrence.

    The honest native envelope is the canonical law: the delta/additive update,
    gate scopes ``none``/``scalar``/``head``/``channel``, and before/after read
    timing — verified against the Torch reference and the pinned sources for
    output and final state. The provider DECLINES the descriptor features whose
    native kernel semantics diverge from the pinned law, so the native tier never
    silently executes the wrong equation:

    - ``normalized`` — the kernel reads ``y/max(q·z, ε)`` but the pinned law is
      ``y/((scale·q·norm) + ε)`` (verified divergence); the reference tier owns it.
    - ``elementwise`` gate scope — the per-element ``[K, V]`` Mamba-1 gate has no
      native branch (single-client, reference-tier per the two-client rule).
    - the A8 transition features (``erase_gate``/``write_gate``/``predict_key``/
      ``low_rank``/``num_deltas > 1``) — the kernel's dual-gate/multi-delta forms
      diverge from the pinned reference (verified); they stay on the reference tier
      until a faithful native schedule is qualified.
    """

    family = "k2"
    tier = "native"

    def decline(self, request) -> str | None:
        from ...ir.program import K2GateScope, LinearDeltaSpec

        spec = request.descriptor
        if not isinstance(spec, LinearDeltaSpec):
            return "K2 providers require a closed LinearDeltaSpec"
        if request.accumulation_dtype != "float32":
            return "native K2 requires float32 accumulation"
        if spec.normalized:
            return (
                "native K2 declines the normalized variant: the kernel reads "
                "y/max(q·z, ε) but the pinned law is y/((scale·q·norm)+ε)"
            )
        if spec.gate_scope is K2GateScope.ELEMENTWISE:
            return (
                "native K2 declines the elementwise gate scope: the per-element "
                "[K,V] gate is reference-tier only (single-client, two-client rule)"
            )
        if spec.erase_gate or spec.write_gate or spec.predict_key or spec.low_rank or spec.num_deltas > 1:
            return (
                "native K2 declines the A8 transition features (erase/write/predict/"
                "low_rank/multi-delta): the kernel forms diverge from the pinned law"
            )
        return None

    def execute(self, request, operands):
        scale_op = operands.get("scale")
        out, final_state = linear_delta_state(
            operands["initial_state"], operands["key"], operands["query"],
            operands["value"], operands["beta"], operands.get("log_decay"),
            spec=request.descriptor,
            scale=None if scale_op is None else float(scale_op),
        )
        return {"output": out, "final_state": final_state}


class K2NativeDiagonalProvider(_K2NativeBase):
    name = "urm_native_diagonal_recurrence_v1"


class K2NativeMatrixProvider(_K2NativeBase):
    name = "urm_native_matrix_state_recurrence_v1"


PROVIDERS = (K2NativeDiagonalProvider(), K2NativeMatrixProvider())


# ---------------------------------------------------------------------------
# K2 diagonal single-token decode schedule (moved from historical)
# ---------------------------------------------------------------------------


def _diagonal_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def compose_affine(a_left, b_left, a_right, b_right):
        return a_right * a_left, b_right + a_right * b_left

    @triton.jit
    def forward_kernel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        OUTPUT,
        STATES,
        FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        GATES_ONE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CHUNK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        state_index = tl.arange(0, BLOCK_N)
        state_mask = state_index < N
        if HAS_INITIAL:
            state = tl.load(
                INITIAL + batch * C * N + channel * N + state_index,
                state_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            state = tl.full((BLOCK_N,), 0.0, tl.float32)
        skip_index = 0 if SKIP_SCALAR else channel
        skip = tl.load(SKIP + skip_index).to(tl.float32)
        if READ_BEFORE:
            for token in range(T):
                x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC).to(
                    tl.float32
                )
                if GATES_ONE:
                    input_gate = tl.full((BLOCK_N,), 1.0, tl.float32)
                    read_gate = tl.full((BLOCK_N,), 1.0, tl.float32)
                else:
                    input_gate = tl.load(
                        INPUT_GATE
                        + batch * IG_SB
                        + token * IG_ST
                        + channel * IG_SC
                        + state_index * IG_SN,
                        state_mask,
                        other=0.0,
                    ).to(tl.float32)
                    read_gate = tl.load(
                        READ_GATE
                        + batch * RG_SB
                        + token * RG_ST
                        + channel * RG_SC
                        + state_index * RG_SN,
                        state_mask,
                        other=0.0,
                    ).to(tl.float32)
                log_decay = tl.load(
                    LOG_DECAY
                    + batch * LD_SB
                    + token * LD_ST
                    + channel * LD_SC
                    + state_index * LD_SN,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
                if HAS_STEP_SIZE:
                    step = tl.load(
                        STEP_SIZE
                        + batch * STEP_SB
                        + token * STEP_ST
                        + channel * STEP_SC
                    ).to(tl.float32)
                else:
                    step = 1.0
                output = tl.sum(state * read_gate, 0) + x * skip
                tl.store(OUTPUT + batch * T * C + token * C + channel, output)
                state = tl.exp(log_decay * step) * state + (x * step) * input_gate
                tl.store(
                    STATES
                    + batch * T * C * N
                    + token * C * N
                    + channel * N
                    + state_index,
                    state,
                    state_mask,
                )
            tl.store(
                FINAL + batch * C * N + channel * N + state_index,
                state,
                state_mask,
            )
        else:
            state_offset = state_index[None, :]
            for chunk_start in range(0, T, CHUNK_T):
                token = chunk_start + tl.arange(0, CHUNK_T)
                token_mask = token < T
                x = tl.load(
                    X + batch * X_SB + token * X_ST + channel * X_SC,
                    token_mask,
                    other=0.0,
                ).to(tl.float32)
                if GATES_ONE:
                    input_gate = tl.full((CHUNK_T, BLOCK_N), 1.0, tl.float32)
                    read_gate = tl.full((CHUNK_T, BLOCK_N), 1.0, tl.float32)
                else:
                    input_gate = tl.load(
                        INPUT_GATE
                        + batch * IG_SB
                        + token[:, None] * IG_ST
                        + channel * IG_SC
                        + state_offset * IG_SN,
                        token_mask[:, None] & state_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    read_gate = tl.load(
                        READ_GATE
                        + batch * RG_SB
                        + token[:, None] * RG_ST
                        + channel * RG_SC
                        + state_offset * RG_SN,
                        token_mask[:, None] & state_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                log_decay = tl.load(
                    LOG_DECAY
                    + batch * LD_SB
                    + token[:, None] * LD_ST
                    + channel * LD_SC
                    + state_offset * LD_SN,
                    token_mask[:, None] & state_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if HAS_STEP_SIZE:
                    step = tl.load(
                        STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC,
                        token_mask,
                        other=1.0,
                    ).to(tl.float32)
                else:
                    step = tl.full((CHUNK_T,), 1.0, tl.float32)
                decay = tl.exp(log_decay * step[:, None])
                decay = tl.where(token_mask[:, None], decay, 1.0)
                update = (x * step)[:, None] * input_gate
                update = tl.where(token_mask[:, None], update, 0.0)
                prefix_decay, prefix_update = tl.associative_scan(
                    (decay, update), axis=0, combine_fn=compose_affine
                )
                state_sequence = prefix_decay * state[None, :] + prefix_update
                output = tl.sum(state_sequence * read_gate, axis=1) + x * skip
                output_offset = batch * T * C + token * C + channel
                tl.store(OUTPUT + output_offset, output, token_mask)
                tl.store(
                    STATES
                    + batch * T * C * N
                    + token[:, None] * C * N
                    + channel * N
                    + state_offset,
                    state_sequence,
                    token_mask[:, None] & state_mask[None, :],
                )
                last_valid = tl.sum(tl.where(token_mask, 1, 0), 0) - 1
                state = tl.sum(
                    tl.where((tl.arange(0, CHUNK_T) == last_valid)[:, None], state_sequence, 0.0),
                    axis=0,
                )
            tl.store(
                FINAL + batch * C * N + channel * N + state_index,
                state,
                state_mask,
            )

    @triton.jit
    def backward_kernel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        GRAD_OUTPUT,
        STATES,
        GRAD_X,
        GRAD_INPUT_GATE,
        GRAD_READ_GATE,
        GRAD_LOG_DECAY,
        GRAD_INITIAL,
        GRAD_STEP_SIZE,
        GRAD_FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        GIG_SB: tl.constexpr,
        GIG_ST: tl.constexpr,
        GIG_SC: tl.constexpr,
        GIG_SN: tl.constexpr,
        GRG_SB: tl.constexpr,
        GRG_ST: tl.constexpr,
        GRG_SC: tl.constexpr,
        GRG_SN: tl.constexpr,
        GLD_SB: tl.constexpr,
        GLD_ST: tl.constexpr,
        GLD_SC: tl.constexpr,
        GLD_SN: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        GST_SB: tl.constexpr,
        GST_ST: tl.constexpr,
        GST_SC: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        state_index = tl.arange(0, BLOCK_N)
        state_mask = state_index < N
        if HAS_GRAD_FINAL:
            carry = tl.load(
                GRAD_FINAL + batch * C * N + channel * N + state_index,
                state_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            carry = tl.full((BLOCK_N,), 0.0, tl.float32)
        for reverse_index in range(T):
            token = T - reverse_index - 1
            x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC).to(tl.float32)
            grad_output = tl.load(GRAD_OUTPUT + batch * T * C + token * C + channel).to(
                tl.float32
            )
            input_gate = tl.load(
                INPUT_GATE
                + batch * IG_SB
                + token * IG_ST
                + channel * IG_SC
                + state_index * IG_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            read_gate = tl.load(
                READ_GATE
                + batch * RG_SB
                + token * RG_ST
                + channel * RG_SC
                + state_index * RG_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            log_decay = tl.load(
                LOG_DECAY
                + batch * LD_SB
                + token * LD_ST
                + channel * LD_SC
                + state_index * LD_SN,
                state_mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_STEP_SIZE:
                step = tl.load(
                    STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC
                ).to(tl.float32)
            else:
                step = 1.0
            decay = tl.exp(log_decay * step)
            state_offset = batch * T * C * N + token * C * N + channel * N + state_index
            state_after = tl.load(STATES + state_offset, state_mask, other=0.0).to(
                tl.float32
            )
            if token > 0:
                state_before = tl.load(
                    STATES + state_offset - C * N, state_mask, other=0.0
                ).to(tl.float32)
            elif HAS_INITIAL:
                state_before = tl.load(
                    INITIAL + batch * C * N + channel * N + state_index,
                    state_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                state_before = tl.full((BLOCK_N,), 0.0, tl.float32)
            if READ_BEFORE:
                grad_read = grad_output * state_before
                state_cotangent = carry
                grad_input = state_cotangent * x * step
                grad_decay = state_cotangent * decay * state_before * step
                grad_x = tl.sum(state_cotangent * input_gate * step, 0)
                if HAS_STEP_SIZE:
                    grad_step = tl.sum(
                        state_cotangent
                        * (decay * log_decay * state_before + x * input_gate),
                        0,
                    )
                carry = state_cotangent * decay + grad_output * read_gate
            else:
                grad_read = grad_output * state_after
                state_cotangent = carry + grad_output * read_gate
                grad_input = state_cotangent * x * step
                grad_decay = state_cotangent * decay * state_before * step
                grad_x = tl.sum(state_cotangent * input_gate * step, 0)
                if HAS_STEP_SIZE:
                    grad_step = tl.sum(
                        state_cotangent
                        * (decay * log_decay * state_before + x * input_gate),
                        0,
                    )
                carry = state_cotangent * decay
            skip = tl.load(SKIP + (0 if SKIP_SCALAR else channel)).to(tl.float32)
            grad_x += grad_output * skip
            tl.store(
                GRAD_INPUT_GATE
                + batch * GIG_SB
                + token * GIG_ST
                + channel * GIG_SC
                + state_index * GIG_SN,
                grad_input,
                state_mask,
            )
            tl.store(
                GRAD_READ_GATE
                + batch * GRG_SB
                + token * GRG_ST
                + channel * GRG_SC
                + state_index * GRG_SN,
                grad_read,
                state_mask,
            )
            tl.store(
                GRAD_LOG_DECAY
                + batch * GLD_SB
                + token * GLD_ST
                + channel * GLD_SC
                + state_index * GLD_SN,
                grad_decay,
                state_mask,
            )
            tl.store(GRAD_X + batch * T * C + token * C + channel, grad_x)
            if HAS_STEP_SIZE:
                tl.store(
                    GRAD_STEP_SIZE + batch * GST_SB + token * GST_ST + channel * GST_SC,
                    grad_step,
                )
        if HAS_INITIAL:
            tl.store(
                GRAD_INITIAL + batch * C * N + channel * N + state_index,
                carry,
                state_mask,
            )

    @triton.jit
    def backward_kernel_parallel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        INITIAL,
        STEP_SIZE,
        SKIP,
        GRAD_OUTPUT,
        STATES,
        GRAD_X,
        GRAD_INPUT_GATE,
        GRAD_READ_GATE,
        GRAD_LOG_DECAY,
        GRAD_INITIAL,
        GRAD_STEP_SIZE,
        GRAD_FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        N: tl.constexpr,
        X_SB: tl.constexpr,
        X_ST: tl.constexpr,
        X_SC: tl.constexpr,
        IG_SB: tl.constexpr,
        IG_ST: tl.constexpr,
        IG_SC: tl.constexpr,
        IG_SN: tl.constexpr,
        RG_SB: tl.constexpr,
        RG_ST: tl.constexpr,
        RG_SC: tl.constexpr,
        RG_SN: tl.constexpr,
        LD_SB: tl.constexpr,
        LD_ST: tl.constexpr,
        LD_SC: tl.constexpr,
        LD_SN: tl.constexpr,
        GIG_SB: tl.constexpr,
        GIG_ST: tl.constexpr,
        GIG_SC: tl.constexpr,
        GIG_SN: tl.constexpr,
        GRG_SB: tl.constexpr,
        GRG_ST: tl.constexpr,
        GRG_SC: tl.constexpr,
        GRG_SN: tl.constexpr,
        GLD_SB: tl.constexpr,
        GLD_ST: tl.constexpr,
        GLD_SC: tl.constexpr,
        GLD_SN: tl.constexpr,
        STEP_SB: tl.constexpr,
        STEP_ST: tl.constexpr,
        STEP_SC: tl.constexpr,
        GST_SB: tl.constexpr,
        GST_ST: tl.constexpr,
        GST_SC: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        HAS_STEP_SIZE: tl.constexpr,
        SKIP_SCALAR: tl.constexpr,
        HAS_GRAD_FINAL: tl.constexpr,
        GATES_ONE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        token = tl.arange(0, BLOCK_T)
        state_index = tl.arange(0, BLOCK_N)
        token_mask = token < T
        state_mask = state_index < N
        mask2 = token_mask[:, None] & state_mask[None, :]
        x = tl.load(X + batch * X_SB + token * X_ST + channel * X_SC, token_mask, other=0.0).to(tl.float32)
        grad_output = tl.load(
            GRAD_OUTPUT + batch * T * C + token * C + channel, token_mask, other=0.0
        ).to(tl.float32)
        if GATES_ONE:
            ig = tl.full((BLOCK_T, BLOCK_N), 1.0, tl.float32)
            rg = tl.full((BLOCK_T, BLOCK_N), 1.0, tl.float32)
        else:
            ig = tl.load(
                INPUT_GATE + batch * IG_SB + token[:, None] * IG_ST + channel * IG_SC + state_index[None, :] * IG_SN,
                mask2, other=0.0,
            ).to(tl.float32)
            rg = tl.load(
                READ_GATE + batch * RG_SB + token[:, None] * RG_ST + channel * RG_SC + state_index[None, :] * RG_SN,
                mask2, other=0.0,
            ).to(tl.float32)
        ld = tl.load(
            LOG_DECAY + batch * LD_SB + token[:, None] * LD_ST + channel * LD_SC + state_index[None, :] * LD_SN,
            mask2, other=0.0,
        ).to(tl.float32)
        if HAS_STEP_SIZE:
            step = tl.load(
                STEP_SIZE + batch * STEP_SB + token * STEP_ST + channel * STEP_SC, token_mask, other=1.0
            ).to(tl.float32)
        else:
            step = tl.full((BLOCK_T,), 1.0, tl.float32)
        decay = tl.exp(ld * step[:, None])
        decay = tl.where(mask2, decay, 1.0)
        state_after = tl.load(
            STATES + batch * T * C * N + token[:, None] * C * N + channel * N + state_index[None, :],
            mask2, other=0.0,
        ).to(tl.float32)
        prev_offset = batch * T * C * N + (token[:, None] - 1) * C * N + channel * N + state_index[None, :]
        state_before = tl.load(
            STATES + prev_offset, (token[:, None] > 0) & state_mask[None, :], other=0.0
        ).to(tl.float32)
        if HAS_INITIAL:
            init = tl.load(INITIAL + batch * C * N + channel * N + state_index, state_mask, other=0.0).to(tl.float32)
            state_before = tl.where((token[:, None] == 0) & state_mask[None, :], init[None, :], state_before)
        ld_next = tl.load(
            LOG_DECAY + batch * LD_SB + (token[:, None] + 1) * LD_ST + channel * LD_SC + state_index[None, :] * LD_SN,
            (token[:, None] + 1 < T) & state_mask[None, :], other=0.0,
        ).to(tl.float32)
        if HAS_STEP_SIZE:
            step_next = tl.load(
                STEP_SIZE + batch * STEP_SB + (token + 1) * STEP_ST + channel * STEP_SC,
                token + 1 < T, other=1.0,
            ).to(tl.float32)
        else:
            step_next = tl.full((BLOCK_T,), 1.0, tl.float32)
        decay_next = tl.exp(ld_next * step_next[:, None])
        g_t = grad_output[:, None] * rg
        if HAS_GRAD_FINAL:
            gf = tl.load(GRAD_FINAL + batch * C * N + channel * N + state_index, state_mask, other=0.0).to(tl.float32)
        else:
            gf = tl.zeros((BLOCK_N,), tl.float32)
        is_last = (token == (T - 1))[:, None]
        a = tl.where(is_last, 0.0, decay_next)
        b = tl.where(is_last, gf[None, :] + g_t, g_t)
        a = tl.where(mask2, a, 1.0)
        b = tl.where(mask2, b, 0.0)
        _, sc = tl.associative_scan((a, b), axis=0, combine_fn=compose_affine, reverse=True)
        skip_index = 0 if SKIP_SCALAR else channel
        skip = tl.load(SKIP + skip_index).to(tl.float32)
        grad_read = grad_output[:, None] * state_after
        grad_input = sc * (x * step)[:, None]
        grad_decay = sc * decay * state_before * step[:, None]
        grad_x = tl.sum(sc * ig * step[:, None], 1) + grad_output * skip
        tl.store(
            GRAD_INPUT_GATE + batch * GIG_SB + token[:, None] * GIG_ST + channel * GIG_SC + state_index[None, :] * GIG_SN,
            grad_input, mask2,
        )
        tl.store(
            GRAD_READ_GATE + batch * GRG_SB + token[:, None] * GRG_ST + channel * GRG_SC + state_index[None, :] * GRG_SN,
            grad_read, mask2,
        )
        tl.store(
            GRAD_LOG_DECAY + batch * GLD_SB + token[:, None] * GLD_ST + channel * GLD_SC + state_index[None, :] * GLD_SN,
            grad_decay, mask2,
        )
        tl.store(GRAD_X + batch * T * C + token * C + channel, grad_x, token_mask)
        if HAS_STEP_SIZE:
            grad_step = tl.sum(sc * (decay * ld * state_before + x[:, None] * ig), 1)
            tl.store(
                GRAD_STEP_SIZE + batch * GST_SB + token * GST_ST + channel * GST_SC, grad_step, token_mask
            )
        if HAS_INITIAL:
            carry0 = tl.sum(tl.where((token == 0)[:, None], sc * decay, 0.0), axis=0)
            tl.store(GRAD_INITIAL + batch * C * N + channel * N + state_index, carry0, state_mask)

    @triton.jit
    def diagonal_decode_step_kernel(
        X,
        INPUT_GATE,
        READ_GATE,
        LOG_DECAY,
        STATE,
        OUTPUT,
        SKIP,
        C: tl.constexpr,
        N: tl.constexpr,
        READ_BEFORE: tl.constexpr,
        GATES_ONE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One fused single-token diagonal recurrence decode step, in place.

        Follows the ATMA decode-kernel pattern: the persistent [C, N] state is
        read and written once, in place, under no_grad with no per-step state
        history and no host sync, so the step is CUDA-graph capturable. One
        program owns one (batch, channel) pair.
        """
        row = tl.program_id(0)
        batch = row // C
        channel = row % C
        state_index = tl.arange(0, BLOCK_N)
        state_mask = state_index < N
        state_offset = batch * C * N + channel * N + state_index
        state = tl.load(STATE + state_offset, state_mask, other=0.0).to(tl.float32)
        x = tl.load(X + batch * C + channel).to(tl.float32)
        if GATES_ONE:
            input_gate = tl.full((BLOCK_N,), 1.0, tl.float32)
            read_gate = tl.full((BLOCK_N,), 1.0, tl.float32)
        else:
            input_gate = tl.load(
                INPUT_GATE + batch * C * N + channel * N + state_index, state_mask, other=0.0
            ).to(tl.float32)
            read_gate = tl.load(
                READ_GATE + batch * C * N + channel * N + state_index, state_mask, other=0.0
            ).to(tl.float32)
        log_decay = tl.load(
            LOG_DECAY + batch * C * N + channel * N + state_index, state_mask, other=0.0
        ).to(tl.float32)
        skip = tl.load(SKIP + channel).to(tl.float32)
        if READ_BEFORE:
            output = tl.sum(state * read_gate, 0) + x * skip
            tl.store(OUTPUT + batch * C + channel, output)
        state = tl.exp(log_decay) * state + x * input_gate
        if not READ_BEFORE:
            output = tl.sum(state * read_gate, 0) + x * skip
            tl.store(OUTPUT + batch * C + channel, output)
        tl.store(STATE + state_offset, state, state_mask)

    return triton, forward_kernel, backward_kernel, backward_kernel_parallel, diagonal_decode_step_kernel


def execute_diagonal_recurrence(
    *,
    x: Any,
    input_gate: Any,
    read_gate: Any,
    log_decay: Any,
    initial_state: Any | None,
    step_size: Any | None,
    skip: Any,
    read_before: bool,
    gates_one: bool = False,
) -> tuple[Any, Any]:
    """Run the fused recurrence and return output plus the final FP32 state.

    ``gates_one`` marks the all-ones gate case (HGRN): the kernels skip loading
    input_gate/read_gate and treat them as 1.0, avoiding both the gate memory
    traffic and the caller's per-call ones-tensor creation on the
    ordinary-invocation path.
    """
    import torch

    triton, forward_kernel, backward_kernel, backward_kernel_parallel, _ = _diagonal_kernels()
    if x.device.type != "cuda":
        raise ValueError("native diagonal SSM requires CUDA tensors")
    _SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("native diagonal SSM supports float32, bfloat16, float16")
    batch, sequence, channels = x.shape
    state_width = input_gate.shape[-1]
    if max(batch, sequence, channels, state_width) <= 0:
        raise ValueError("native diagonal SSM dimensions must be positive")
    if any(
        tensor.dtype not in _SUPPORTED_DTYPES
        for tensor in (input_gate, read_gate, log_decay)
    ):
        raise ValueError("native diagonal SSM gates must be float32, bfloat16, or float16")
    if initial_state is None:
        initial_tensor = _placeholder(x.device, batch, channels, state_width)
        has_initial = False
    else:
        initial_tensor = initial_state.contiguous()
        has_initial = True
        if initial_tensor.shape != (batch, channels, state_width):
            raise ValueError("initial_state must have shape [B,C,N]")
        if initial_tensor.dtype is not torch.float32:
            raise ValueError("native diagonal initial_state must be float32")
    input_gate = _expand_gate(input_gate, batch, sequence, channels)
    read_gate = _expand_gate(read_gate, batch, sequence, channels)
    log_decay = _expand_gate(log_decay, batch, sequence, channels)
    has_step_size = step_size is not None
    if step_size is None:
        step_tensor = _placeholder(x.device, batch, sequence, channels)
    else:
        if step_size.shape not in ((batch, sequence), (batch, sequence, channels)):
            raise ValueError("step_size must use [B,T] or [B,T,C]")
        if step_size.ndim == 2:
            step_size = step_size.unsqueeze(-1).expand(batch, sequence, channels)
        if step_size.dtype is not torch.float32 or step_size.device != x.device:
            raise ValueError(
                "native diagonal step_size must be float32 on the x device"
            )
        step_tensor = step_size
    if any(
        tensor.device != x.device
        for tensor in (input_gate, read_gate, log_decay, initial_tensor, step_tensor)
    ):
        raise ValueError("native diagonal SSM tensors must share a device")
    skip_tensor = _skip_tensor(skip, x.device)
    skip_scalar = skip_tensor.numel() == 1
    if not skip_scalar and tuple(skip_tensor.shape) != (channels,):
        raise ValueError("skip must be a scalar or channel vector")
    skip_tensor = skip_tensor.contiguous()
    block_n = triton.next_power_of_2(state_width)
    block_t = triton.next_power_of_2(sequence)
    chunk_t = min(block_t, 128)
    warps = 4 if block_n <= 128 else 8

    scan_cls = _scan_class(
        sequence, channels, state_width, has_initial, has_step_size, skip_scalar,
        read_before, gates_one, block_t, block_n, chunk_t, warps,
    )
    return scan_cls.apply(
        x, input_gate, read_gate, log_decay, initial_tensor, step_tensor, skip_tensor
    )


@lru_cache(maxsize=128)
def _scan_class(
    sequence, channels, state_width, has_initial, has_step_size, skip_scalar,
    read_before, gates_one, block_t, block_n, chunk_t, warps,
):
    import torch

    triton, forward_kernel, backward_kernel, backward_kernel_parallel, _ = _diagonal_kernels()
    batch = None

    class _DiagonalScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, input_gate, read_gate, log_decay, initial, step, skip):
            batch = x.shape[0]
            output = torch.empty(
                (batch, sequence, channels), device=x.device, dtype=x.dtype
            )
            states = torch.empty(
                (batch, sequence, channels, state_width),
                device=x.device,
                dtype=torch.float32,
            )
            final = torch.empty(
                (batch, channels, state_width), device=x.device, dtype=torch.float32
            )
            forward_kernel[(batch * channels,)](
                x,
                input_gate,
                read_gate,
                log_decay,
                initial,
                step,
                skip,
                output,
                states,
                final,
                sequence,
                channels,
                state_width,
                *x.stride(),
                *input_gate.stride(),
                *read_gate.stride(),
                *log_decay.stride(),
                has_initial,
                has_step_size,
                *step.stride(),
                skip_scalar,
                read_before,
                gates_one,
                block_t,
                block_n,
                chunk_t,
                num_warps=warps,
            )
            ctx.save_for_backward(
                x, input_gate, read_gate, log_decay, initial, step, skip, states
            )
            ctx.has_initial = has_initial
            ctx.has_step_size = has_step_size
            ctx.read_before = read_before
            ctx.skip_scalar = skip_scalar
            ctx.gates_one = gates_one
            return output, final

        @staticmethod
        def backward(ctx, grad_output, grad_final):
            x, input_gate, read_gate, log_decay, initial, step, skip, states = (
                ctx.saved_tensors
            )
            batch = x.shape[0]
            if grad_output is None:
                grad_output = torch.zeros_like(x)
            grad_output = grad_output.contiguous()
            grad_x = torch.empty(
                (batch, sequence, channels), device=x.device, dtype=x.dtype
            )
            grad_input_gate = torch.empty_like(
                input_gate, memory_format=torch.contiguous_format
            )
            grad_read_gate = torch.empty_like(
                read_gate, memory_format=torch.contiguous_format
            )
            grad_log_decay = torch.empty_like(
                log_decay, memory_format=torch.contiguous_format
            )
            grad_initial = torch.empty_like(initial)
            grad_step = torch.empty_like(step, memory_format=torch.contiguous_format)
            grad_final_tensor = (
                torch.zeros_like(initial)
                if grad_final is None
                else grad_final.contiguous()
            )
            if not ctx.read_before:
                backward_kernel_parallel[(batch * channels,)](
                    x,
                    input_gate,
                    read_gate,
                    log_decay,
                    initial,
                    step,
                    skip,
                    grad_output,
                    states,
                    grad_x,
                    grad_input_gate,
                    grad_read_gate,
                    grad_log_decay,
                    grad_initial,
                    grad_step,
                    grad_final_tensor,
                    sequence,
                    channels,
                    state_width,
                    *x.stride(),
                    *input_gate.stride(),
                    *read_gate.stride(),
                    *log_decay.stride(),
                    *grad_input_gate.stride(),
                    *grad_read_gate.stride(),
                    *grad_log_decay.stride(),
                    *step.stride(),
                    *grad_step.stride(),
                    ctx.has_initial,
                    ctx.has_step_size,
                    ctx.skip_scalar,
                    grad_final is not None,
                    gates_one,
                    block_t,
                    block_n,
                    num_warps=warps,
                )
            else:
                backward_kernel[(batch * channels,)](
                    x,
                    input_gate,
                    read_gate,
                    log_decay,
                    initial,
                    step,
                    skip,
                    grad_output,
                    states,
                    grad_x,
                    grad_input_gate,
                    grad_read_gate,
                    grad_log_decay,
                    grad_initial,
                    grad_step,
                    grad_final_tensor,
                    sequence,
                    channels,
                    state_width,
                    *x.stride(),
                    *input_gate.stride(),
                    *read_gate.stride(),
                    *log_decay.stride(),
                    *grad_input_gate.stride(),
                    *grad_read_gate.stride(),
                    *grad_log_decay.stride(),
                    *step.stride(),
                    *grad_step.stride(),
                    ctx.has_initial,
                    ctx.has_step_size,
                    ctx.skip_scalar,
                    ctx.read_before,
                    grad_final is not None,
                    block_n,
                    num_warps=warps,
                )
            grad_skip_full = grad_output * x
            if ctx.skip_scalar:
                grad_skip = grad_skip_full.sum().reshape_as(skip)
            else:
                grad_skip = grad_skip_full.sum(dim=(0, 1)).reshape_as(skip)
            return (
                grad_x,
                None if ctx.gates_one else grad_input_gate,
                None if ctx.gates_one else grad_read_gate,
                grad_log_decay,
                grad_initial if ctx.has_initial else None,
                grad_step if ctx.has_step_size else None,
                grad_skip,
            )

    return _DiagonalScan


def _expand_gate(gate: Any, batch: int, sequence: int, channels: int):
    if gate.ndim == 3:
        if gate.shape[:2] != (batch, sequence):
            raise ValueError("diagonal SSM gate batch/sequence shape must match x")
        return gate.unsqueeze(2).expand(batch, sequence, channels, gate.shape[-1])
    if gate.ndim == 4:
        if gate.shape[:2] != (batch, sequence) or gate.shape[2] not in (1, channels):
            raise ValueError("diagonal SSM gate must use channel width one or C")
        return gate.expand(batch, sequence, channels, gate.shape[-1])
    raise ValueError("diagonal SSM gates use [B,T,N] or [B,T,C,N]")


@lru_cache(maxsize=64)
def _placeholder(device: Any, *shape: int):
    """A cached uninitialized placeholder for tensors the kernel never reads.

    The kernel unpacks these tensors' strides but skips the loads (the matching
    HAS_* flag is false), so their contents are irrelevant. Caching avoids a
    per-call allocation on the ordinary-invocation path. Contents are never read,
    so sharing one buffer across calls is safe.
    """
    import torch

    return torch.empty(shape if shape else (1,), device=device, dtype=torch.float32)


@lru_cache(maxsize=64)
def _skip_tensor_cached(device: Any, value: float):
    import torch

    return torch.full((1,), value, device=device, dtype=torch.float32)


def _skip_tensor(skip: Any, device: Any):
    import torch

    if isinstance(skip, (int, float)):
        return _skip_tensor_cached(device, float(skip))
    return torch.as_tensor(skip, device=device, dtype=torch.float32)


def execute_diagonal_decode_step(
    *,
    x: Any,
    log_decay: Any,
    input_gate: Any | None,
    read_gate: Any | None,
    state: Any,
    read_before: bool,
) -> Any:
    """Run one fused single-token diagonal recurrence decode step, in place.

    The persistent ``[B, C, N]`` fp32 state is read and written once, in place,
    under ``torch.no_grad()`` with no per-step state-history allocation and no
    autograd graph, so the step is CUDA-graph capturable. ``x`` is ``[B, C]``;
    ``log_decay``/``input_gate``/``read_gate`` are ``[B, C, N]`` (or ``[B, C]``
    for the HGRN gates-one path); ``state`` is the persistent ``[B, C, N]`` fp32
    tensor updated in place. Returns the output ``[B, C]``.
    """
    import torch

    triton, _, _, _, decode_step_kernel = _diagonal_kernels()
    if x.device.type != "cuda":
        raise ValueError("native diagonal decode step requires CUDA tensors")
    batch, channels = x.shape
    if tuple(state.shape)[:2] != (batch, channels):
        raise ValueError("state must use [B,C,N] matching x")
    if state.dtype is not torch.float32:
        raise ValueError("the persistent diagonal state is fp32")
    state_width = state.shape[-1]
    gates_one = input_gate is None and read_gate is None
    skip_tensor = _skip_tensor(0.0, x.device)
    def _expand(gate, name):
        if gate is None:
            return log_decay
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        if gate.shape != (batch, channels, state_width):
            gate = gate.expand(batch, channels, state_width)
        return gate.contiguous()

    log_decay_e = log_decay.unsqueeze(-1) if log_decay.dim() == 2 else log_decay
    log_decay_e = log_decay_e.expand(batch, channels, state_width).contiguous()
    ig = _expand(input_gate, "input_gate")
    rg = _expand(read_gate, "read_gate")
    output = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
    block_n = triton.next_power_of_2(state_width)
    skip_c = skip_tensor.contiguous()
    if skip_c.numel() == 1:
        skip_c = skip_c.expand(channels).contiguous()
    with torch.no_grad():
        decode_step_kernel[(batch * channels,)](
            x.contiguous(), ig, rg, log_decay_e, state, output, skip_c,
            channels, state_width, read_before, gates_one, block_n, num_warps=4,
        )
    return output


__all__ = ["execute_diagonal_recurrence", "execute_diagonal_decode_step"]
