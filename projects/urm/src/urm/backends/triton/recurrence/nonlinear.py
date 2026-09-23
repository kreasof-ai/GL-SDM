"""Native Triton kernels for the distinguished K2 nonlinear recurrence operators.

Each kernel mirrors its NumPy canonical executor in
``urm.oracles.nonlinear_recurrence`` exactly: the same per-token math with fp32
accumulation. One program owns one (batch, head/channel) pair and scans the
sequence sequentially. Forward-only native execution (no backward).

Covered operators (canonical executor -> recipe):

- ``tanh_rnn`` (rnn_core): tanh RNN, state [B,N,H], weight [N,H,H].
- ``gated_rnn`` (gru_core): GRU, state [B,N,H], gate weights [N,H,H].
- ``multiplicative_rnn`` (m2rnn_core): second-order matrix memory [B,N,K,V].
- ``rwkv4_scalar_state`` (rwkv4_memory_core): scalar (alpha, denom, log_scale)
  per channel, state [B,3,1,C].
- ``rwkv6_bonus_corrected`` (rwkv6_memory_core): matrix state [B,H,K,V].
- ``mamba2_structured_ssm`` (mamba2_ssm_core): structured SSM state [B,H,P,N].
- ``trapezoidal_ssm`` (mamba3_siso_core): rotary + trapezoidal four-state SSM.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def tanh_rnn_forward_kernel(
        Q,
        W,
        INITIAL,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // N
        n = row % N
        h_index = tl.arange(0, BLOCK_H)
        h_mask = h_index < H
        state = tl.load(
            INITIAL + batch * N * H + n * H + h_index, h_mask, other=0.0
        ).to(tl.float32)
        w_tile = tl.load(
            W + n * H * H + h_index[:, None] * H + h_index[None, :],
            h_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        for token in range(T):
            x_t = tl.load(
                Q + batch * (T * N * H) + token * (N * H) + n * H + h_index,
                h_mask,
                other=0.0,
            ).to(tl.float32)
            acc = tl.sum(state[:, None] * w_tile, axis=0)
            state = libdevice.tanh(acc + x_t)
            tl.store(
                OUTPUT + batch * (T * N * H) + token * (N * H) + n * H + h_index,
                state,
                h_mask,
            )
        tl.store(FINAL + batch * N * H + n * H + h_index, state, h_mask)

    @triton.jit
    def gated_rnn_forward_kernel(
        Q,
        W,
        FI,
        FW,
        RI,
        RW,
        INITIAL,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // N
        n = row % N
        h_index = tl.arange(0, BLOCK_H)
        h_mask = h_index < H
        full_mask = h_mask[:, None] & h_mask[None, :]
        state = tl.load(
            INITIAL + batch * N * H + n * H + h_index, h_mask, other=0.0
        ).to(tl.float32)
        w_tile = tl.load(
            W + n * H * H + h_index[:, None] * H + h_index[None, :],
            full_mask,
            other=0.0,
        ).to(tl.float32)
        fw_tile = tl.load(
            FW + n * H * H + h_index[:, None] * H + h_index[None, :],
            full_mask,
            other=0.0,
        ).to(tl.float32)
        rw_tile = tl.load(
            RW + n * H * H + h_index[:, None] * H + h_index[None, :],
            full_mask,
            other=0.0,
        ).to(tl.float32)
        token_base = batch * (T * N * H) + n * H
        for token in range(T):
            off = token_base + token * (N * H) + h_index
            fi_t = tl.load(FI + off, h_mask, other=0.0).to(tl.float32)
            ri_t = tl.load(RI + off, h_mask, other=0.0).to(tl.float32)
            x_t = tl.load(Q + off, h_mask, other=0.0).to(tl.float32)
            forget = tl.sigmoid(tl.sum(state[:, None] * fw_tile, axis=0) + fi_t)
            reset = tl.sigmoid(tl.sum(state[:, None] * rw_tile, axis=0) + ri_t)
            sr = state * reset
            candidate = libdevice.tanh(tl.sum(sr[:, None] * w_tile, axis=0) + x_t)
            state = forget * state + (1.0 - forget) * candidate
            tl.store(OUTPUT + off, state, h_mask)
        tl.store(FINAL + batch * N * H + n * H + h_index, state, h_mask)

    @triton.jit
    def multiplicative_rnn_forward_kernel(
        Q,
        K,
        V,
        W,
        FI,
        INITIAL,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        N: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // N
        n = row % N
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_offset = (
            batch * N * K_DIM * V_DIM
            + n * K_DIM * V_DIM
            + k_index[:, None] * V_DIM
            + v_index[None, :]
        )
        state = tl.load(INITIAL + state_offset, kv_mask, other=0.0).to(tl.float32)
        w_tile = tl.load(
            W + n * V_DIM * V_DIM + v_index[:, None] * V_DIM + v_index[None, :],
            v_mask[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        qk_token_base = batch * (T * N * K_DIM) + n * K_DIM
        v_token_base = batch * (T * N * V_DIM) + n * V_DIM
        fi_token_base = batch * (T * N) + n
        for token in range(T):
            k_t = tl.load(
                K + qk_token_base + token * (N * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(
                V + v_token_base + token * (N * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)
            q_t = tl.load(
                Q + qk_token_base + token * (N * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            forget = tl.load(FI + fi_token_base + token * N).to(tl.float32)
            update = k_t[:, None] * v_t[None, :]
            sw = tl.sum(state[:, :, None] * w_tile[None, :, :], axis=1)
            candidate = libdevice.tanh(sw + update)
            state = forget * state + (1.0 - forget) * candidate
            read = tl.sum(state * q_t[:, None], axis=0)
            tl.store(
                OUTPUT + v_token_base + token * (N * V_DIM) + v_index, read, v_mask
            )
        tl.store(FINAL + state_offset, state, kv_mask)

    @triton.jit
    def rwkv4_forward_kernel(
        W,
        U,
        KEY,
        VALUE,
        STATE_IN,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_index = tl.arange(0, BLOCK_C)
        c_mask = c_index < C
        w = tl.load(W + c_index, c_mask, other=0.0).to(tl.float32)
        u = tl.load(U + c_index, c_mask, other=0.0).to(tl.float32)
        decay = -tl.exp(w)
        bonus = u
        alpha = tl.load(
            STATE_IN + batch * 3 * C + 0 * C + c_index, c_mask, other=0.0
        ).to(tl.float32)
        denom = tl.load(
            STATE_IN + batch * 3 * C + 1 * C + c_index, c_mask, other=0.0
        ).to(tl.float32)
        log_scale = tl.load(
            STATE_IN + batch * 3 * C + 2 * C + c_index, c_mask, other=0.0
        ).to(tl.float32)
        for token in range(T):
            key_t = tl.load(
                KEY + batch * T * C + token * C + c_index, c_mask, other=0.0
            ).to(tl.float32)
            value_t = tl.load(
                VALUE + batch * T * C + token * C + c_index, c_mask, other=0.0
            ).to(tl.float32)
            bonus_key = bonus + key_t
            read_scale = tl.maximum(log_scale, bonus_key)
            read_state_scale = tl.exp(log_scale - read_scale)
            read_value_scale = tl.exp(bonus_key - read_scale)
            output_t = (read_state_scale * alpha + read_value_scale * value_t) / (
                read_state_scale * denom + read_value_scale
            )
            tl.store(
                OUTPUT + batch * T * C + token * C + c_index, output_t, c_mask
            )
            decayed_scale = decay + log_scale
            log_scale_next = tl.maximum(decayed_scale, key_t)
            old_scale = tl.exp(decayed_scale - log_scale_next)
            new_scale = tl.exp(key_t - log_scale_next)
            alpha = old_scale * alpha + new_scale * value_t
            denom = old_scale * denom + new_scale
            log_scale = log_scale_next
        tl.store(FINAL + batch * 3 * C + 0 * C + c_index, alpha, c_mask)
        tl.store(FINAL + batch * 3 * C + 1 * C + c_index, denom, c_mask)
        tl.store(FINAL + batch * 3 * C + 2 * C + c_index, log_scale, c_mask)

    @triton.jit
    def rwkv6_forward_kernel(
        Q,
        K,
        V,
        G,
        BONUS,
        INITIAL,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        H: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
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
        bonus = tl.load(BONUS + head * K_DIM + k_index, k_mask, other=0.0).to(
            tl.float32
        )
        qk_token_base = batch * (T * H * K_DIM) + head * K_DIM
        v_token_base = batch * (T * H * V_DIM) + head * V_DIM
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
            g_t = tl.load(
                G + qk_token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            decay = tl.exp(g_t)
            decayed_state = state * decay[:, None]
            bonus_write = (k_t * bonus)[:, None] * v_t[None, :]
            read_state = state + bonus_write
            output = SCALE * tl.sum(read_state * q_t[:, None], axis=0)
            tl.store(
                OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                output,
                v_mask,
            )
            state = decayed_state + k_t[:, None] * v_t[None, :]
        tl.store(FINAL + state_offset, state, kv_mask)

    @triton.jit
    def mamba2_forward_kernel(
        X,
        DT,
        A,
        B,
        C,
        INITIAL,
        OUTPUT,
        FINAL,
        T: tl.constexpr,
        H: tl.constexpr,
        P: tl.constexpr,
        G: tl.constexpr,
        N: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        group = head // (H // G)
        p_index = tl.arange(0, BLOCK_P)
        n_index = tl.arange(0, BLOCK_N)
        p_mask = p_index < P
        n_mask = n_index < N
        pn_mask = p_mask[:, None] & n_mask[None, :]
        state_base = ((batch * H + head) * P) * N
        state_offset = state_base + p_index[:, None] * N + n_index[None, :]
        if HAS_INITIAL:
            state = tl.load(INITIAL + state_offset, pn_mask, other=0.0).to(tl.float32)
        else:
            state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
        a = tl.load(A + head).to(tl.float32)
        x_token_base = batch * (T * H * P) + head * P
        dt_token_base = batch * (T * H) + head
        bc_token_base = batch * (T * G * N) + group * N
        for token in range(T):
            step = tl.load(DT + dt_token_base + token * H).to(tl.float32)
            decay = tl.exp(step * a)
            x_t = tl.load(
                X + x_token_base + token * (H * P) + p_index, p_mask, other=0.0
            ).to(tl.float32)
            b_t = tl.load(
                B + bc_token_base + token * (G * N) + n_index, n_mask, other=0.0
            ).to(tl.float32)
            c_t = tl.load(
                C + bc_token_base + token * (G * N) + n_index, n_mask, other=0.0
            ).to(tl.float32)
            state = state * decay + (x_t[:, None] * b_t[None, :]) * step
            output = tl.sum(state * c_t[None, :], axis=1)
            tl.store(
                OUTPUT + x_token_base + token * (H * P) + p_index, output, p_mask
            )
        tl.store(FINAL + state_offset, state, pn_mask)

    @triton.jit
    def trapezoidal_ssm_forward_kernel(
        Q,
        K,
        V,
        ADT,
        DT,
        TRAP,
        Q_BIAS,
        K_BIAS,
        ANGLES,
        OUTPUT,
        FINAL_ANGLE,
        FINAL_SSM,
        FINAL_KEY,
        FINAL_VALUE,
        T: tl.constexpr,
        H: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        A_DIM: tl.constexpr,
        BLOCK_KP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_A: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        half_k = K_DIM // 2
        p_index = tl.arange(0, BLOCK_KP)
        p_mask = p_index < half_k
        k_index = tl.arange(0, BLOCK_K)
        k_mask = k_index < K_DIM
        v_index = tl.arange(0, BLOCK_V)
        v_mask = v_index < V_DIM
        vk_mask = v_mask[:, None] & k_mask[None, :]
        a_index = tl.arange(0, BLOCK_A)
        a_mask = a_index < A_DIM

        angle_pair = tl.zeros((BLOCK_KP,), dtype=tl.float32)
        ssm_state = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        key_state = tl.zeros((BLOCK_K,), dtype=tl.float32)
        value_state = tl.zeros((BLOCK_V,), dtype=tl.float32)

        qk_token_base = batch * (T * H * K_DIM) + head * K_DIM
        v_token_base = batch * (T * H * V_DIM) + head * V_DIM
        sched_token_base = batch * (H * T) + head * T
        ang_token_base = batch * (T * H * A_DIM) + head * A_DIM

        PI = 3.141592653589793
        TWO_PI = 6.283185307179586

        k_match_first = (k_index[:, None] == (p_index * 2)[None, :]) & (
            k_mask[:, None] & p_mask[None, :]
        )
        k_match_second = (k_index[:, None] == (p_index * 2 + 1)[None, :]) & (
            k_mask[:, None] & p_mask[None, :]
        )
        a_match = (a_index[:, None] == p_index[None, :]) & (
            a_mask[:, None] & (p_index < A_DIM)[None, :]
        )

        for token in range(T):
            dt_t = tl.load(DT + sched_token_base + token).to(tl.float32)
            adt_t = tl.load(ADT + sched_token_base + token).to(tl.float32)
            trap_raw = tl.load(TRAP + sched_token_base + token).to(tl.float32)
            angles_p = tl.load(
                ANGLES + ang_token_base + token * (H * A_DIM) + p_index,
                p_index < A_DIM,
                other=0.0,
            ).to(tl.float32)
            angle_pair = angle_pair + (libdevice.tanh(angles_p) * PI) * dt_t
            angle_pair = angle_pair - TWO_PI * tl.math.floor(angle_pair / TWO_PI)
            in_angle = p_index < A_DIM
            cosine = tl.where(in_angle, tl.cos(angle_pair), 1.0)
            sine = tl.where(in_angle, tl.sin(angle_pair), 0.0)

            q_first = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + p_index * 2,
                p_mask,
                other=0.0,
            ).to(tl.float32) + tl.load(
                Q_BIAS + head * K_DIM + p_index * 2, p_mask, other=0.0
            ).to(tl.float32)
            q_second = tl.load(
                Q + qk_token_base + token * (H * K_DIM) + p_index * 2 + 1,
                p_mask,
                other=0.0,
            ).to(tl.float32) + tl.load(
                Q_BIAS + head * K_DIM + p_index * 2 + 1, p_mask, other=0.0
            ).to(tl.float32)
            k_first = tl.load(
                K + qk_token_base + token * (H * K_DIM) + p_index * 2,
                p_mask,
                other=0.0,
            ).to(tl.float32) + tl.load(
                K_BIAS + head * K_DIM + p_index * 2, p_mask, other=0.0
            ).to(tl.float32)
            k_second = tl.load(
                K + qk_token_base + token * (H * K_DIM) + p_index * 2 + 1,
                p_mask,
                other=0.0,
            ).to(tl.float32) + tl.load(
                K_BIAS + head * K_DIM + p_index * 2 + 1, p_mask, other=0.0
            ).to(tl.float32)
            q_rot_first = q_first * cosine - q_second * sine
            q_rot_second = q_first * sine + q_second * cosine
            k_rot_first = k_first * cosine - k_second * sine
            k_rot_second = k_first * sine + k_second * cosine

            q_rot = tl.sum(
                tl.where(k_match_first, q_rot_first[None, :], 0.0), axis=1
            ) + tl.sum(tl.where(k_match_second, q_rot_second[None, :], 0.0), axis=1)
            k_rot = tl.sum(
                tl.where(k_match_first, k_rot_first[None, :], 0.0), axis=1
            ) + tl.sum(tl.where(k_match_second, k_rot_second[None, :], 0.0), axis=1)

            v_t = tl.load(
                V + v_token_base + token * (H * V_DIM) + v_index, v_mask, other=0.0
            ).to(tl.float32)

            trap_t = tl.sigmoid(trap_raw)
            alpha = tl.exp(adt_t)
            beta = (1.0 - trap_t) * dt_t * alpha
            gamma = trap_t * dt_t

            ssm_state = (
                alpha * ssm_state
                + beta * (key_state[None, :] * value_state[:, None])
                + gamma * (k_rot[None, :] * v_t[:, None])
            )
            output = tl.sum(ssm_state * q_rot[None, :], axis=1)
            tl.store(
                OUTPUT + v_token_base + token * (H * V_DIM) + v_index,
                output,
                v_mask,
            )
            key_state = k_rot
            value_state = v_t

        final_angle = tl.sum(tl.where(a_match, angle_pair[None, :], 0.0), axis=1)
        tl.store(
            FINAL_ANGLE + batch * H * A_DIM + head * A_DIM + a_index,
            final_angle,
            a_mask,
        )
        ssm_offset = (
            batch * H * V_DIM * K_DIM
            + head * V_DIM * K_DIM
            + v_index[:, None] * K_DIM
            + k_index[None, :]
        )
        tl.store(FINAL_SSM + ssm_offset, ssm_state, vk_mask)
        tl.store(
            FINAL_KEY + batch * H * K_DIM + head * K_DIM + k_index,
            key_state,
            k_mask,
        )
        tl.store(
            FINAL_VALUE + batch * H * V_DIM + head * V_DIM + v_index,
            value_state,
            v_mask,
        )

    return (
        triton,
        tanh_rnn_forward_kernel,
        gated_rnn_forward_kernel,
        multiplicative_rnn_forward_kernel,
        rwkv4_forward_kernel,
        rwkv6_forward_kernel,
        mamba2_forward_kernel,
        trapezoidal_ssm_forward_kernel,
    )


def execute_tanh_rnn(*, query: Any, weight: Any, initial_state: Any):
    """Native tanh RNN: ``h_t = tanh(h_{t-1} @ W + x_t)``; output is the state.

    ``query`` [B,T,N,H], ``weight`` [N,H,H], ``initial_state`` [B,N,H].
    Returns (output [B,T,N,H], final_state [B,N,H]) in fp32.
    """
    import torch

    triton, kernel, *_ = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native tanh_rnn requires CUDA tensors")
    batch, sequence, n, h = query.shape
    q = query.contiguous().float()
    w = weight.contiguous().float()
    init = initial_state.contiguous().float()
    output = torch.empty((batch, sequence, n, h), device=q.device, dtype=torch.float32)
    final = torch.empty((batch, n, h), device=q.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(h)
    with torch.no_grad():
        kernel[(batch * n,)](
            q, w, init, output, final, sequence, n, h, block_h, num_warps=4
        )
    return output, final


def execute_gated_rnn(
    *,
    query: Any,
    weight: Any,
    forget_input: Any,
    forget_weight: Any,
    reset_input: Any,
    reset_weight: Any,
    initial_state: Any,
):
    """Native GRU mirroring ``gated_rnn``. Inputs [B,T,N,H] / [N,H,H].

    Returns (output [B,T,N,H], final_state [B,N,H]) in fp32.
    """
    import torch

    triton, _, kernel, *_ = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native gated_rnn requires CUDA tensors")
    batch, sequence, n, h = query.shape
    q = query.contiguous().float()
    w = weight.contiguous().float()
    fi = forget_input.contiguous().float()
    fw = forget_weight.contiguous().float()
    ri = reset_input.contiguous().float()
    rw = reset_weight.contiguous().float()
    init = initial_state.contiguous().float()
    output = torch.empty((batch, sequence, n, h), device=q.device, dtype=torch.float32)
    final = torch.empty((batch, n, h), device=q.device, dtype=torch.float32)
    block_h = triton.next_power_of_2(h)
    with torch.no_grad():
        kernel[(batch * n,)](
            q, w, fi, fw, ri, rw, init, output, final,
            sequence, n, h, block_h, num_warps=4,
        )
    return output, final


def execute_multiplicative_rnn(
    *,
    query: Any,
    key: Any,
    value: Any,
    weight: Any,
    forget_input: Any,
    initial_state: Any,
):
    """Native second-order multiplicative RNN mirroring ``multiplicative_rnn``.

    ``query``/``key`` [B,T,N,K], ``value`` [B,T,N,V], ``weight`` [N,V,V],
    ``forget_input`` [B,T,N], ``initial_state`` [B,N,K,V].
    Returns (output [B,T,N,V], final_state [B,N,K,V]) in fp32.
    """
    import torch

    triton, _, _, kernel, *_ = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native multiplicative_rnn requires CUDA tensors")
    batch, sequence, n, k_dim = query.shape
    v_dim = value.shape[-1]
    q = query.contiguous().float()
    k = key.contiguous().float()
    v = value.contiguous().float()
    w = weight.contiguous().float()
    fi = forget_input.contiguous().float()
    init = initial_state.contiguous().float()
    output = torch.empty(
        (batch, sequence, n, v_dim), device=q.device, dtype=torch.float32
    )
    final = torch.empty(
        (batch, n, k_dim, v_dim), device=q.device, dtype=torch.float32
    )
    block_k = triton.next_power_of_2(k_dim)
    block_v = triton.next_power_of_2(v_dim)
    with torch.no_grad():
        kernel[(batch * n,)](
            q, k, v, w, fi, init, output, final,
            sequence, n, k_dim, v_dim, block_k, block_v, num_warps=4,
        )
    return output, final


def execute_rwkv4_scalar_state(
    *, w: Any, u: Any, key: Any, value: Any, state_input: Any
):
    """Native RWKV-4 scalar-state recurrence mirroring ``rwkv4_scalar_state``.

    ``w``/``u`` [C]; ``key``/``value`` [B,T,C]; ``state_input`` [B,3,1,C].
    Returns (output [B,T,C], final_state [B,3,1,C]) in fp32.
    """
    import torch

    triton, _, _, _, kernel, *_ = _kernels()
    if key.device.type != "cuda":
        raise ValueError("native rwkv4 requires CUDA tensors")
    batch, sequence, channels = key.shape
    w_c = w.contiguous().float()
    u_c = u.contiguous().float()
    key_c = key.contiguous().float()
    value_c = value.contiguous().float()
    state_c = state_input.contiguous().float()
    output = torch.empty(
        (batch, sequence, channels), device=key.device, dtype=torch.float32
    )
    final = torch.empty(
        (batch, 3, 1, channels), device=key.device, dtype=torch.float32
    )
    block_c = triton.next_power_of_2(channels)
    with torch.no_grad():
        kernel[(batch,)](
            w_c, u_c, key_c, value_c, state_c, output, final,
            sequence, channels, block_c, num_warps=4,
        )
    return output, final


def execute_rwkv6_bonus_corrected(
    *,
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    bonus: Any,
    initial_state: Any | None = None,
):
    """Native RWKV-6 bonus-corrected recurrence mirroring ``rwkv6_bonus_corrected``.

    ``query``/``key`` [B,T,H,K], ``value`` [B,T,H,V], ``log_decay`` [B,T,H,K],
    ``bonus`` [H,K], ``initial_state`` [B,H,K,V] or None.
    Returns (output [B,T,H,V], final_state [B,H,K,V]) in fp32.
    """
    import torch

    triton, _, _, _, _, kernel, _, _ = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native rwkv6 requires CUDA tensors")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    scale = key_dim ** -0.5
    q = query.contiguous().float()
    k = key.contiguous().float()
    v = value.contiguous().float()
    g = log_decay.contiguous().float()
    bonus_c = bonus.contiguous().float()
    if initial_state is None:
        init = torch.zeros(
            (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
        )
        has_initial = False
    else:
        init = initial_state.contiguous().float()
        has_initial = True
    output = torch.empty(
        (batch, sequence, heads, value_dim), device=q.device, dtype=torch.float32
    )
    final = torch.empty(
        (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    with torch.no_grad():
        kernel[(batch * heads,)](
            q, k, v, g, bonus_c, init, output, final,
            sequence, heads, key_dim, value_dim, scale, has_initial,
            block_k, block_v, num_warps=4,
        )
    return output, final


def execute_mamba2_structured_ssm(
    *,
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    initial_states: Any | None = None,
):
    """Native Mamba-2 structured SSM mirroring ``mamba2_structured_ssm``.

    ``x`` [B,T,H,P]; ``dt`` [B,T,H]; ``A`` [H]; ``B``/``C`` [B,T,G,N];
    ``initial_states`` [B,H,P,N] or None.
    Returns (output [B,T,H,P], final_state [B,H,P,N]) in fp32.
    """
    import torch

    triton, _, _, _, _, _, kernel, _ = _kernels()
    if x.device.type != "cuda":
        raise ValueError("native mamba2 requires CUDA tensors")
    batch, sequence, heads, head_dim = x.shape
    state_dim = B.shape[-1]
    groups = B.shape[2]
    x_c = x.contiguous().float()
    dt_c = dt.contiguous().float()
    a_c = A.contiguous().float()
    b_c = B.contiguous().float()
    c_c = C.contiguous().float()
    if initial_states is None:
        init = torch.zeros(
            (batch, heads, head_dim, state_dim), device=x.device, dtype=torch.float32
        )
        has_initial = False
    else:
        init = initial_states.contiguous().float()
        has_initial = True
    output = torch.empty(
        (batch, sequence, heads, head_dim), device=x.device, dtype=torch.float32
    )
    final = torch.empty(
        (batch, heads, head_dim, state_dim), device=x.device, dtype=torch.float32
    )
    block_p = triton.next_power_of_2(head_dim)
    block_n = triton.next_power_of_2(state_dim)
    with torch.no_grad():
        kernel[(batch * heads,)](
            x_c, dt_c, a_c, b_c, c_c, init, output, final,
            sequence, heads, head_dim, groups, state_dim, has_initial,
            block_p, block_n, num_warps=4,
        )
    return output, final


def execute_trapezoidal_ssm(
    *,
    query: Any,
    key: Any,
    value: Any,
    adt: Any,
    dt: Any,
    trap: Any,
    query_bias: Any,
    key_bias: Any,
    angles: Any,
):
    """Native Mamba-3 trapezoidal SSM with rotary, mirroring ``trapezoidal_ssm``.

    ``query``/``key`` [B,T,H,K] (K even); ``value`` [B,T,H,V];
    ``adt``/``dt``/``trap`` [B,H,T]; ``query_bias``/``key_bias`` [H,K];
    ``angles`` [B,T,H,A]. Zero initial states. Returns
    (output [B,T,H,V], (angle [B,H,A], ssm [B,H,V,K], key [B,H,K],
    value [B,H,V])) in fp32.
    """
    import torch

    triton, *_, kernel = _kernels()
    if query.device.type != "cuda":
        raise ValueError("native trapezoidal_ssm requires CUDA tensors")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    angle_dim = angles.shape[-1]
    q = query.contiguous().float()
    k = key.contiguous().float()
    v = value.contiguous().float()
    adt_c = adt.contiguous().float()
    dt_c = dt.contiguous().float()
    trap_c = trap.contiguous().float()
    qb = query_bias.contiguous().float()
    kb = key_bias.contiguous().float()
    ang = angles.contiguous().float()
    output = torch.empty(
        (batch, sequence, heads, value_dim), device=q.device, dtype=torch.float32
    )
    final_angle = torch.empty(
        (batch, heads, angle_dim), device=q.device, dtype=torch.float32
    )
    final_ssm = torch.empty(
        (batch, heads, value_dim, key_dim), device=q.device, dtype=torch.float32
    )
    final_key = torch.empty(
        (batch, heads, key_dim), device=q.device, dtype=torch.float32
    )
    final_value = torch.empty(
        (batch, heads, value_dim), device=q.device, dtype=torch.float32
    )
    half_k = key_dim // 2
    block_kp = triton.next_power_of_2(half_k)
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    block_a = triton.next_power_of_2(angle_dim)
    with torch.no_grad():
        kernel[(batch * heads,)](
            q, k, v, adt_c, dt_c, trap_c, qb, kb, ang,
            output, final_angle, final_ssm, final_key, final_value,
            sequence, heads, key_dim, value_dim, angle_dim,
            block_kp, block_k, block_v, block_a, num_warps=4,
        )
    return output, (final_angle, final_ssm, final_key, final_value)


__all__ = [
    "execute_tanh_rnn",
    "execute_gated_rnn",
    "execute_multiplicative_rnn",
    "execute_rwkv4_scalar_state",
    "execute_rwkv6_bonus_corrected",
    "execute_mamba2_structured_ssm",
    "execute_trapezoidal_ssm",
]
