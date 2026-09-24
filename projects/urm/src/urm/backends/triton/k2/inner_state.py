"""Native executors for the distinguished K2 inner-state / convolution / solve operators.

These are URM's own native execution paths for the recurrence equations the
``recurrence_operator`` IR field distinguishes, mirroring the float64 NumPy
canonical executors in ``urm/oracles/nonlinear_recurrence.py`` exactly (the same
per-token math, fp32 accumulation). Forward-only; no backward is implemented.

Approach per operator:

- ``execute_layernorm_inner_state`` (ttt_linear_core): Triton kernel, one
  program per (batch, head) scanning the sequence chunk by chunk with the
  [D, D] memory and [D] bias held in registers (fp32).
- ``execute_momentum_inner_state`` (titans_linear_memory_core): Triton kernel,
  one program per (batch, head), tokenwise scan with the [D, D] memory and
  momentum states in registers (fp32).
- ``execute_regularized_solve`` (mesa_net_core): Triton kernel, one program per
  (batch, head); the per-token K x K regularized solve runs as an in-register
  Gaussian elimination with partial pivoting, mirroring ``np.linalg.solve``.
- ``execute_second_order_cumsum`` (hla_second_order_core): Triton kernel, one
  program per (batch, head) accumulating the inclusive prefix-sum states
  S [K, K], C [K, V], G [K, V] in registers (fp32).
- ``execute_fft_convolution`` (hyena_fftconv_core): torch-native executor using
  ``torch.fft.rfft``/``irfft`` (a Triton FFT is impractical; a torch-native FFT
  convolution is genuinely native - no upstream library dispatch).
- ``execute_two_stage_fft_convolution`` (h3_ssm_fft_core): torch-native
  executor built on the same ``torch.fft`` causal convolution.
- ``execute_slot_attention_two_stage`` (abc_core, gsa_core): Triton kernel, one
  program per (batch, query head); stage 1 accumulates the key state and the
  per-token slot scores, a softmax over slots is taken per token, then stage 2
  accumulates the value state and reads it by the slot probabilities.
- ``execute_momentum_delta`` (momentum_delta_core): Triton kernel, one program
  per (batch, head) with the [K, V] state and momentum in registers (fp32).
- ``execute_gated_oja`` (gated_oja_core): Triton kernel, one program per
  (batch, head) with the [K, V] state in registers (fp32).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def layernorm_inner_state_kernel(
        Q,
        K,
        V,
        W,
        B,
        ETA,
        MEMORY,
        MEMORY_BIAS,
        OUTPUT,
        FINAL_MEMORY,
        FINAL_BIAS,
        H: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """TTT-Linear chunkwise inner-loss update (mirrors ``layernorm_inner_state``).

        One program owns one (batch, head) pair and scans chunk by chunk; the
        [D, D] memory matrix and [D] bias stay in registers in fp32.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        d_index = tl.arange(0, BLOCK_D)
        d_mask = d_index < D
        dd_mask = d_mask[:, None] & d_mask[None, :]
        state_base = (batch * H + head) * D * D
        state_offset = state_base + d_index[:, None] * D + d_index[None, :]
        bias_base = (batch * H + head) * D
        if HAS_INITIAL:
            memory = tl.load(MEMORY + state_offset, dd_mask, other=0.0).to(tl.float32)
            memory_bias = tl.load(MEMORY_BIAS + bias_base + d_index, d_mask, other=0.0).to(
                tl.float32
            )
        else:
            memory = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
            memory_bias = tl.zeros((BLOCK_D,), dtype=tl.float32)
        w = tl.load(W + head * D + d_index, d_mask, other=0.0).to(tl.float32)
        b = tl.load(B + head * D + d_index, d_mask, other=0.0).to(tl.float32)
        token_base = batch * (T * H * D) + head * D
        for start in range(0, T, CHUNK):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(start + CHUNK, T)
            q = tl.load(
                Q + token_base + token[:, None] * (H * D) + d_index[None, :],
                token_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32) * (D ** -0.5)
            k = tl.load(
                K + token_base + token[:, None] * (H * D) + d_index[None, :],
                token_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + token_base + token[:, None] * (H * D) + d_index[None, :],
                token_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            eta = tl.load(
                ETA + batch * (T * H) + token * H + head, token_mask, other=0.0
            ).to(tl.float32)
            kh = tl.sum(memory[None, :, :] * k[:, :, None], axis=1) + memory_bias[None, :]
            target = v - k
            mean = tl.sum(kh, axis=1) / D
            centered = tl.where(d_mask[None, :], kh - mean[:, None], 0.0)
            var = tl.sum(centered * centered, axis=1) / D
            rstd = 1.0 / tl.sqrt(var + EPS)
            kh_hat = centered * rstd[:, None]
            grad = (w[None, :] * kh_hat + b[None, :] - target) * w[None, :]
            grad_sum = tl.sum(grad, axis=1)
            grad_kh_sum = tl.sum(grad * kh_hat, axis=1)
            grad = (
                D * grad - grad_sum[:, None] - kh_hat * grad_kh_sum[:, None]
            ) * (rstd / D)[:, None]
            attention = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            attention = tl.where(
                (token[:, None] >= token[None, :]) & token_mask[:, None] & token_mask[None, :],
                attention,
                0.0,
            )
            eta_tril = tl.where(
                (token[:, None] >= token[None, :]) & token_mask[:, None] & token_mask[None, :],
                eta[:, None],
                0.0,
            )
            output_chunk = (
                tl.sum(memory[None, :, :] * q[:, :, None], axis=1)
                - tl.sum(grad[None, :, :] * (attention * eta[:, None])[:, :, None], axis=1)
                + memory_bias[None, :]
                - tl.sum(grad[None, :, :] * eta_tril[:, :, None], axis=1)
            )
            last_valid = tl.sum(tl.where(token_mask, 1, 0), 0) - 1
            eta_last = tl.sum(tl.where(token == start + last_valid, eta, 0.0), 0)
            memory = memory - tl.sum(
                k[:, :, None] * grad[:, None, :], axis=0
            ) * eta_last
            memory_bias = memory_bias - eta_last * tl.sum(grad, axis=0)
            out_mean = tl.sum(output_chunk, axis=1) / D
            out_centered = tl.where(d_mask[None, :], output_chunk - out_mean[:, None], 0.0)
            out_var = tl.sum(out_centered * out_centered, axis=1) / D
            out_rstd = 1.0 / tl.sqrt(out_var + EPS)
            output = output_chunk + out_centered * out_rstd[:, None] * w[None, :] + b[None, :]
            tl.store(
                OUTPUT + token_base + token[:, None] * (H * D) + d_index[None, :],
                output,
                token_mask[:, None] & d_mask[None, :],
            )
        tl.store(FINAL_MEMORY + state_offset, memory, dd_mask)
        tl.store(FINAL_BIAS + bias_base + d_index, memory_bias, d_mask)

    @triton.jit
    def momentum_inner_state_kernel(
        Q,
        K,
        V,
        W,
        B,
        THETA,
        ALPHA,
        ETA,
        INITIAL,
        OUTPUT,
        FINAL_MEMORY,
        H: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Titans tokenwise memory with chunk-frozen target and momentum.

        One program owns one (batch, head) pair; the [D, D] memory, momentum,
        and chunk-frozen update base stay in registers in fp32.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        d_index = tl.arange(0, BLOCK_D)
        d_mask = d_index < D
        dd_mask = d_mask[:, None] & d_mask[None, :]
        state_base = (batch * H + head) * D * D
        state_offset = state_base + d_index[:, None] * D + d_index[None, :]
        if HAS_INITIAL:
            memory = tl.load(INITIAL + state_offset, dd_mask, other=0.0).to(tl.float32)
        else:
            memory = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        momentum = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        update_base = memory
        w = tl.load(W + head * D + d_index, d_mask, other=0.0).to(tl.float32)
        b = tl.load(B + head * D + d_index, d_mask, other=0.0).to(tl.float32)
        token_base = batch * (T * H * D) + head * D
        gate_base = batch * (T * H) + head
        for token in range(T):
            q_t = tl.load(Q + token_base + token * (H * D) + d_index, d_mask, other=0.0).to(
                tl.float32
            )
            k_t = tl.load(K + token_base + token * (H * D) + d_index, d_mask, other=0.0).to(
                tl.float32
            )
            v_t = tl.load(V + token_base + token * (H * D) + d_index, d_mask, other=0.0).to(
                tl.float32
            )
            theta_t = tl.load(THETA + gate_base + token * H).to(tl.float32)
            alpha_t = tl.load(ALPHA + gate_base + token * H).to(tl.float32)
            eta_t = tl.load(ETA + gate_base + token * H).to(tl.float32)
            km = tl.sum(update_base * k_t[:, None], axis=0)
            reconstruction_target = v_t - k_t
            mean = tl.sum(km, axis=0) / D
            centered = tl.where(d_mask, km - mean, 0.0)
            var = tl.sum(centered * centered, axis=0) / D
            rstd = tl.sqrt(var + EPS)
            km_hat = centered / rstd
            grad = (w * km_hat + b - reconstruction_target) * w
            v_new = D * grad - tl.sum(grad, axis=0) / (rstd * D)
            v_new = v_new - km_hat * (tl.sum(grad * km_hat, axis=0) / (rstd * D))
            momentum = eta_t * momentum - 2.0 * theta_t * (k_t[:, None] * v_new[None, :])
            memory = (1.0 - alpha_t) * memory + momentum
            output = tl.sum(memory * q_t[:, None], axis=0)
            output_mean = tl.sum(output, axis=0) / D
            output_centered = tl.where(d_mask, output - output_mean, 0.0)
            output_var = tl.sum(output_centered * output_centered, axis=0) / D
            output_rstd = tl.sqrt(output_var + EPS)
            output = output + output_centered / output_rstd * w + b
            tl.store(OUTPUT + token_base + token * (H * D) + d_index, output, d_mask)
            if (token + 1) % CHUNK == 0:
                update_base = memory
        tl.store(FINAL_MEMORY + state_offset, memory, dd_mask)

    @triton.jit
    def regularized_solve_kernel(
        Q,
        K,
        V,
        LOG_DECAY,
        BETA,
        LAMB,
        WORK_A,
        WORK_B,
        OUTPUT,
        FINAL_KK,
        FINAL_KV,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """MesaNet dual covariance-state recurrence with a per-token regularized solve.

        One program owns one (batch, head) pair; the [K, K] states h_kk and h_kv
        stay in registers in fp32. The per-token solve runs as Gaussian
        elimination with partial pivoting, mirroring ``np.linalg.solve(h_kk +
        diag(lamb), q)``. The elimination works on a per-program scratch copy
        (WORK_A/WORK_B): rows are reloaded from the scratch with a dynamic index
        because a 2D->1D register reduction of a ``tl.where``-selected tile
        mis-compiles under Triton 3.8 (it reads the diagonal, not the row).
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        k_mask = k_index < K_DIM
        kk_mask = k_mask[:, None] & k_mask[None, :]
        state_base = (batch * H + head) * K_DIM * K_DIM
        state_offset = state_base + k_index[:, None] * K_DIM + k_index[None, :]
        h_kk = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        h_kv = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        lamb = tl.load(LAMB + head * K_DIM + k_index, k_mask, other=0.0).to(tl.float32)
        token_base = batch * (T * H * K_DIM) + head * K_DIM
        gate_base = batch * (T * H) + head
        work_a = WORK_A + row * K_DIM * K_DIM
        work_b = WORK_B + row * K_DIM
        for token in range(T):
            q_t = tl.load(
                Q + token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            k_t = tl.load(
                K + token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(
                V + token_base + token * (H * K_DIM) + k_index, k_mask, other=0.0
            ).to(tl.float32)
            decay = tl.exp(tl.load(LOG_DECAY + gate_base + token * H).to(tl.float32))
            beta_t = tl.load(BETA + gate_base + token * H).to(tl.float32)
            k_beta = k_t * beta_t
            h_kk = decay * h_kk + k_beta[:, None] * k_t[None, :]
            h_kv = decay * h_kv + k_beta[:, None] * v_t[None, :]
            a = tl.where(k_index[:, None] == k_index[None, :], h_kk + lamb[:, None], h_kk)
            tl.store(work_a + k_index[:, None] * K_DIM + k_index[None, :], a, kk_mask)
            tl.store(work_b + k_index, q_t, k_mask)
            for col in range(K_DIM):
                column = tl.load(work_a + k_index * K_DIM + col, k_mask, other=0.0).to(
                    tl.float32
                )
                below = tl.where(k_index >= col, tl.abs(column), 0.0)
                pivot = tl.argmax(below, axis=0)
                pivot_row = tl.load(work_a + pivot * K_DIM + k_index, k_mask, other=0.0).to(
                    tl.float32
                )
                pivot_rhs = tl.load(work_b + pivot).to(tl.float32)
                col_row = tl.load(work_a + col * K_DIM + k_index, k_mask, other=0.0).to(
                    tl.float32
                )
                col_rhs = tl.load(work_b + col).to(tl.float32)
                tl.store(work_a + col * K_DIM + k_index, pivot_row, k_mask)
                tl.store(work_a + pivot * K_DIM + k_index, col_row, k_mask)
                tl.store(work_b + col, pivot_rhs)
                tl.store(work_b + pivot, col_rhs)
                pivot_val = tl.sum(tl.where(k_index == col, pivot_row, 0.0), axis=0)
                factor = tl.where(k_index > col, column / pivot_val, 0.0)
                new_a = tl.load(
                    work_a + k_index[:, None] * K_DIM + k_index[None, :], kk_mask, other=0.0
                ).to(tl.float32)
                new_a = new_a - factor[:, None] * pivot_row[None, :]
                tl.store(
                    work_a + k_index[:, None] * K_DIM + k_index[None, :], new_a, kk_mask
                )
                new_b = tl.load(work_b + k_index, k_mask, other=0.0).to(tl.float32)
                new_b = new_b - factor * pivot_rhs
                tl.store(work_b + k_index, new_b, k_mask)
            solution = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for step in range(K_DIM):
                i = K_DIM - 1 - step
                row_i = tl.load(work_a + i * K_DIM + k_index, k_mask, other=0.0).to(
                    tl.float32
                )
                diag = tl.sum(tl.where(k_index == i, row_i, 0.0), axis=0)
                tail = tl.sum(tl.where(k_index > i, row_i * solution, 0.0), axis=0)
                rhs_i = tl.load(work_b + i).to(tl.float32)
                value = (rhs_i - tail) / diag
                solution = tl.where(k_index == i, value, solution)
            output = tl.sum(h_kv * solution[:, None], axis=0)
            tl.store(OUTPUT + token_base + token * (H * K_DIM) + k_index, output, k_mask)
        tl.store(FINAL_KK + state_offset, h_kk, kk_mask)
        tl.store(FINAL_KV + state_offset, h_kv, kk_mask)

    @triton.jit
    def second_order_cumsum_kernel(
        Q,
        K,
        V,
        OUTPUT,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """HLA masked second-order causal attention via inclusive prefix sums.

        One program owns one (batch, head) pair and accumulates the prefix-sum
        states S [K, K], C [K, V], G [K, V] in registers in fp32.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        state_s = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
        state_c = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        state_g = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        qk_base = batch * (T * H * K_DIM) + head * K_DIM
        v_base = batch * (T * H * V_DIM) + head * V_DIM
        for token in range(T):
            q_t = tl.load(Q + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            k_t = tl.load(K + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            v_t = tl.load(V + v_base + token * (H * V_DIM) + v_index, v_mask, other=0.0).to(
                tl.float32
            )
            previous_c = state_c
            state_s = state_s + k_t[:, None] * k_t[None, :]
            state_c = state_c + q_t[:, None] * v_t[None, :]
            key_previous_c = tl.sum(previous_c * k_t[:, None], axis=0)
            state_g = state_g + k_t[:, None] * key_previous_c[None, :]
            query_state = tl.sum(state_s * q_t[:, None], axis=0)
            output = tl.sum(state_c * query_state[:, None], axis=0) - tl.sum(
                state_g * q_t[:, None], axis=0
            )
            tl.store(OUTPUT + v_base + token * (H * V_DIM) + v_index, output, v_mask)

    @triton.jit
    def slot_attention_two_stage_kernel(
        Q,
        K,
        V,
        SLOT_WEIGHTS,
        LOG_DECAY,
        SLOT_PROBABILITY,
        OUTPUT,
        FINAL_KEY_STATE,
        FINAL_VALUE_STATE,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        S_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """ABC/GSA two-stage slot-addressed recurrence (group heads pre-expanded).

        One program owns one (batch, query head) pair. Stage 1 accumulates the
        key state [K, S] and the per-token slot scores; a softmax over slots is
        taken per token; stage 2 accumulates the value state [S, V] and reads it
        by the slot probabilities.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        s_index = tl.arange(0, BLOCK_S)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        s_mask = s_index < S_DIM
        v_mask = v_index < V_DIM
        ks_mask = k_mask[:, None] & s_mask[None, :]
        sv_mask = s_mask[:, None] & v_mask[None, :]
        qk_base = batch * (T * H * K_DIM) + head * K_DIM
        sv_in_base = batch * (T * H * S_DIM) + head * S_DIM
        v_base = batch * (T * H * V_DIM) + head * V_DIM
        key_state = tl.zeros((BLOCK_K, BLOCK_S), dtype=tl.float32)
        scale = K_DIM ** -0.5
        for token in range(T):
            decay = tl.exp(
                tl.load(LOG_DECAY + sv_in_base + token * (H * S_DIM) + s_index, s_mask, other=0.0).to(
                    tl.float32
                )
            )
            k_t = tl.load(K + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            sw_t = tl.load(
                SLOT_WEIGHTS + sv_in_base + token * (H * S_DIM) + s_index, s_mask, other=0.0
            ).to(tl.float32)
            q_t = tl.load(Q + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            key_state = key_state * decay[None, :] + k_t[:, None] * sw_t[None, :]
            score = tl.sum(key_state * (q_t * scale)[:, None], axis=0)
            score = tl.where(s_mask, score, float("-inf"))
            max_score = tl.max(score, axis=0)
            exp_score = tl.exp(score - max_score)
            probability = exp_score / tl.sum(exp_score, axis=0)
            tl.store(
                SLOT_PROBABILITY + sv_in_base + token * (H * S_DIM) + s_index,
                probability,
                s_mask,
            )
        value_state = tl.zeros((BLOCK_S, BLOCK_V), dtype=tl.float32)
        for token in range(T):
            decay = tl.exp(
                tl.load(LOG_DECAY + sv_in_base + token * (H * S_DIM) + s_index, s_mask, other=0.0).to(
                    tl.float32
                )
            )
            sw_t = tl.load(
                SLOT_WEIGHTS + sv_in_base + token * (H * S_DIM) + s_index, s_mask, other=0.0
            ).to(tl.float32)
            v_t = tl.load(V + v_base + token * (H * V_DIM) + v_index, v_mask, other=0.0).to(
                tl.float32
            )
            value_state = value_state * decay[:, None] + sw_t[:, None] * v_t[None, :]
            probability = tl.load(
                SLOT_PROBABILITY + sv_in_base + token * (H * S_DIM) + s_index, s_mask, other=0.0
            ).to(tl.float32)
            output = tl.sum(value_state * probability[:, None], axis=0)
            tl.store(OUTPUT + v_base + token * (H * V_DIM) + v_index, output, v_mask)
        key_state_base = (batch * H + head) * K_DIM * S_DIM
        value_state_base = (batch * H + head) * S_DIM * V_DIM
        tl.store(
            FINAL_KEY_STATE + key_state_base + k_index[:, None] * S_DIM + s_index[None, :],
            key_state,
            ks_mask,
        )
        tl.store(
            FINAL_VALUE_STATE + value_state_base + s_index[:, None] * V_DIM + v_index[None, :],
            value_state,
            sv_mask,
        )

    @triton.jit
    def momentum_delta_kernel(
        Q,
        K,
        V,
        P,
        LOG_ALPHA,
        LOG_MU,
        BETA,
        ETA,
        INITIAL_STATE,
        INITIAL_MOMENTUM,
        OUTPUT,
        FINAL_STATE,
        FINAL_MOMENTUM,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Momentum DeltaNet two-matrix-state recurrence (state + momentum).

        One program owns one (batch, head) pair; the [K, V] state and momentum
        stay in registers in fp32.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_base = (batch * H + head) * K_DIM * V_DIM
        state_offset = state_base + k_index[:, None] * V_DIM + v_index[None, :]
        if HAS_INITIAL:
            state = tl.load(INITIAL_STATE + state_offset, kv_mask, other=0.0).to(tl.float32)
            momentum = tl.load(INITIAL_MOMENTUM + state_offset, kv_mask, other=0.0).to(
                tl.float32
            )
        else:
            state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
            momentum = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        qk_base = batch * (T * H * K_DIM) + head * K_DIM
        v_base = batch * (T * H * V_DIM) + head * V_DIM
        gate_base = batch * (T * H) + head
        for token in range(T):
            q_t = tl.load(Q + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            k_t = tl.load(K + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            p_t = tl.load(P + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            v_t = tl.load(V + v_base + token * (H * V_DIM) + v_index, v_mask, other=0.0).to(
                tl.float32
            )
            alpha_t = tl.exp(tl.load(LOG_ALPHA + gate_base + token * H).to(tl.float32))
            mu_t = tl.exp(tl.load(LOG_MU + gate_base + token * H).to(tl.float32))
            beta_t = tl.load(BETA + gate_base + token * H).to(tl.float32)
            eta_t = tl.load(ETA + gate_base + token * H).to(tl.float32)
            prediction = tl.sum(state * p_t[:, None], axis=0)
            residual = v_t - prediction
            momentum = mu_t * momentum - (eta_t * k_t)[:, None] * residual[None, :]
            state = alpha_t * state - beta_t * momentum
            output = tl.sum(state * (q_t * SCALE)[:, None], axis=0)
            tl.store(OUTPUT + v_base + token * (H * V_DIM) + v_index, output, v_mask)
        tl.store(FINAL_STATE + state_offset, state, kv_mask)
        tl.store(FINAL_MOMENTUM + state_offset, momentum, kv_mask)

    @triton.jit
    def gated_oja_kernel(
        Q,
        K,
        V,
        GATE,
        BETA,
        INITIAL,
        OUTPUT,
        FINAL_STATE,
        H: tl.constexpr,
        T: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Gated Oja value-channel recurrence (a transposed Hebbian/delta rule).

        One program owns one (batch, head) pair; the [K, V] state stays in
        registers in fp32.
        """
        row = tl.program_id(0)
        batch = row // H
        head = row % H
        k_index = tl.arange(0, BLOCK_K)
        v_index = tl.arange(0, BLOCK_V)
        k_mask = k_index < K_DIM
        v_mask = v_index < V_DIM
        kv_mask = k_mask[:, None] & v_mask[None, :]
        state_base = (batch * H + head) * K_DIM * V_DIM
        state_offset = state_base + k_index[:, None] * V_DIM + v_index[None, :]
        if HAS_INITIAL:
            state = tl.load(INITIAL + state_offset, kv_mask, other=0.0).to(tl.float32)
        else:
            state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        qk_base = batch * (T * H * K_DIM) + head * K_DIM
        v_base = batch * (T * H * V_DIM) + head * V_DIM
        gate_base = batch * (T * H) + head
        for token in range(T):
            q_t = tl.load(Q + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            k_t = tl.load(K + qk_base + token * (H * K_DIM) + k_index, k_mask, other=0.0).to(
                tl.float32
            )
            v_t = tl.load(V + v_base + token * (H * V_DIM) + v_index, v_mask, other=0.0).to(
                tl.float32
            )
            gate_t = tl.load(GATE + v_base + token * (H * V_DIM) + v_index, v_mask, other=0.0).to(
                tl.float32
            )
            beta_t = tl.load(BETA + gate_base + token * H).to(tl.float32)
            state = state * tl.exp(gate_t)[None, :]
            prediction = tl.sum(state * v_t[None, :], axis=1)
            correction = beta_t * (k_t - prediction)
            state = state + correction[:, None] * v_t[None, :]
            output = tl.sum(state * (q_t * SCALE)[:, None], axis=0)
            tl.store(OUTPUT + v_base + token * (H * V_DIM) + v_index, output, v_mask)
        tl.store(FINAL_STATE + state_offset, state, kv_mask)

    return (
        triton,
        layernorm_inner_state_kernel,
        momentum_inner_state_kernel,
        regularized_solve_kernel,
        second_order_cumsum_kernel,
        slot_attention_two_stage_kernel,
        momentum_delta_kernel,
        gated_oja_kernel,
    )


def _check_cuda(name: str, *tensors: Any) -> None:
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError(f"native {name} requires CUDA tensors")


def _check_float32(name: str, *tensors: Any) -> None:
    import torch

    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise ValueError(f"native {name} requires float32 tensors")


def execute_layernorm_inner_state(
    *,
    query: Any,
    key: Any,
    value: Any,
    w: Any,
    b: Any,
    eta: Any,
    initial_state: Any | None = None,
    initial_state_bias: Any | None = None,
    chunk_size: int = 16,
    eps: float = 1e-6,
) -> tuple[Any, tuple[Any, Any]]:
    """TTT-Linear chunkwise inner-loss update; mirrors ``layernorm_inner_state``.

    query/key/value are [B, T, H, D]; w/b are [H, D]; eta is [B, T, H] or
    [B, T, H, 1]. Returns (output [B, T, H, D], (memory [B, H, D, D],
    memory_bias [B, H, D])) in fp32.
    """
    import torch

    (
        triton,
        layernorm_inner_state_kernel,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _kernels()
    _check_cuda("layernorm_inner_state", query, key, value, w, b, eta)
    _check_float32("layernorm_inner_state", query, key, value, w, b, eta)
    batch, sequence, heads, dim = query.shape
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("layernorm_inner_state query/key/value shapes must match")
    if w.shape != (heads, dim) or b.shape != (heads, dim):
        raise ValueError("layernorm_inner_state w/b must be [H, D]")
    if eta.shape == (batch, sequence, heads, 1):
        eta = eta.squeeze(-1)
    if eta.shape != (batch, sequence, heads):
        raise ValueError("layernorm_inner_state eta must be [B,T,H] or [B,T,H,1]")
    has_initial = initial_state is not None
    if has_initial:
        if tuple(initial_state.shape) != (batch, heads, dim, dim):
            raise ValueError("initial_state must be [B,H,D,D]")
        if initial_state_bias is None or tuple(initial_state_bias.shape) not in (
            (batch, heads, 1, dim),
            (batch, heads, dim),
        ):
            raise ValueError("initial_state_bias must be [B,H,1,D] or [B,H,D]")
        memory_init = initial_state.contiguous()
        bias_init = initial_state_bias.reshape(batch, heads, dim).contiguous()
        _check_cuda("layernorm_inner_state", memory_init, bias_init)
        _check_float32("layernorm_inner_state", memory_init, bias_init)
    else:
        memory_init = query
        bias_init = query
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    eta = eta.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    output = torch.empty_like(query)
    final_memory = torch.empty((batch, heads, dim, dim), device=query.device, dtype=torch.float32)
    final_bias = torch.empty((batch, heads, dim), device=query.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(dim)
    block_t = triton.next_power_of_2(min(chunk_size, sequence))
    layernorm_inner_state_kernel[(batch * heads,)](
        query,
        key,
        value,
        w,
        b,
        eta,
        memory_init,
        bias_init,
        output,
        final_memory,
        final_bias,
        heads,
        sequence,
        dim,
        chunk_size,
        eps,
        has_initial,
        block_t,
        block_d,
        num_warps=4,
    )
    return output, (final_memory, final_bias)


def execute_momentum_inner_state(
    *,
    query: Any,
    key: Any,
    value: Any,
    w: Any,
    b: Any,
    theta: Any,
    alpha: Any,
    eta: Any,
    initial_state: Any | None = None,
    chunk_size: int = 16,
    eps: float = 1e-6,
) -> tuple[Any, Any]:
    """Titans tokenwise memory with momentum; mirrors ``momentum_inner_state``.

    query/key/value are [B, T, H, D]; w/b are [H, D]; theta/alpha/eta are
    [B, T, H] or [B, T, H, 1]. Returns (output [B, T, H, D], memory
    [B, H, D, D]) in fp32.
    """
    import torch

    (
        triton,
        _,
        momentum_inner_state_kernel,
        _,
        _,
        _,
        _,
        _,
    ) = _kernels()
    _check_cuda("momentum_inner_state", query, key, value, w, b, theta, alpha, eta)
    _check_float32("momentum_inner_state", query, key, value, w, b, theta, alpha, eta)
    batch, sequence, heads, dim = query.shape
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("momentum_inner_state query/key/value shapes must match")
    if w.shape != (heads, dim) or b.shape != (heads, dim):
        raise ValueError("momentum_inner_state w/b must be [H, D]")

    def _gate(tensor, name):
        if tensor.shape == (batch, sequence, heads, 1):
            tensor = tensor.squeeze(-1)
        if tensor.shape != (batch, sequence, heads):
            raise ValueError(f"momentum_inner_state {name} must be [B,T,H] or [B,T,H,1]")
        return tensor.contiguous()

    theta = _gate(theta, "theta")
    alpha = _gate(alpha, "alpha")
    eta = _gate(eta, "eta")
    has_initial = initial_state is not None
    if has_initial:
        if tuple(initial_state.shape) != (batch, heads, dim, dim):
            raise ValueError("initial_state must be [B,H,D,D]")
        initial = initial_state.contiguous()
        _check_cuda("momentum_inner_state", initial)
        _check_float32("momentum_inner_state", initial)
    else:
        initial = query
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    output = torch.empty_like(query)
    final_memory = torch.empty((batch, heads, dim, dim), device=query.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(dim)
    momentum_inner_state_kernel[(batch * heads,)](
        query,
        key,
        value,
        w,
        b,
        theta,
        alpha,
        eta,
        initial,
        output,
        final_memory,
        heads,
        sequence,
        dim,
        chunk_size,
        eps,
        has_initial,
        block_d,
        num_warps=4,
    )
    return output, final_memory


def execute_regularized_solve(
    *,
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    beta: Any,
    lamb: Any,
) -> tuple[Any, tuple[Any, Any]]:
    """MesaNet per-token regularized solve; mirrors ``regularized_solve``.

    query/key/value are [B, T, H, K]; log_decay/beta are [B, T, H]; lamb is
    [H, K]. Returns (output [B, T, H, K], (h_kk [B, H, K, K],
    h_kv [B, H, K, K])) in fp32.
    """
    import torch

    (
        triton,
        _,
        _,
        regularized_solve_kernel,
        _,
        _,
        _,
        _,
    ) = _kernels()
    _check_cuda("regularized_solve", query, key, value, log_decay, beta, lamb)
    _check_float32("regularized_solve", query, key, value, log_decay, beta, lamb)
    batch, sequence, heads, key_dim = query.shape
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("regularized_solve query/key/value shapes must match")
    if log_decay.shape != (batch, sequence, heads) or beta.shape != (batch, sequence, heads):
        raise ValueError("regularized_solve log_decay/beta must be [B,T,H]")
    if lamb.shape != (heads, key_dim):
        raise ValueError("regularized_solve lamb must be [H,K]")
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    log_decay = log_decay.contiguous()
    beta = beta.contiguous()
    lamb = lamb.contiguous()
    output = torch.empty_like(query)
    final_kk = torch.empty(
        (batch, heads, key_dim, key_dim), device=query.device, dtype=torch.float32
    )
    final_kv = torch.empty_like(final_kk)
    work_a = torch.empty(
        (batch * heads, key_dim, key_dim), device=query.device, dtype=torch.float32
    )
    work_b = torch.empty((batch * heads, key_dim), device=query.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(key_dim)
    regularized_solve_kernel[(batch * heads,)](
        query,
        key,
        value,
        log_decay,
        beta,
        lamb,
        work_a,
        work_b,
        output,
        final_kk,
        final_kv,
        heads,
        sequence,
        key_dim,
        block_k,
        num_warps=4,
    )
    return output, (final_kk, final_kv)


def execute_second_order_cumsum(*, query: Any, key: Any, value: Any) -> tuple[Any, None]:
    """HLA masked second-order causal attention; mirrors ``second_order_cumsum``.

    query/key are [B, T, H, K]; value is [B, T, H, V]. Returns
    (output [B, T, H, V], None) in fp32.
    """
    import torch

    (
        triton,
        _,
        _,
        _,
        second_order_cumsum_kernel,
        _,
        _,
        _,
    ) = _kernels()
    _check_cuda("second_order_cumsum", query, key, value)
    _check_float32("second_order_cumsum", query, key, value)
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if key.shape != query.shape:
        raise ValueError("second_order_cumsum query/key shapes must match")
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("second_order_cumsum value must be [B,T,H,V]")
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(value)
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    second_order_cumsum_kernel[(batch * heads,)](
        query,
        key,
        value,
        output,
        heads,
        sequence,
        key_dim,
        value_dim,
        block_k,
        block_v,
        num_warps=4,
    )
    return output, None


def _causal_fft_convolution(x: Any, kernel: Any, direct: Any) -> Any:
    """Causal linear convolution via FFT with a pointwise direct term.

    ``x`` is [B, T] (per channel), ``kernel`` is the [L] filter, and ``direct``
    is a scalar direct/skip coefficient. Mirrors the NumPy ``fft_convolution``.
    """
    import torch

    length = x.shape[-1]
    fft_size = kernel.shape[-1] + length
    spectrum = torch.fft.rfft(kernel, fft_size) * torch.fft.rfft(x, fft_size)
    out = torch.fft.irfft(spectrum, fft_size)[..., :length]
    return out + direct * x


def execute_fft_convolution(*, query: Any, kernel: Any, direct: Any) -> tuple[Any, None]:
    """Hyena single implicit-filter causal FFT convolution (torch-native).

    Torch-native executor (``torch.fft``): a Triton FFT is impractical, and a
    torch-native FFT convolution is genuinely native - no upstream library
    dispatch. Mirrors ``hyena_fft_convolution``: query is [B, T, C], kernel is
    [C, T], direct is [C]. Returns (output [B, T, C], None).
    """
    import torch

    _check_cuda("fft_convolution", query, kernel, direct)
    _check_float32("fft_convolution", query, kernel, direct)
    batch, sequence, channels = query.shape
    if kernel.shape != (channels, sequence):
        raise ValueError("fft_convolution kernel must be [C,T]")
    if direct.shape != (channels,):
        raise ValueError("fft_convolution direct must be [C]")
    x = query.transpose(1, 2)
    fft_size = 2 * sequence
    kernel_spectrum = torch.fft.rfft(kernel, n=fft_size) / fft_size
    input_spectrum = torch.fft.rfft(x, n=fft_size)
    out = torch.fft.irfft(
        input_spectrum * kernel_spectrum, n=fft_size, norm="forward"
    )[..., :sequence]
    out = out + x * direct[None, :, None]
    return out.transpose(1, 2).contiguous(), None


def execute_two_stage_fft_convolution(
    *,
    query: Any,
    key: Any,
    value: Any,
    ssm_kernel: Any,
    ssm_k_kernel: Any,
    ssm_k_direct: Any,
    skip: Any,
) -> tuple[Any, None]:
    """H3 two-stage causal FFT convolution (torch-native).

    Torch-native executor (``torch.fft``), mirroring
    ``two_stage_fft_convolution``: all of query/key/value are [B, T, H, 1]; the
    kernels are [H, L] conv filters and the ``*_direct``/``skip`` terms are
    [H] pointwise. Returns (output [B, T, H, 1], None).
    """
    _check_cuda(
        "two_stage_fft_convolution", query, key, value, ssm_kernel, ssm_k_kernel, ssm_k_direct, skip
    )
    _check_float32(
        "two_stage_fft_convolution", query, key, value, ssm_kernel, ssm_k_kernel, ssm_k_direct, skip
    )
    batch, sequence, heads, width = query.shape
    if width != 1 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("two_stage_fft_convolution query/key/value must be [B,T,H,1]")
    if ssm_kernel.shape != (heads, sequence) or ssm_k_kernel.shape != (heads, sequence):
        raise ValueError("two_stage_fft_convolution kernels must be [H,T]")
    if ssm_k_direct.shape != (heads,) or skip.shape != (heads,):
        raise ValueError("two_stage_fft_convolution ssm_k_direct/skip must be [H]")
    key_bht = key[..., 0].transpose(1, 2)
    value_bht = value[..., 0].transpose(1, 2)
    query_bht = query[..., 0].transpose(1, 2)
    shifted_key = _causal_fft_convolution(key_bht, ssm_k_kernel, ssm_k_direct[:, None])
    read = _causal_fft_convolution(shifted_key * value_bht, ssm_kernel, skip[:, None])
    output = (read * query_bht).transpose(1, 2)
    return output.unsqueeze(-1).contiguous(), None


def execute_slot_attention_two_stage(
    *,
    query: Any,
    key: Any,
    value: Any,
    slot_weights: Any,
    log_decay: Any,
    group_size: int | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """ABC/GSA two-stage slot-addressed recurrence; mirrors ``slot_attention_two_stage``.

    query is [B, T, Hq, K]; key/slot_weights/log_decay are [B, T, Hk, *]; value
    is [B, T, Hk, V]. Group heads are expanded to the query-head granularity
    (``group_size = Hq // Hk`` when not given). Returns (output [B, T, Hq, V],
    (key_state [B, Hk, K, S], value_state [B, Hk, S, V])) in fp32, folded back
    to the key-head granularity like the canonical executor.
    """
    import torch

    (
        triton,
        _,
        _,
        _,
        _,
        slot_attention_two_stage_kernel,
        _,
        _,
    ) = _kernels()
    _check_cuda("slot_attention_two_stage", query, key, value, slot_weights, log_decay)
    _check_float32("slot_attention_two_stage", query, key, value, slot_weights, log_decay)
    batch, sequence, query_heads, key_dim = query.shape
    key_heads = key.shape[2]
    slots = slot_weights.shape[-1]
    value_dim = value.shape[-1]
    if group_size is None:
        group_size = query_heads // key_heads
    if query_heads != key_heads * group_size:
        raise ValueError("query heads must equal key heads times group_size")
    if key.shape[:3] != (batch, sequence, key_heads) or key.shape[-1] != key_dim:
        raise ValueError("slot_attention_two_stage key must be [B,T,Hk,K]")
    if value.shape[:3] != (batch, sequence, key_heads):
        raise ValueError("slot_attention_two_stage value must be [B,T,Hk,V]")
    if slot_weights.shape[:3] != (batch, sequence, key_heads):
        raise ValueError("slot_attention_two_stage slot_weights must be [B,T,Hk,S]")
    if log_decay.shape != slot_weights.shape:
        raise ValueError("slot_attention_two_stage log_decay must match slot_weights")

    def _expand_heads(tensor):
        return (
            tensor.unsqueeze(3)
            .expand(batch, sequence, key_heads, group_size, *tensor.shape[3:])
            .reshape(batch, sequence, query_heads, *tensor.shape[3:])
            .contiguous()
        )

    key_e = _expand_heads(key)
    value_e = _expand_heads(value)
    slot_weights_e = _expand_heads(slot_weights)
    log_decay_e = _expand_heads(log_decay)
    query = query.contiguous()
    slot_scores = torch.empty(
        (batch, sequence, query_heads, slots), device=query.device, dtype=torch.float32
    )
    output = torch.empty(
        (batch, sequence, query_heads, value_dim), device=query.device, dtype=torch.float32
    )
    final_key_state = torch.empty(
        (batch, query_heads, key_dim, slots), device=query.device, dtype=torch.float32
    )
    final_value_state = torch.empty(
        (batch, query_heads, slots, value_dim), device=query.device, dtype=torch.float32
    )
    block_k = triton.next_power_of_2(key_dim)
    block_s = triton.next_power_of_2(slots)
    block_v = triton.next_power_of_2(value_dim)
    slot_attention_two_stage_kernel[(batch * query_heads,)](
        query,
        key_e,
        value_e,
        slot_weights_e,
        log_decay_e,
        slot_scores,
        output,
        final_key_state,
        final_value_state,
        query_heads,
        sequence,
        key_dim,
        slots,
        value_dim,
        block_k,
        block_s,
        block_v,
        num_warps=4,
    )
    final_key_state = final_key_state.reshape(batch, key_heads, group_size, key_dim, slots)[:, :, 0]
    final_value_state = final_value_state.reshape(
        batch, key_heads, group_size, slots, value_dim
    )[:, :, 0]
    return output, (final_key_state.contiguous(), final_value_state.contiguous())


def execute_momentum_delta(
    *,
    query: Any,
    key: Any,
    value: Any,
    p: Any,
    log_alpha: Any,
    log_mu: Any,
    beta: Any,
    eta: Any,
    initial_state: Any | None = None,
    initial_momentum: Any | None = None,
    scale: float | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Momentum DeltaNet two-matrix-state recurrence; mirrors ``momentum_delta``.

    query/key/p are [B, T, H, K]; value is [B, T, H, V]; log_alpha/log_mu/
    beta/eta are [B, T, H] per-head schedules. Returns (output [B, T, H, V],
    (state [B, H, K, V], momentum [B, H, K, V])) in fp32.
    """
    import torch

    (
        triton,
        _,
        _,
        _,
        _,
        _,
        momentum_delta_kernel,
        _,
    ) = _kernels()
    _check_cuda("momentum_delta", query, key, value, p, log_alpha, log_mu, beta, eta)
    _check_float32("momentum_delta", query, key, value, p, log_alpha, log_mu, beta, eta)
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if key.shape != query.shape or p.shape != query.shape:
        raise ValueError("momentum_delta query/key/p shapes must match")
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("momentum_delta value must be [B,T,H,V]")
    for name, gate in (("log_alpha", log_alpha), ("log_mu", log_mu), ("beta", beta), ("eta", eta)):
        if gate.shape != (batch, sequence, heads):
            raise ValueError(f"momentum_delta {name} must be [B,T,H]")
    if scale is None:
        scale = key_dim ** -0.5
    has_initial = initial_state is not None
    if has_initial:
        if tuple(initial_state.shape) != (batch, heads, key_dim, value_dim):
            raise ValueError("initial_state must be [B,H,K,V]")
        if initial_momentum is None or tuple(initial_momentum.shape) != (
            batch,
            heads,
            key_dim,
            value_dim,
        ):
            raise ValueError("initial_momentum must be [B,H,K,V]")
        state_init = initial_state.contiguous()
        momentum_init = initial_momentum.contiguous()
        _check_cuda("momentum_delta", state_init, momentum_init)
        _check_float32("momentum_delta", state_init, momentum_init)
    else:
        state_init = query
        momentum_init = query
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    p = p.contiguous()
    log_alpha = log_alpha.contiguous()
    log_mu = log_mu.contiguous()
    beta = beta.contiguous()
    eta = eta.contiguous()
    output = torch.empty_like(value)
    final_state = torch.empty(
        (batch, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
    )
    final_momentum = torch.empty_like(final_state)
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    momentum_delta_kernel[(batch * heads,)](
        query,
        key,
        value,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        state_init,
        momentum_init,
        output,
        final_state,
        final_momentum,
        heads,
        sequence,
        key_dim,
        value_dim,
        scale,
        has_initial,
        block_k,
        block_v,
        num_warps=4,
    )
    return output, (final_state, final_momentum)


def execute_gated_oja(
    *,
    query: Any,
    key: Any,
    value: Any,
    gate: Any,
    beta: Any,
    initial_state: Any | None = None,
    scale: float | None = None,
) -> tuple[Any, Any]:
    """Gated Oja value-channel recurrence; mirrors ``gated_oja``.

    query/key are [B, T, H, K]; value/gate are [B, T, H, V]; beta is
    [B, T, H]. Returns (output [B, T, H, V], state [B, H, K, V]) in fp32.
    """
    import torch

    (
        triton,
        _,
        _,
        _,
        _,
        _,
        _,
        gated_oja_kernel,
    ) = _kernels()
    _check_cuda("gated_oja", query, key, value, gate, beta)
    _check_float32("gated_oja", query, key, value, gate, beta)
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if key.shape != query.shape:
        raise ValueError("gated_oja query/key shapes must match")
    if value.shape[:3] != (batch, sequence, heads) or gate.shape != value.shape:
        raise ValueError("gated_oja value/gate must be [B,T,H,V]")
    if beta.shape != (batch, sequence, heads):
        raise ValueError("gated_oja beta must be [B,T,H]")
    if scale is None:
        scale = key_dim ** -0.5
    has_initial = initial_state is not None
    if has_initial:
        if tuple(initial_state.shape) != (batch, heads, key_dim, value_dim):
            raise ValueError("initial_state must be [B,H,K,V]")
        initial = initial_state.contiguous()
        _check_cuda("gated_oja", initial)
        _check_float32("gated_oja", initial)
    else:
        initial = query
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    output = torch.empty_like(value)
    final_state = torch.empty(
        (batch, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = triton.next_power_of_2(value_dim)
    gated_oja_kernel[(batch * heads,)](
        query,
        key,
        value,
        gate,
        beta,
        initial,
        output,
        final_state,
        heads,
        sequence,
        key_dim,
        value_dim,
        scale,
        has_initial,
        block_k,
        block_v,
        num_warps=4,
    )
    return output, final_state


__all__ = [
    "execute_layernorm_inner_state",
    "execute_momentum_inner_state",
    "execute_regularized_solve",
    "execute_second_order_cumsum",
    "execute_fft_convolution",
    "execute_two_stage_fft_convolution",
    "execute_slot_attention_two_stage",
    "execute_momentum_delta",
    "execute_gated_oja",
]
