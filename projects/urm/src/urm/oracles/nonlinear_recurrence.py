"""Float64 canonical executors for the distinct K2 recurrence operators.

These are the canonical NumPy execution paths for the recurrence equations the
``recurrence_operator`` IR field distinguishes (the former additive/no-decay
collision group). Each operator is a genuinely distinct equation - nonlinear
gated RNNs, inner-loss gradient states, trapezoidal SSMs, regularized solves,
second-order cumsums, and FFT long convolutions - so each gets its own canonical
executor here rather than being silently folded into the plain linear recurrence.

These are oracles for representation and composition checks, never performance
backends. The nonlinear recurrences are sequential-only (they do not admit an
efficient parallel scan); the FFT convolutions and the second-order cumsum are
linear and parallelizable.
"""

from __future__ import annotations

import numpy as np


def tanh_rnn(query, weight, initial_state):
    """h_t = tanh(h_{t-1} W + x_t); the output is the state. query [B,T,N,H]."""
    q = np.asarray(query, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    batch, sequence = q.shape[:2]
    outputs = []
    for t in range(sequence):
        state = np.tanh(np.einsum("bnh,nhk->bnk", state, w) + q[:, t])
        outputs.append(state)
    return np.stack(outputs, axis=1), state


def gated_rnn(query, weight, forget_input, forget_weight, reset_input,
              reset_weight, initial_state):
    """GRU: h_t = z⊙h + (1-z)⊙tanh((r⊙h)W + x). query/forget/reset [B,T,N,H]."""
    q = np.asarray(query, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    fi = np.asarray(forget_input, dtype=np.float64)
    fw = np.asarray(forget_weight, dtype=np.float64)
    ri = np.asarray(reset_input, dtype=np.float64)
    rw = np.asarray(reset_weight, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    outputs = []
    for t in range(q.shape[1]):
        forget = sigmoid(np.einsum("bnh,nhk->bnk", state, fw) + fi[:, t])
        reset = sigmoid(np.einsum("bnh,nhk->bnk", state, rw) + ri[:, t])
        candidate = np.tanh(np.einsum("bnh,nhk->bnk", state * reset, w) + q[:, t])
        state = forget * state + (1.0 - forget) * candidate
        outputs.append(state)
    return np.stack(outputs, axis=1), state


def multiplicative_rnn(query, key, value, weight, forget_input, initial_state):
    """Second-order matrix memory: S_t = f⊙S + (1-f)⊙tanh(S W + k v^T); read q^T S."""
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    fi = np.asarray(forget_input, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    outputs = []
    for t in range(q.shape[1]):
        update = k[:, t][..., None] * v[:, t][..., None, :]
        candidate = np.tanh(np.einsum("bnkv,nvw->bnkw", state, w) + update)
        forget = fi[:, t].reshape(fi.shape[0], fi.shape[2], 1, 1)
        state = forget * state + (1.0 - forget) * candidate
        outputs.append(np.einsum("bnk,bnkv->bnv", q[:, t], state))
    return np.stack(outputs, axis=1), state


def fft_convolution(x, kernel, direct):
    """Causal linear convolution via FFT with a pointwise direct term.

    ``x`` is ``[B, T]`` (per channel), ``kernel`` is the ``[L]`` filter, and
    ``direct`` is a scalar direct/skip coefficient.
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    t = x.shape[-1]
    fft_size = kernel.shape[-1] + t
    spectrum = np.fft.rfft(kernel, fft_size) * np.fft.rfft(x, fft_size)
    out = np.fft.irfft(spectrum, fft_size)[..., :t]
    return out + direct * x


def two_stage_fft_convolution(query, key, value, ssm_kernel, ssm_k_kernel,
                              ssm_k_direct, skip):
    """H3 two-stage causal FFT convolution: conv(K).V -> conv -> .Q.

    All operands are ``[B, T, H, 1]``; the kernels are conv filters and the
    ``*_direct``/``skip`` terms are pointwise.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    batch, sequence, heads = q.shape[:3]
    output = np.empty_like(q)
    for b in range(batch):
        for h in range(heads):
            kb = k[b, :, h, 0]
            vb = v[b, :, h, 0]
            qb = q[b, :, h, 0]
            shifted_key = fft_convolution(kb, ssm_k_kernel[h], ssm_k_direct[h])
            read = fft_convolution(shifted_key * vb, ssm_kernel[h], skip[h])
            output[b, :, h, 0] = read * qb
    return output, None


def hyena_fft_convolution(query, kernel, direct):
    """Hyena single implicit-filter causal FFT convolution with a direct term.

    ``query`` is ``[B, T, C]``, ``kernel`` is ``[C, T]``, ``direct`` is ``[C]``.
    """
    q = np.asarray(query, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    direct = np.asarray(direct, dtype=np.float64)
    batch, sequence, channels = q.shape
    x = q.transpose(0, 2, 1)  # [B, C, T]
    fft_size = 2 * sequence
    kernel_spectrum = np.fft.rfft(kernel, n=fft_size) / fft_size
    input_spectrum = np.fft.rfft(x, n=fft_size)
    # norm="forward" matches the reference: the inverse transform is unnormalized.
    out = np.fft.irfft(
        input_spectrum * kernel_spectrum, n=fft_size, norm="forward"
    )[..., :sequence]
    out = out + x * direct[None, :, None]
    return out.transpose(0, 2, 1), None


def layernorm_inner_state(query, key, value, w, b, eta, initial_state=None,
                          initial_state_bias=None, chunk_size=16, eps=1e-6):
    """TTT-Linear chunkwise inner-loss update with matrix + bias memory states.

    Per chunk: a LayerNorm-inner-loss gradient step updates the memory matrix and
    bias; the output is the chunk readout post-processed by a second LayerNorm
    with affine. query/key/value [B,T,H,D] (K=V); w/b [H,D]; eta [B,T,H,1].
    Returns (output, (memory, memory_bias)).
    """
    q0 = np.asarray(query, dtype=np.float64)
    k0 = np.asarray(key, dtype=np.float64)
    v0 = np.asarray(value, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    eta0 = np.asarray(eta, dtype=np.float64)
    batch, sequence, heads, dim = q0.shape
    memory = (
        np.zeros((batch, heads, dim, dim))
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    memory_bias = (
        np.zeros((batch, heads, 1, dim))
        if initial_state_bias is None
        else np.asarray(initial_state_bias, dtype=np.float64).copy()
    )
    # BTHD -> BHTD; q pre-scaled by dim**-0.5.
    q = q0.transpose(0, 2, 1, 3) * (dim ** -0.5)
    k = k0.transpose(0, 2, 1, 3)
    v = v0.transpose(0, 2, 1, 3)
    eta = eta0.transpose(0, 2, 1, 3)
    w = w.reshape(heads, 1, dim)
    b = b.reshape(heads, 1, dim)
    outputs = []
    for start in range(0, sequence, chunk_size):
        stop = start + chunk_size
        qc, kc, vc, ec = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], eta[:, :, start:stop]
        kh = kc @ memory + memory_bias
        target = vc - kc
        mean = kh.mean(axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(kh.var(axis=-1, keepdims=True) + eps)
        kh_hat = (kh - mean) * rstd
        grad = (w * kh_hat + b - target) * w
        grad = (
            dim * grad
            - grad.sum(axis=-1, keepdims=True)
            - kh_hat * (grad * kh_hat).sum(axis=-1, keepdims=True)
        ) * rstd / dim
        attention = np.tril(qc @ kc.transpose(0, 1, 3, 2))
        output_chunk = (
            qc @ memory
            - (ec * attention) @ grad
            + memory_bias
            - np.tril(np.broadcast_to(ec, attention.shape)) @ grad
        )
        eta_last = ec[:, :, -1, :, None]
        memory = memory - (eta_last * kc).transpose(0, 1, 3, 2) @ grad
        memory_bias = memory_bias - (eta_last * grad).sum(axis=-2, keepdims=True)
        out_mean = output_chunk.mean(axis=-1, keepdims=True)
        out_rstd = 1.0 / np.sqrt(output_chunk.var(axis=-1, keepdims=True) + eps)
        outputs.append(output_chunk + (output_chunk - out_mean) * out_rstd * w + b)
    output = np.concatenate(outputs, axis=-2).transpose(0, 2, 1, 3)
    return output, (memory, memory_bias)


def momentum_inner_state(query, key, value, w, b, theta, alpha, eta,
                         initial_state=None, chunk_size=16, eps=1e-6):
    """Titans tokenwise memory with a chunk-frozen learning target and momentum.

    Per token: read the inner target from the chunk-frozen ``update_base``; take a
    momentum-SGD step on the LayerNorm inner loss; decay the memory by
    ``(1-alpha)`` and add the momentum. The output is ``q @ memory`` post-processed
    by a LayerNorm-with-affine. query/key/value [B,T,H,D]; w/b [H,D];
    theta/alpha/eta [B,T,H,1]. Returns (output, memory).
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    batch, sequence, heads, dim = q.shape
    memory = (
        np.zeros((batch, heads, dim, dim))
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    momentum = np.zeros_like(memory)
    update_base = memory.copy()
    outputs = []
    for token in range(sequence):
        # Per-token [B,H,D] views.
        q_t = q[:, token]  # [B,H,D]
        k_t = k[:, token]  # [B,H,D]
        v_t = v[:, token]  # [B,H,D]
        km = np.einsum("bhd,bhde->bhe", k_t, update_base)  # [B,H,D]
        reconstruction_target = v_t - k_t
        mean = km.mean(axis=-1, keepdims=True)
        rstd = np.sqrt(km.var(axis=-1, keepdims=True) + eps)
        km_hat = (km - mean) / rstd
        grad = (w[None] * km_hat + b[None] - reconstruction_target) * w[None]
        v_new = dim * grad - grad.sum(axis=-1, keepdims=True) / (rstd * dim)
        v_new = v_new - km_hat * (grad * km_hat).sum(axis=-1, keepdims=True) / (rstd * dim)
        # Gates are [B,T,H,1]; take the token -> [B,H,1], then append a trailing
        # axis to get [B,H,1,1] so they broadcast over the state's [D,D] axes.
        theta_t = theta[:, token][..., None]  # [B,H,1,1]
        alpha_t = alpha[:, token][..., None]  # [B,H,1,1]
        eta_t = eta[:, token][..., None]      # [B,H,1,1]
        momentum = eta_t * momentum - 2.0 * theta_t * (k_t[..., None] * v_new[..., None, :])
        memory = (1.0 - alpha_t[..., :1]) * memory + momentum
        output = np.einsum("bhd,bhde->bhe", q_t, memory)  # [B,H,D]
        output_mean = output.mean(axis=-1, keepdims=True)
        output_rstd = np.sqrt(output.var(axis=-1, keepdims=True) + eps)
        output = (
            output
            + (output - output_mean) / output_rstd * w[None]
            + b[None]
        )
        outputs.append(output)  # [B,H,D]
        if (token + 1) % chunk_size == 0:
            update_base = memory.copy()
    # outputs are [B,H,D] per token; stack over time to [B,T,H,D].
    return np.stack(outputs, axis=1), memory


def trapezoidal_ssm(query, key, value, adt, dt, trap, query_bias, key_bias,
                    angles, d_skip=None, gate=None, initial_states=None):
    """Mamba-3 SISO rotary angle accumulator with a trapezoidal four-state SSM.

    State: (angle_state, ssm_state, key_state, value_state). Per token the rotary
    angle integrates tanh(angles)*pi*dt (wrapped to [0,2pi)), Q/K are rotated, and
    the SSM state takes a trapezoidal (previous + current K/V) update with
    continuous-time decay exp(adt). query/key [B,T,Hq,K] (K even); value
    [B,T,Hv,V]; adt/dt/trap [B,Hv,T]; query_bias/key_bias [Hv,K]; angles
    [B,T,Hv,A]. Returns (output, (angle, ssm, key, value)).
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    adt = np.asarray(adt, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    trap = np.asarray(trap, dtype=np.float64)
    query_bias = np.asarray(query_bias, dtype=np.float64)
    key_bias = np.asarray(key_bias, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    batch, sequence, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    angle_dim = angles.shape[-1]
    if query_heads != value_heads:
        repeat = value_heads // query_heads
        q = np.repeat(q, repeat, axis=2)
        k = np.repeat(k, repeat, axis=2)
    if initial_states is None:
        angle_state = np.zeros((batch, value_heads, angle_dim))
        ssm_state = np.zeros((batch, value_heads, value_dim, key_dim))
        key_state = np.zeros((batch, value_heads, key_dim))
        value_state = np.zeros((batch, value_heads, value_dim))
    else:
        angle_state, ssm_state, key_state, value_state = (
            np.asarray(x, dtype=np.float64).copy() for x in initial_states
        )

    def rotary(tensor, cosine, sine):
        paired = tensor.reshape(batch, value_heads, key_dim // 2, 2)
        first, second = paired[..., 0], paired[..., 1]
        if cosine.shape[-1] < key_dim // 2:
            pad = key_dim // 2 - cosine.shape[-1]
            cosine = np.pad(cosine, ((0, 0),) * (cosine.ndim - 1) + ((0, pad),), constant_values=1.0)
            sine = np.pad(sine, ((0, 0),) * (sine.ndim - 1) + ((0, pad),), constant_values=0.0)
        return np.stack(
            (first * cosine - second * sine, first * sine + second * cosine), axis=-1
        ).reshape(batch, value_heads, key_dim)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    outputs = []
    for token in range(sequence):
        angle_state = angle_state + (
            np.tanh(angles[:, token]) * np.pi
        ) * dt[:, :, token][..., None]
        angle_state = angle_state - (2.0 * np.pi) * np.floor(angle_state / (2.0 * np.pi))
        cosine, sine = np.cos(angle_state), np.sin(angle_state)
        q_t = rotary(q[:, token] + query_bias[None], cosine, sine)
        k_t = rotary(k[:, token] + key_bias[None], cosine, sine)
        v_t = v[:, token]
        trap_t = sigmoid(trap[:, :, token])
        dt_t = dt[:, :, token]
        alpha = np.exp(adt[:, :, token])
        beta = (1.0 - trap_t) * dt_t * alpha
        gamma = trap_t * dt_t
        ssm_state = (
            alpha[..., None, None] * ssm_state
            + beta[..., None, None] * (key_state[..., None, :] * value_state[..., None])
            + gamma[..., None, None] * (k_t[..., None, :] * v_t[..., None])
        )
        output = np.einsum("bhvd,bhd->bhv", ssm_state, q_t)
        if d_skip is not None:
            output = output + np.asarray(d_skip, dtype=np.float64)[None, :, None] * v_t
        if gate is not None:
            g = np.asarray(gate, dtype=np.float64)[:, token]
            output = output * (g * sigmoid(g))  # silu
        outputs.append(output)
        key_state, value_state = k_t, v_t
    return np.stack(outputs, axis=1), (angle_state, ssm_state, key_state, value_state)


def regularized_solve(query, key, value, log_decay, beta, lamb,
                      h_kk_init=None, h_kv_init=None):
    """MesaNet dual covariance-state recurrence with a per-token regularized solve.

    State: two matrices ``h_kk, h_kv`` per (batch, head). Per token:
    ``h_kk = decay*h_kk + (beta*k)⊗k``; ``h_kv = decay*h_kv + (beta*k)⊗v``;
    read ``q_star = solve(h_kk + diag(lamb), q)`` then ``out = q_star^T h_kv``.
    query/key/value [B,T,H,K]; log_decay/beta [B,T,H]; lamb [H,K].
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    g = np.asarray(log_decay, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    lamb = np.asarray(lamb, dtype=np.float64)
    batch, sequence, heads, key_dim = q.shape
    state_shape = (batch, heads, key_dim, key_dim)
    h_kk = np.zeros(state_shape) if h_kk_init is None else np.asarray(h_kk_init, dtype=np.float64).copy()
    h_kv = np.zeros(state_shape) if h_kv_init is None else np.asarray(h_kv_init, dtype=np.float64).copy()
    regularizer = np.apply_along_axis(np.diag, -1, lamb)  # [H, K, K]
    outputs = []
    for t in range(sequence):
        decay = np.exp(g[:, t])[:, :, None, None]
        beta_t = b[:, t][:, :, None]
        k_beta = k[:, t] * beta_t
        h_kk = decay * h_kk + k_beta[..., None] * k[:, t][..., None, :]
        h_kv = decay * h_kv + k_beta[..., None] * v[:, t][..., None, :]
        q_star = np.linalg.solve(h_kk + regularizer[None], q[:, t][..., None]).squeeze(-1)
        outputs.append(np.einsum("bhk,bhkv->bhv", q_star, h_kv))
    return np.stack(outputs, axis=1), (h_kk, h_kv)


def second_order_cumsum(query, key, value):
    """HLA masked second-order causal attention via inclusive prefix sums.

    out_t = (q_t^T S_t) C_t - q_t^T G_t, where S_t = sum k k^T, C_t = sum q v^T,
    and G_t = sum k (k^T C_{prev}). Linear and scan-parallelizable.
    query/key [B,T,H,K], value [B,T,H,V].
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    delta_s = np.einsum("bthk,bthl->bthkl", k, k)
    delta_c = np.einsum("bthk,bthv->bthkv", q, v)
    state_s = np.cumsum(delta_s, axis=1)
    state_c = np.cumsum(delta_c, axis=1)
    previous_c = state_c - delta_c
    key_previous_c = np.einsum("bthk,bthkv->bthv", k, previous_c)
    delta_g = np.einsum("bthk,bthv->bthkv", k, key_previous_c)
    masked_correction = np.cumsum(delta_g, axis=1)
    query_state = np.einsum("bthk,bthkl->bthl", q, state_s)
    output = (
        np.einsum("bthl,bthlv->bthv", query_state, state_c)
        - np.einsum("bthk,bthkv->bthv", q, masked_correction)
    )
    return output, None
