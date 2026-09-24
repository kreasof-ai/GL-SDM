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
    x = q.transpose(0, 2, 1)
    fft_size = 2 * sequence
    kernel_spectrum = np.fft.rfft(kernel, n=fft_size) / fft_size
    input_spectrum = np.fft.rfft(x, n=fft_size)
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
        q_t = q[:, token]
        k_t = k[:, token]
        v_t = v[:, token]
        km = np.einsum("bhd,bhde->bhe", k_t, update_base)
        reconstruction_target = v_t - k_t
        mean = km.mean(axis=-1, keepdims=True)
        rstd = np.sqrt(km.var(axis=-1, keepdims=True) + eps)
        km_hat = (km - mean) / rstd
        grad = (w[None] * km_hat + b[None] - reconstruction_target) * w[None]
        v_new = dim * grad - grad.sum(axis=-1, keepdims=True) / (rstd * dim)
        v_new = v_new - km_hat * (grad * km_hat).sum(axis=-1, keepdims=True) / (rstd * dim)
        theta_t = theta[:, token][..., None]
        alpha_t = alpha[:, token][..., None]
        eta_t = eta[:, token][..., None]
        momentum = eta_t * momentum - 2.0 * theta_t * (k_t[..., None] * v_new[..., None, :])
        memory = (1.0 - alpha_t[..., :1]) * memory + momentum
        output = np.einsum("bhd,bhde->bhe", q_t, memory)
        output_mean = output.mean(axis=-1, keepdims=True)
        output_rstd = np.sqrt(output.var(axis=-1, keepdims=True) + eps)
        output = (
            output
            + (output - output_mean) / output_rstd * w[None]
            + b[None]
        )
        outputs.append(output)
        if (token + 1) % chunk_size == 0:
            update_base = memory.copy()
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
            output = output * (g * sigmoid(g))
        outputs.append(output)
        key_state, value_state = k_t, v_t
    return np.stack(outputs, axis=1), (angle_state, ssm_state, key_state, value_state)


def momentum_delta(query, key, value, p, log_alpha, log_mu, beta, eta,
                   initial_state=None, initial_momentum=None, scale=None):
    """Momentum DeltaNet two-matrix-state recurrence (state + momentum).

    Per token: ``prediction = p^T state``; ``residual = v - prediction``;
    ``momentum = mu*momentum - (eta*k)⊗residual``; ``state = alpha*state -
    beta*momentum``; read ``(q*scale)^T state``. query/key/value/p [B,T,H,*];
    log_alpha/log_mu/beta/eta per-head schedules. Returns (output, (state,
    momentum)).
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_mu = np.asarray(log_mu, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state_shape = (batch, heads, key_dim, value_dim)
    state = np.zeros(state_shape) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    momentum = np.zeros(state_shape) if initial_momentum is None else np.asarray(initial_momentum, dtype=np.float64).copy()
    if scale is None:
        scale = key_dim ** -0.5
    outputs = []
    for token in range(sequence):
        q_t, k_t, v_t, p_t = q[:, token], k[:, token], v[:, token], p[:, token]
        alpha_t = np.exp(log_alpha[:, token])[..., None, None]
        mu_t = np.exp(log_mu[:, token])[..., None, None]
        beta_t = beta[:, token][..., None, None]
        eta_t = eta[:, token][..., None]
        prediction = np.einsum("bhk,bhkv->bhv", p_t, state)
        residual = v_t - prediction
        momentum = mu_t * momentum - (eta_t * k_t)[..., None] * residual[:, :, None, :]
        state = alpha_t * state - beta_t * momentum
        outputs.append(np.einsum("bhk,bhkv->bhv", q_t * scale, state))
    return np.stack(outputs, axis=1), (state, momentum)


def gated_oja(query, key, value, gate, beta, initial_state=None, scale=None):
    """Gated Oja value-channel recurrence (a transposed Hebbian/delta rule).

    State ``M`` is ``[K, V]`` per (batch, head). Per token: ``M *= exp(gate)``
    (value-channel decay); ``prediction = M @ v`` (contract the value axis);
    ``correction = beta * (k - prediction)``; ``M += correction ⊗ v``; read
    ``(q*scale)^T M``. The retrieval contracts the value axis (transposed delta).
    query/key [B,T,H,K]; value/gate [B,T,H,V]; beta [B,T,H].
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = (
        np.zeros((batch, heads, key_dim, value_dim))
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    if scale is None:
        scale = key_dim ** -0.5
    outputs = []
    for token in range(sequence):
        state = state * np.exp(gate[:, token])[:, :, None, :]
        prediction = np.einsum("bhkv,bhv->bhk", state, v[:, token])
        correction = beta[:, token][..., None] * (k[:, token] - prediction)
        state = state + correction[..., None] * v[:, token][:, :, None, :]
        outputs.append(np.einsum("bhk,bhkv->bhv", q[:, token] * scale, state))
    return np.stack(outputs, axis=1), state


def slot_attention_two_stage(query, key, value, slot_weights, log_decay,
                             initial_key_state=None, initial_value_state=None,
                             group_size=1):
    """ABC/GSA two-stage slot-addressed recurrence.

    Stage 1 accumulates a key state ``[K, S]`` (key ⊗ slot_weights, decayed) and
    routes slots by a softmax over ``q^T key_state``. Stage 2 accumulates a value
    state ``[S, V]`` (slot_weights ⊗ value, decayed) and reads it by the slot
    probabilities. query [B,T,Hq,K]; key/slot_weights/log_decay [B,T,Hk,*];
    value [B,T,Hk,V]. Returns (output, (key_state, value_state)) at the
    key-head granularity.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    sw = np.asarray(slot_weights, dtype=np.float64)
    g = np.asarray(log_decay, dtype=np.float64)
    batch, sequence, query_heads, key_dim = q.shape
    key_heads = k.shape[2]
    slots = sw.shape[-1]
    value_dim = v.shape[-1]
    rep_k = np.repeat(k, group_size, axis=2)
    rep_v = np.repeat(v, group_size, axis=2)
    rep_s = np.repeat(sw, group_size, axis=2)
    rep_g = np.repeat(g, group_size, axis=2)
    key_state = (
        np.zeros((batch, query_heads, key_dim, slots))
        if initial_key_state is None
        else np.repeat(np.asarray(initial_key_state, dtype=np.float64), group_size, axis=1)
    )
    value_state = (
        np.zeros((batch, query_heads, slots, value_dim))
        if initial_value_state is None
        else np.repeat(np.asarray(initial_value_state, dtype=np.float64), group_size, axis=1)
    )
    scale = key_dim ** -0.5
    slot_scores = []
    for token in range(sequence):
        decay = np.exp(rep_g[:, token])
        key_state = key_state * decay[:, :, None, :] + rep_k[:, token][..., None] * rep_s[:, token][:, :, None, :]
        slot_scores.append(
            ((q[:, token] * scale)[..., None] * key_state).sum(axis=-2)
        )
    slot_probability = np.stack(slot_scores, axis=1)
    slot_probability = np.exp(slot_probability - slot_probability.max(axis=-1, keepdims=True))
    slot_probability = slot_probability / slot_probability.sum(axis=-1, keepdims=True)
    outputs = []
    for token in range(sequence):
        decay = np.exp(rep_g[:, token])
        value_state = value_state * decay[..., None] + (
            rep_s[:, token][..., None] * rep_v[:, token][:, :, None, :]
        )
        outputs.append(
            (slot_probability[:, token][..., None] * value_state).sum(axis=-2)
        )
    output = np.stack(outputs, axis=1)
    final_key_state = key_state.reshape(batch, key_heads, group_size, key_dim, slots)[:, :, 0]
    final_value_state = value_state.reshape(batch, key_heads, group_size, slots, value_dim)[:, :, 0]
    return output, (final_key_state, final_value_state)


def rwkv4_scalar_state(w, u, key, value, state_input):
    """RWKV-4 scalar-state recurrence with log-space (exp-space) numerics.

    State per channel is ``(alpha, denominator_state, log_scale)``; the recurrence
    accumulates in a rescaled exp space for numerical stability. ``w``/``u`` are
    ``[C]`` decay/bonus; ``key``/``value`` are ``[B,T,C]``; ``state_input`` is
    ``[B,3,1,C]`` (alpha, denominator, log_scale). Returns (output, final_state)
    with final_state ``[B,3,1,C]``.
    """
    w = np.asarray(w, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    state_input = np.asarray(state_input, dtype=np.float64)
    batch, sequence, channels = key.shape
    decay = -np.exp(w)
    bonus = u
    alpha = state_input[:, 0, 0, :].copy()
    denominator_state = state_input[:, 1, 0, :].copy()
    log_scale = state_input[:, 2, 0, :].copy()
    outputs = []
    for token in range(sequence):
        key_t = key[:, token]
        value_t = value[:, token]
        bonus_key = bonus + key_t
        read_scale = np.maximum(log_scale, bonus_key)
        read_state_scale = np.exp(log_scale - read_scale)
        read_value_scale = np.exp(bonus_key - read_scale)
        output_t = (read_state_scale * alpha + read_value_scale * value_t) / (
            read_state_scale * denominator_state + read_value_scale
        )
        outputs.append(output_t)
        decayed_scale = decay + log_scale
        log_scale_next = np.maximum(decayed_scale, key_t)
        old_scale = np.exp(decayed_scale - log_scale_next)
        new_scale = np.exp(key_t - log_scale_next)
        alpha = old_scale * alpha + new_scale * value_t
        denominator_state = old_scale * denominator_state + new_scale
        log_scale = log_scale_next
    final_state = np.stack((alpha, denominator_state, log_scale), axis=1)[:, :, None, :]
    return np.stack(outputs, axis=1), final_state


def rwkv6_bonus_corrected(query, key, value, log_decay, bonus, initial_state=None):
    """RWKV-6 bonus-corrected additive matrix-state recurrence.

    Per token: ``decayed = state * exp(log_decay)`` (key-channel); the read uses
    ``state + (k*bonus)⊗v`` (a bonus correction on the pre-update state); the
    state updates to ``decayed + k⊗v``. query/key [B,T,H,K]; value [B,T,H,V];
    log_decay [B,T,H,K]; bonus [H,K]. Returns (output, state).
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    g = np.asarray(log_decay, dtype=np.float64)
    bonus = np.asarray(bonus, dtype=np.float64)
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = (
        np.zeros((batch, heads, key_dim, value_dim))
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).copy()
    )
    scale = key_dim ** -0.5
    outputs = []
    for token in range(sequence):
        decay = np.exp(g[:, token])
        decayed_state = state * decay[..., None]
        bonus_write = (k[:, token] * bonus[None])[..., None] * v[:, token][:, :, None, :]
        read_state = state + bonus_write
        outputs.append(np.einsum("bhk,bhkv->bhv", q[:, token] * scale, read_state))
        state = decayed_state + k[:, token][..., None] * v[:, token][:, :, None, :]
    return np.stack(outputs, axis=1), state


def mamba2_structured_ssm(x, dt, A, B, C, initial_states=None, groups=1):
    """Mamba-2 structured SSM recurrence.

    State ``[H, P, N]`` (head_dim x state_dim). Per token: ``decay = exp(dt*A)``;
    ``state = state*decay + (x ⊗ B)*dt``; read ``sum_N(state * C)``. ``x`` is
    ``[B,T,H,P]``; ``dt``/``A`` are per-head; ``B``/``C`` are ``[B,T,G,N]``
    (grouped, broadcast over heads). Returns (output, state).
    """
    x = np.asarray(x, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    batch, sequence, heads, head_dim = x.shape
    state_dim = B.shape[-1]
    groups = B.shape[2]
    b_heads = np.repeat(B, heads // groups, axis=2)
    c_heads = np.repeat(C, heads // groups, axis=2)
    state = (
        np.zeros((batch, heads, head_dim, state_dim))
        if initial_states is None
        else np.asarray(initial_states, dtype=np.float64).copy()
    )
    outputs = []
    for token in range(sequence):
        step = dt[:, token]
        decay = np.exp(step * A[None, :])
        state = state * decay[:, :, None, None]
        state = state + (
            x[:, token][..., None] * b_heads[:, token][:, :, None, :] * step[:, :, None, None]
        )
        outputs.append((state * c_heads[:, token][:, :, None, :]).sum(axis=-1))
    return np.stack(outputs, axis=1), state


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
    regularizer = np.apply_along_axis(np.diag, -1, lamb)
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
