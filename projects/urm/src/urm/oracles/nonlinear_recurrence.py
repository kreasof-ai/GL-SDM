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
