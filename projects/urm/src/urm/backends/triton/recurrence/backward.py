"""Differentiable backwards for the distinguished K2 recurrence operators.

Each native executor in :mod:`urm.backends.triton.recurrence.nonlinear` and
:mod:`urm.backends.triton.recurrence.inner_state` is a forward-only Triton
kernel. This module wraps those kernels in :class:`torch.autograd.Function`
objects whose ``backward`` recomputes the exact per-token recurrence in
differentiable PyTorch (matching the canonical NumPy core and the Triton kernel
math) and differentiates through it with ``torch.autograd.grad``. This is a
correct differentiable recomputation path: the forward output is the native
Triton kernel, and the input gradients come from a from-scratch recomputation
of the same equation - not from the reference executor.

Correctness, not speed, is the goal; the recomputation runs in fp32 on the same
device as the saved inputs.
"""

from __future__ import annotations

import torch


def _grads(recomputed, inputs, grad_output):
    """Input gradients of ``recomputed`` w.r.t. ``inputs`` under ``grad_output``."""
    return torch.autograd.grad(
        recomputed,
        inputs,
        grad_outputs=grad_output.to(recomputed.dtype),
        allow_unused=True,
    )


# ----------------------------------------------------------------------
# tanh_rnn: h_t = tanh(h_{t-1} @ W + x_t); output is the state.
# ----------------------------------------------------------------------


class _TanhRnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, weight, initial_state):
        from urm.backends.triton.recurrence.nonlinear import execute_tanh_rnn

        output, final = execute_tanh_rnn(
            query=query, weight=weight, initial_state=initial_state
        )
        ctx.save_for_backward(query, weight, initial_state)
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, weight, initial_state = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        init = initial_state.detach().requires_grad_(True)
        with torch.enable_grad():
            state = init
            outs = []
            for t in range(q.shape[1]):
                state = torch.tanh(
                    torch.einsum("bnh,nhk->bnk", state, w) + q[:, t]
                )
                outs.append(state)
            recomputed = torch.stack(outs, dim=1)
        return (*_grads(recomputed, (q, w, init), grad_output),)


def _tanh_rnn_backwardable(query, weight, initial_state):
    return _TanhRnn.apply(query, weight, initial_state)


# ----------------------------------------------------------------------
# gated_rnn (GRU): reset/update-gated nonlinear state update.
# ----------------------------------------------------------------------


class _GatedRnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, weight, forget_input, forget_weight, reset_input,
                reset_weight, initial_state):
        from urm.backends.triton.recurrence.nonlinear import execute_gated_rnn

        output, final = execute_gated_rnn(
            query=query, weight=weight, forget_input=forget_input,
            forget_weight=forget_weight, reset_input=reset_input,
            reset_weight=reset_weight, initial_state=initial_state,
        )
        ctx.save_for_backward(
            query, weight, forget_input, forget_weight, reset_input,
            reset_weight, initial_state,
        )
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (query, weight, forget_input, forget_weight, reset_input,
         reset_weight, initial_state) = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        fi = forget_input.detach().requires_grad_(True)
        fw = forget_weight.detach().requires_grad_(True)
        ri = reset_input.detach().requires_grad_(True)
        rw = reset_weight.detach().requires_grad_(True)
        init = initial_state.detach().requires_grad_(True)
        with torch.enable_grad():
            state = init
            outs = []
            for t in range(q.shape[1]):
                forget = torch.sigmoid(
                    torch.einsum("bnh,nhk->bnk", state, fw) + fi[:, t]
                )
                reset = torch.sigmoid(
                    torch.einsum("bnh,nhk->bnk", state, rw) + ri[:, t]
                )
                candidate = torch.tanh(
                    torch.einsum("bnh,nhk->bnk", state * reset, w) + q[:, t]
                )
                state = forget * state + (1.0 - forget) * candidate
                outs.append(state)
            recomputed = torch.stack(outs, dim=1)
        return _grads(recomputed, (q, w, fi, fw, ri, rw, init), grad_output)


def _gated_rnn_backwardable(query, weight, forget_input, forget_weight,
                            reset_input, reset_weight, initial_state):
    return _GatedRnn.apply(
        query, weight, forget_input, forget_weight, reset_input, reset_weight,
        initial_state,
    )


# ----------------------------------------------------------------------
# multiplicative_rnn (m2rnn): second-order matrix memory.
# ----------------------------------------------------------------------


class _MultiplicativeRnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, weight, forget_input, initial_state):
        from urm.backends.triton.recurrence.nonlinear import (
            execute_multiplicative_rnn,
        )

        output, final = execute_multiplicative_rnn(
            query=query, key=key, value=value, weight=weight,
            forget_input=forget_input, initial_state=initial_state,
        )
        ctx.save_for_backward(query, key, value, weight, forget_input,
                              initial_state)
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, key, value, weight, forget_input, initial_state = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        w = weight.detach().requires_grad_(True)
        fi = forget_input.detach().requires_grad_(True)
        init = initial_state.detach().requires_grad_(True)
        with torch.enable_grad():
            state = init
            outs = []
            for t in range(q.shape[1]):
                update = k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
                candidate = torch.tanh(
                    torch.matmul(state, w.unsqueeze(0)) + update
                )
                forget = fi[:, t].unsqueeze(-1).unsqueeze(-1)
                state = forget * state + (1.0 - forget) * candidate
                outs.append(
                    torch.matmul(q[:, t].unsqueeze(-2), state).squeeze(-2)
                )
            recomputed = torch.stack(outs, dim=1)
        return _grads(recomputed, (q, k, v, w, fi, init), grad_output)


def _multiplicative_rnn_backwardable(query, key, value, weight, forget_input,
                                     initial_state):
    return _MultiplicativeRnn.apply(
        query, key, value, weight, forget_input, initial_state
    )


# ----------------------------------------------------------------------
# rwkv4_scalar_state: per-channel scalar (alpha, denom, log_scale).
# ----------------------------------------------------------------------


class _Rwkv4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, key, value, state_input):
        from urm.backends.triton.recurrence.nonlinear import (
            execute_rwkv4_scalar_state,
        )

        output, final = execute_rwkv4_scalar_state(
            w=w, u=u, key=key, value=value, state_input=state_input
        )
        ctx.save_for_backward(w, u, key, value, state_input)
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        w, u, key, value, state_input = ctx.saved_tensors
        w_ = w.detach().requires_grad_(True)
        u_ = u.detach().requires_grad_(True)
        key_ = key.detach().requires_grad_(True)
        value_ = value.detach().requires_grad_(True)
        state_ = state_input.detach().requires_grad_(True)
        with torch.enable_grad():
            decay = -torch.exp(w_)
            bonus = u_
            alpha = state_[:, 0, 0, :]
            denom = state_[:, 1, 0, :]
            log_scale = state_[:, 2, 0, :]
            outs = []
            for t in range(key_.shape[1]):
                key_t = key_[:, t]
                value_t = value_[:, t]
                bonus_key = bonus + key_t
                read_scale = torch.maximum(log_scale, bonus_key)
                read_state_scale = torch.exp(log_scale - read_scale)
                read_value_scale = torch.exp(bonus_key - read_scale)
                output_t = (
                    read_state_scale * alpha + read_value_scale * value_t
                ) / (read_state_scale * denom + read_value_scale)
                outs.append(output_t)
                decayed_scale = decay + log_scale
                log_scale_next = torch.maximum(decayed_scale, key_t)
                old_scale = torch.exp(decayed_scale - log_scale_next)
                new_scale = torch.exp(key_t - log_scale_next)
                alpha = old_scale * alpha + new_scale * value_t
                denom = old_scale * denom + new_scale
                log_scale = log_scale_next
            recomputed = torch.stack(outs, dim=1)
        return _grads(recomputed, (w_, u_, key_, value_, state_), grad_output)


def _rwkv4_backwardable(w, u, key, value, state_input):
    return _Rwkv4.apply(w, u, key, value, state_input)


# ----------------------------------------------------------------------
# rwkv6_bonus_corrected: matrix state with a static bonus read.
# ----------------------------------------------------------------------


class _Rwkv6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, log_decay, bonus, initial_state):
        from urm.backends.triton.recurrence.nonlinear import (
            execute_rwkv6_bonus_corrected,
        )

        has_initial = initial_state is not None
        output, final = execute_rwkv6_bonus_corrected(
            query=query, key=key, value=value, log_decay=log_decay, bonus=bonus,
            initial_state=initial_state,
        )
        ctx.save_for_backward(query, key, value, log_decay, bonus, initial_state)
        ctx.has_initial = has_initial
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, key, value, log_decay, bonus, initial_state = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        g = log_decay.detach().requires_grad_(True)
        bonus_ = bonus.detach().requires_grad_(True)
        batch, _, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        if ctx.has_initial:
            init = initial_state.detach().requires_grad_(True)
            state0 = init
        else:
            init = None
            state0 = torch.zeros(
                (batch, heads, key_dim, value_dim), device=q.device,
                dtype=torch.float32,
            )
        scale = key_dim ** -0.5
        with torch.enable_grad():
            state = state0
            outs = []
            for t in range(q.shape[1]):
                decay = torch.exp(g[:, t])
                decayed_state = state * decay.unsqueeze(-1)
                bonus_write = (k[:, t] * bonus_.unsqueeze(0)).unsqueeze(-1) * v[
                    :, t
                ].unsqueeze(-2)
                read_state = state + bonus_write
                outs.append(
                    torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, read_state)
                )
                state = decayed_state + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
            recomputed = torch.stack(outs, dim=1)
        inputs = (q, k, v, g, bonus_) + ((init,) if ctx.has_initial else ())
        grads = _grads(recomputed, inputs, grad_output)
        if not ctx.has_initial:
            grads = (*grads, None)
        return grads


def _rwkv6_backwardable(query, key, value, log_decay, bonus, initial_state):
    return _Rwkv6.apply(query, key, value, log_decay, bonus, initial_state)


# ----------------------------------------------------------------------
# mamba2_structured_ssm: structured SSM with continuous-time head decay.
# ----------------------------------------------------------------------


class _Mamba2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, initial_states):
        from urm.backends.triton.recurrence.nonlinear import (
            execute_mamba2_structured_ssm,
        )

        has_initial = initial_states is not None
        output, final = execute_mamba2_structured_ssm(
            x=x, dt=dt, A=A, B=B, C=C, initial_states=initial_states
        )
        ctx.save_for_backward(x, dt, A, B, C, initial_states)
        ctx.has_initial = has_initial
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        x, dt, A, B, C, initial_states = ctx.saved_tensors
        x_ = x.detach().requires_grad_(True)
        dt_ = dt.detach().requires_grad_(True)
        A_ = A.detach().requires_grad_(True)
        B_ = B.detach().requires_grad_(True)
        C_ = C.detach().requires_grad_(True)
        batch, _, heads, head_dim = x_.shape
        groups = B_.shape[2]
        state_dim = B_.shape[-1]
        if ctx.has_initial:
            init = initial_states.detach().requires_grad_(True)
            state0 = init
        else:
            init = None
            state0 = torch.zeros(
                (batch, heads, head_dim, state_dim), device=x_.device,
                dtype=torch.float32,
            )
        with torch.enable_grad():
            b_heads = B_.repeat_interleave(heads // groups, dim=2)
            c_heads = C_.repeat_interleave(heads // groups, dim=2)
            state = state0
            outs = []
            for t in range(x_.shape[1]):
                step = dt_[:, t]
                decay = torch.exp(step * A_.unsqueeze(0))
                state = state * decay.unsqueeze(-1).unsqueeze(-1)
                state = state + (
                    x_[:, t].unsqueeze(-1)
                    * b_heads[:, t].unsqueeze(-2)
                    * step.unsqueeze(-1).unsqueeze(-1)
                )
                outs.append((state * c_heads[:, t].unsqueeze(-2)).sum(dim=-1))
            recomputed = torch.stack(outs, dim=1)
        inputs = (x_, dt_, A_, B_, C_) + ((init,) if ctx.has_initial else ())
        grads = _grads(recomputed, inputs, grad_output)
        if not ctx.has_initial:
            grads = (*grads, None)
        return grads


def _mamba2_backwardable(x, dt, A, B, C, initial_states):
    return _Mamba2.apply(x, dt, A, B, C, initial_states)


# ----------------------------------------------------------------------
# gated_oja: value-channel recurrence with key residual correction.
# ----------------------------------------------------------------------


class _GatedOja(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, gate, beta, initial_state, scale):
        from urm.backends.triton.recurrence.inner_state import execute_gated_oja

        has_initial = initial_state is not None
        output, final = execute_gated_oja(
            query=query, key=key, value=value, gate=gate, beta=beta,
            initial_state=initial_state, scale=scale,
        )
        ctx.save_for_backward(query, key, value, gate, beta, initial_state)
        ctx.has_initial = has_initial
        ctx.scale = scale
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, key, value, gate, beta, initial_state = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        g = gate.detach().requires_grad_(True)
        beta_ = beta.detach().requires_grad_(True)
        batch, _, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        if ctx.has_initial:
            init = initial_state.detach().requires_grad_(True)
            state0 = init
        else:
            init = None
            state0 = torch.zeros(
                (batch, heads, key_dim, value_dim), device=q.device,
                dtype=torch.float32,
            )
        scale = ctx.scale if ctx.scale is not None else key_dim ** -0.5
        with torch.enable_grad():
            state = state0
            outs = []
            for t in range(q.shape[1]):
                state = state * torch.exp(g[:, t]).unsqueeze(-2)
                prediction = (state * v[:, t].unsqueeze(-2)).sum(dim=-1)
                correction = beta_[:, t].unsqueeze(-1) * (k[:, t] - prediction)
                state = state + correction.unsqueeze(-1) * v[:, t].unsqueeze(-2)
                outs.append(
                    torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state)
                )
            recomputed = torch.stack(outs, dim=1)
        inputs = (q, k, v, g, beta_) + ((init,) if ctx.has_initial else ())
        grads = _grads(recomputed, inputs, grad_output)
        if not ctx.has_initial:
            grads = (*grads, None)
        return (*grads, None)  # trailing None for scale


def _gated_oja_backwardable(query, key, value, gate, beta, initial_state, scale):
    return _GatedOja.apply(query, key, value, gate, beta, initial_state, scale)


# ----------------------------------------------------------------------
# regularized_solve (mesa_net): dual covariance-state recurrence + solve.
# ----------------------------------------------------------------------


class _RegularizedSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, log_decay, beta, lamb):
        from urm.backends.triton.recurrence.inner_state import (
            execute_regularized_solve,
        )

        output, final = execute_regularized_solve(
            query=query, key=key, value=value, log_decay=log_decay, beta=beta,
            lamb=lamb,
        )
        ctx.save_for_backward(query, key, value, log_decay, beta, lamb)
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, key, value, log_decay, beta, lamb = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        g = log_decay.detach().requires_grad_(True)
        beta_ = beta.detach().requires_grad_(True)
        lamb_ = lamb.detach().requires_grad_(True)
        batch, _, heads, key_dim = q.shape
        with torch.enable_grad():
            regularizer = torch.diag_embed(lamb_)
            h_kk = torch.zeros(
                (batch, heads, key_dim, key_dim), device=q.device,
                dtype=torch.float32,
            )
            h_kv = torch.zeros_like(h_kk)
            outs = []
            for t in range(q.shape[1]):
                decay = torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)
                beta_t = beta_[:, t].unsqueeze(-1)
                k_beta = k[:, t] * beta_t
                h_kk = decay * h_kk + k_beta.unsqueeze(-1) * k[:, t].unsqueeze(-2)
                h_kv = decay * h_kv + k_beta.unsqueeze(-1) * v[:, t].unsqueeze(-2)
                q_star = torch.linalg.solve(
                    h_kk + regularizer.unsqueeze(0), q[:, t].unsqueeze(-1)
                ).squeeze(-1)
                outs.append(torch.einsum("bhk,bhkv->bhv", q_star, h_kv))
            recomputed = torch.stack(outs, dim=1)
        return _grads(recomputed, (q, k, v, g, beta_, lamb_), grad_output)


def _regularized_solve_backwardable(query, key, value, log_decay, beta, lamb):
    return _RegularizedSolve.apply(query, key, value, log_decay, beta, lamb)


# ----------------------------------------------------------------------
# slot_attention_two_stage (abc_core, gsa_core).
# ----------------------------------------------------------------------


class _SlotAttentionTwoStage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, slot_weights, log_decay, group_size):
        from urm.backends.triton.recurrence.inner_state import (
            execute_slot_attention_two_stage,
        )

        output, final = execute_slot_attention_two_stage(
            query=query, key=key, value=value, slot_weights=slot_weights,
            log_decay=log_decay, group_size=group_size,
        )
        ctx.save_for_backward(query, key, value, slot_weights, log_decay)
        ctx.group_size = group_size
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        query, key, value, slot_weights, log_decay = ctx.saved_tensors
        group_size = ctx.group_size
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        sw = slot_weights.detach().requires_grad_(True)
        ld = log_decay.detach().requires_grad_(True)
        batch, sequence, query_heads, key_dim = q.shape
        slots = sw.shape[-1]
        value_dim = v.shape[-1]
        scale = key_dim ** -0.5
        with torch.enable_grad():
            rep_k = k.repeat_interleave(group_size, dim=2)
            rep_v = v.repeat_interleave(group_size, dim=2)
            rep_s = sw.repeat_interleave(group_size, dim=2)
            rep_g = ld.repeat_interleave(group_size, dim=2)
            key_state = torch.zeros(
                (batch, query_heads, key_dim, slots), device=q.device,
                dtype=torch.float32,
            )
            slot_scores = []
            for t in range(sequence):
                decay = torch.exp(rep_g[:, t])
                key_state = key_state * decay.unsqueeze(-2) + rep_k[:, t].unsqueeze(
                    -1
                ) * rep_s[:, t].unsqueeze(-2)
                slot_scores.append(
                    ((q[:, t] * scale).unsqueeze(-1) * key_state).sum(dim=-2)
                )
            slot_probability = torch.stack(slot_scores, dim=1).softmax(dim=-1)
            value_state = torch.zeros(
                (batch, query_heads, slots, value_dim), device=q.device,
                dtype=torch.float32,
            )
            outs = []
            for t in range(sequence):
                decay = torch.exp(rep_g[:, t])
                value_state = value_state * decay.unsqueeze(-1) + (
                    rep_s[:, t].unsqueeze(-1) * rep_v[:, t].unsqueeze(-2)
                )
                outs.append(
                    (slot_probability[:, t].unsqueeze(-1) * value_state).sum(dim=-2)
                )
            recomputed = torch.stack(outs, dim=1)
        grads = _grads(recomputed, (q, k, v, sw, ld), grad_output)
        return (*grads, None)  # trailing None for group_size


def _slot_attention_backwardable(query, key, value, slot_weights, log_decay,
                                 group_size):
    return _SlotAttentionTwoStage.apply(
        query, key, value, slot_weights, log_decay, group_size
    )


# ----------------------------------------------------------------------
# trapezoidal_ssm (mamba3_siso): rotary angle accumulator + four-state SSM.
# ----------------------------------------------------------------------


class _TrapezoidalSsm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, adt, dt, trap, query_bias, key_bias,
                angles):
        from urm.backends.triton.recurrence.nonlinear import (
            execute_trapezoidal_ssm,
        )

        output, final = execute_trapezoidal_ssm(
            query=query, key=key, value=value, adt=adt, dt=dt, trap=trap,
            query_bias=query_bias, key_bias=key_bias, angles=angles,
        )
        ctx.save_for_backward(
            query, key, value, adt, dt, trap, query_bias, key_bias, angles
        )
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (query, key, value, adt, dt, trap, query_bias, key_bias,
         angles) = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        adt_ = adt.detach().requires_grad_(True)
        dt_ = dt.detach().requires_grad_(True)
        trap_ = trap.detach().requires_grad_(True)
        qb = query_bias.detach().requires_grad_(True)
        kb = key_bias.detach().requires_grad_(True)
        ang = angles.detach().requires_grad_(True)
        batch, sequence, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        angle_dim = ang.shape[-1]
        pi = 3.141592653589793
        two_pi = 2.0 * pi

        def rotary(tensor, cosine, sine):
            paired = tensor.reshape(batch, heads, key_dim // 2, 2)
            first, second = paired.unbind(dim=-1)
            if cosine.shape[-1] < key_dim // 2:
                padding = key_dim // 2 - cosine.shape[-1]
                cosine = torch.nn.functional.pad(cosine, (0, padding), value=1.0)
                sine = torch.nn.functional.pad(sine, (0, padding), value=0.0)
            return torch.stack(
                (first * cosine - second * sine, first * sine + second * cosine),
                dim=-1,
            ).reshape(batch, heads, key_dim)

        with torch.enable_grad():
            angle_state = torch.zeros(
                (batch, heads, angle_dim), device=q.device, dtype=torch.float32
            )
            ssm_state = torch.zeros(
                (batch, heads, value_dim, key_dim), device=q.device,
                dtype=torch.float32,
            )
            key_state = torch.zeros(
                (batch, heads, key_dim), device=q.device, dtype=torch.float32
            )
            value_state = torch.zeros(
                (batch, heads, value_dim), device=q.device, dtype=torch.float32
            )
            outs = []
            for t in range(sequence):
                angle_state = angle_state + (
                    torch.tanh(ang[:, t]) * pi
                ) * dt_[:, :, t].unsqueeze(-1)
                angle_state = angle_state - two_pi * torch.floor(
                    angle_state / two_pi
                )
                cosine, sine = angle_state.cos(), angle_state.sin()
                q_t = rotary(q[:, t] + qb.unsqueeze(0), cosine, sine)
                k_t = rotary(k[:, t] + kb.unsqueeze(0), cosine, sine)
                v_t = v[:, t]
                trap_t = torch.sigmoid(trap_[:, :, t])
                dt_t = dt_[:, :, t]
                alpha = adt_[:, :, t].exp()
                beta = (1.0 - trap_t) * dt_t * alpha
                gamma = trap_t * dt_t
                ssm_state = (
                    alpha.unsqueeze(-1).unsqueeze(-1) * ssm_state
                    + beta.unsqueeze(-1).unsqueeze(-1)
                    * (key_state.unsqueeze(-2) * value_state.unsqueeze(-1))
                    + gamma.unsqueeze(-1).unsqueeze(-1)
                    * (k_t.unsqueeze(-2) * v_t.unsqueeze(-1))
                )
                outs.append(torch.einsum("bhvd,bhd->bhv", ssm_state, q_t))
                key_state, value_state = k_t, v_t
            recomputed = torch.stack(outs, dim=1)
        return _grads(
            recomputed, (q, k, v, adt_, dt_, trap_, qb, kb, ang), grad_output
        )


def _trapezoidal_ssm_backwardable(query, key, value, adt, dt, trap, query_bias,
                                  key_bias, angles):
    return _TrapezoidalSsm.apply(
        query, key, value, adt, dt, trap, query_bias, key_bias, angles
    )


# ----------------------------------------------------------------------
# layernorm_inner_state (ttt_linear): chunkwise inner-loss update.
# ----------------------------------------------------------------------


class _LayernormInnerState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, w, b, eta, initial_state,
                initial_state_bias, chunk_size, eps):
        from urm.backends.triton.recurrence.inner_state import (
            execute_layernorm_inner_state,
        )

        output, final = execute_layernorm_inner_state(
            query=query, key=key, value=value, w=w, b=b, eta=eta,
            initial_state=initial_state, initial_state_bias=initial_state_bias,
            chunk_size=chunk_size, eps=eps,
        )
        ctx.save_for_backward(
            query, key, value, w, b, eta, initial_state, initial_state_bias
        )
        ctx.chunk_size = chunk_size
        ctx.eps = eps
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (query, key, value, w, b, eta, initial_state,
         initial_state_bias) = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        eps = ctx.eps
        q0 = query.detach().requires_grad_(True)
        k0 = key.detach().requires_grad_(True)
        v0 = value.detach().requires_grad_(True)
        w_ = w.detach().requires_grad_(True)
        b_ = b.detach().requires_grad_(True)
        eta0 = eta.detach().requires_grad_(True)
        batch, sequence, heads, dim = q0.shape
        has_initial = initial_state is not None
        init_mem = (
            initial_state.detach().requires_grad_(True) if has_initial else None
        )
        init_bias = (
            initial_state_bias.detach().requires_grad_(True)
            if has_initial and initial_state_bias is not None
            else None
        )
        with torch.enable_grad():
            memory = (
                torch.zeros(
                    (batch, heads, dim, dim), device=q0.device, dtype=torch.float32
                )
                if init_mem is None
                else init_mem
            )
            memory_bias = (
                torch.zeros(
                    (batch, heads, 1, dim), device=q0.device, dtype=torch.float32
                )
                if init_bias is None
                else init_bias.reshape(batch, heads, 1, dim)
            )
            q = q0.transpose(1, 2) * (dim ** -0.5)
            k = k0.transpose(1, 2)
            v = v0.transpose(1, 2)
            eta_t = eta0.transpose(1, 2)
            w_r = w_.reshape(heads, 1, dim)
            b_r = b_.reshape(heads, 1, dim)
            outs = []
            for start in range(0, sequence, chunk_size):
                stop = start + chunk_size
                qc, kc, vc, ec = (
                    q[:, :, start:stop],
                    k[:, :, start:stop],
                    v[:, :, start:stop],
                    eta_t[:, :, start:stop],
                )
                kh = kc @ memory + memory_bias
                target = vc - kc
                mean = kh.mean(dim=-1, keepdim=True)
                rstd = torch.rsqrt(kh.var(dim=-1, unbiased=False, keepdim=True) + eps)
                kh_hat = (kh - mean) * rstd
                grad = (w_r * kh_hat + b_r - target) * w_r
                grad = (
                    (
                        dim * grad
                        - grad.sum(dim=-1, keepdim=True)
                        - kh_hat * (grad * kh_hat).sum(dim=-1, keepdim=True)
                    )
                    * rstd
                    / dim
                )
                attention = torch.tril(qc @ kc.transpose(-1, -2))
                output_chunk = (
                    qc @ memory
                    - (ec * attention) @ grad
                    + memory_bias
                    - torch.tril(ec.expand_as(attention)) @ grad
                )
                eta_last = ec[:, :, -1, :, None]
                memory = memory - (eta_last * kc).transpose(-1, -2) @ grad
                memory_bias = memory_bias - (eta_last * grad).sum(
                    dim=-2, keepdim=True
                )
                out_mean = output_chunk.mean(dim=-1, keepdim=True)
                out_rstd = torch.rsqrt(
                    output_chunk.var(dim=-1, unbiased=False, keepdim=True) + eps
                )
                outs.append(
                    output_chunk + (output_chunk - out_mean) * out_rstd * w_r + b_r
                )
            recomputed = torch.cat(outs, dim=-2).transpose(1, 2)
        inputs = [q0, k0, v0, w_, b_, eta0]
        if init_mem is not None:
            inputs.append(init_mem)
        if init_bias is not None:
            inputs.append(init_bias)
        grads = list(_grads(recomputed, tuple(inputs), grad_output))
        if init_mem is None:
            grads.append(None)
        if init_bias is None:
            grads.append(None)
        return (*grads, None, None)  # trailing Nones for chunk_size, eps


def _layernorm_inner_state_backwardable(query, key, value, w, b, eta,
                                        initial_state, initial_state_bias,
                                        chunk_size, eps):
    return _LayernormInnerState.apply(
        query, key, value, w, b, eta, initial_state, initial_state_bias,
        chunk_size, eps,
    )


# ----------------------------------------------------------------------
# momentum_inner_state (titans_linear_memory): tokenwise memory + momentum.
# ----------------------------------------------------------------------


class _MomentumInnerState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, w, b, theta, alpha, eta, initial_state,
                chunk_size, eps):
        from urm.backends.triton.recurrence.inner_state import (
            execute_momentum_inner_state,
        )

        output, final = execute_momentum_inner_state(
            query=query, key=key, value=value, w=w, b=b, theta=theta,
            alpha=alpha, eta=eta, initial_state=initial_state,
            chunk_size=chunk_size, eps=eps,
        )
        ctx.save_for_backward(
            query, key, value, w, b, theta, alpha, eta, initial_state
        )
        ctx.chunk_size = chunk_size
        ctx.eps = eps
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (query, key, value, w, b, theta, alpha, eta,
         initial_state) = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        eps = ctx.eps
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        w_ = w.detach().requires_grad_(True)
        b_ = b.detach().requires_grad_(True)
        theta = theta.detach().requires_grad_(True)
        alpha = alpha.detach().requires_grad_(True)
        eta = eta.detach().requires_grad_(True)
        batch, sequence, heads, dim = q.shape
        has_initial = initial_state is not None
        init = (
            initial_state.detach().requires_grad_(True) if has_initial else None
        )
        with torch.enable_grad():
            memory = (
                torch.zeros(
                    (batch, heads, dim, dim), device=q.device, dtype=torch.float32
                )
                if init is None
                else init
            )
            momentum = torch.zeros_like(memory)
            update_base = memory
            outs = []
            for token in range(sequence):
                q_t = q[:, token].unsqueeze(-2)
                k_t = k[:, token].unsqueeze(-2)
                v_t = v[:, token].unsqueeze(-2)
                km = k_t @ update_base
                reconstruction_target = v_t - k_t
                mean = km.mean(dim=-1, keepdim=True)
                rstd = torch.sqrt(km.var(dim=-1, unbiased=False, keepdim=True) + eps)
                km_hat = (km - mean) / rstd
                grad = (
                    w_[None, :, None, :] * km_hat
                    + b_[None, :, None, :]
                    - reconstruction_target
                ) * w_[None, :, None, :]
                v_new = dim * grad - grad.sum(dim=-1, keepdim=True) / (rstd * dim)
                v_new = v_new - km_hat * (grad * km_hat).sum(
                    dim=-1, keepdim=True
                ) / (rstd * dim)
                theta_t = theta[:, token].unsqueeze(-2)
                alpha_t = alpha[:, token].unsqueeze(-2)
                eta_t = eta[:, token].unsqueeze(-2)
                momentum = eta_t * momentum - 2.0 * theta_t * (
                    k_t.transpose(-1, -2) @ v_new
                )
                memory = (1.0 - alpha_t[..., :1]) * memory + momentum
                output = q_t @ memory
                output_mean = output.mean(dim=-1, keepdim=True)
                output_rstd = torch.sqrt(
                    output.var(dim=-1, unbiased=False, keepdim=True) + eps
                )
                output = (
                    output
                    + (output - output_mean) / output_rstd * w_[None, :, None, :]
                    + b_[None, :, None, :]
                )
                outs.append(output.squeeze(-2))
                if (token + 1) % chunk_size == 0:
                    update_base = memory
            recomputed = torch.stack(outs, dim=1)
        inputs = [q, k, v, w_, b_, theta, alpha, eta]
        if init is not None:
            inputs.append(init)
        grads = list(_grads(recomputed, tuple(inputs), grad_output))
        if init is None:
            grads.append(None)
        return (*grads, None, None)  # trailing Nones for chunk_size, eps


def _momentum_inner_state_backwardable(query, key, value, w, b, theta, alpha,
                                       eta, initial_state, chunk_size, eps):
    return _MomentumInnerState.apply(
        query, key, value, w, b, theta, alpha, eta, initial_state, chunk_size,
        eps,
    )


# ----------------------------------------------------------------------
# momentum_delta (momentum_delta_core): coupled fast-weight + momentum
# matrix states. prediction = p^T state; residual = v - prediction;
# momentum = mu*momentum - (eta*k) outer residual; state = alpha*state -
# beta*momentum; read (q*scale)^T state.
# ----------------------------------------------------------------------


class _MomentumDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, p, log_alpha, log_mu, beta, eta,
                initial_state, initial_momentum, scale):
        from urm.backends.triton.recurrence.inner_state import execute_momentum_delta

        output, final = execute_momentum_delta(
            query=query, key=key, value=value, p=p, log_alpha=log_alpha,
            log_mu=log_mu, beta=beta, eta=eta, initial_state=initial_state,
            initial_momentum=initial_momentum, scale=scale,
        )
        ctx.save_for_backward(
            query, key, value, p, log_alpha, log_mu, beta, eta,
            initial_state, initial_momentum,
        )
        ctx.scale = scale
        return output, final

    @staticmethod
    def backward(ctx, grad_output, grad_final):
        (query, key, value, p, log_alpha, log_mu, beta, eta,
         initial_state, initial_momentum) = ctx.saved_tensors
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        p_ = p.detach().requires_grad_(True)
        la = log_alpha.detach().requires_grad_(True)
        lm = log_mu.detach().requires_grad_(True)
        beta_ = beta.detach().requires_grad_(True)
        eta_ = eta.detach().requires_grad_(True)
        batch, _, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        state_shape = (batch, heads, key_dim, value_dim)
        init_s = (
            initial_state.detach().requires_grad_(True)
            if initial_state is not None else None
        )
        init_m = (
            initial_momentum.detach().requires_grad_(True)
            if initial_momentum is not None else None
        )
        state0 = init_s if init_s is not None else torch.zeros(
            state_shape, device=q.device, dtype=torch.float32)
        mom0 = init_m if init_m is not None else torch.zeros(
            state_shape, device=q.device, dtype=torch.float32)
        scale = ctx.scale if ctx.scale is not None else key_dim ** -0.5
        with torch.enable_grad():
            state = state0
            momentum = mom0
            outs = []
            for t in range(q.shape[1]):
                alpha_t = torch.exp(la[:, t])[..., None, None]
                mu_t = torch.exp(lm[:, t])[..., None, None]
                beta_t = beta_[:, t][..., None, None]
                eta_t = eta_[:, t][..., None]
                prediction = torch.einsum("bhk,bhkv->bhv", p_[:, t], state)
                residual = v[:, t] - prediction
                momentum = mu_t * momentum - (eta_t * k[:, t])[..., None] * residual[:, :, None, :]
                state = alpha_t * state - beta_t * momentum
                outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
            recomputed = torch.stack(outs, dim=1)
        inputs = [q, k, v, p_, la, lm, beta_, eta_]
        if init_s is not None:
            inputs.append(init_s)
        if init_m is not None:
            inputs.append(init_m)
        grads = list(_grads(recomputed, tuple(inputs), grad_output))
        if init_s is None:
            grads.append(None)
        if init_m is None:
            grads.append(None)
        return (*grads, None)  # trailing None for scale


def _momentum_delta_backwardable(query, key, value, p, log_alpha, log_mu, beta,
                                 eta, initial_state, initial_momentum, scale):
    return _MomentumDelta.apply(
        query, key, value, p, log_alpha, log_mu, beta, eta, initial_state,
        initial_momentum, scale,
    )
