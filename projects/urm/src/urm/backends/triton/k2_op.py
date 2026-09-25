"""Compile-opaque wrapper for the native K2 matrix-state recurrence (ATMA custom-op pattern).

The fused Triton scan builds its autograd.Function inside an ``@lru_cache``'d closure that
dynamo cannot trace. This module wraps the recurrence as a PAIR of ``torch.library``
custom ops — a fused forward op and a fused backward op — so a ``torch.compile``'d model
treats the mixer as opaque nodes in BOTH the forward and the compiled backward graph, with
no trace into the kernel closure and no graph break. This is ATMA's
``external_baselines/custom_ops.py`` pattern: the forward and backward are separate opaque
ops; the autograd bridge calls the opaque backward op.

Scope: the canonical K2 envelope (delta/additive, gate scopes none/scalar/head/channel,
before/after read, no normalizer/A8 transition features) — the native provider's envelope.
"""

from __future__ import annotations

import torch

from .k2 import execute_matrix_state_recurrence


@torch.library.custom_op("urm::k2_recurrence_fwd", mutates_args=())
def _k2_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor | None,
    beta: torch.Tensor | None,
    initial_state: torch.Tensor,
    scale: float | None,
    decay_granularity: str,
    is_delta: bool,
    read_before: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, final = execute_matrix_state_recurrence(
        query=query, key=key, value=value, log_decay=log_decay, beta=beta,
        initial_state=initial_state, scale=scale, decay_granularity=decay_granularity,
        is_delta=is_delta, read_before=read_before, normalizer=False,
    )
    return out, final


@_k2_fwd.register_fake
def _(query, key, value, log_decay, beta, initial_state, scale, decay_granularity,
      is_delta, read_before):
    B, T, H, K = query.shape
    V = value.shape[-1]
    return (query.new_empty((B, T, H, V)),
            query.new_empty((B, H, K, V), dtype=torch.float32))


@torch.library.custom_op("urm::k2_recurrence_bwd", mutates_args=())
def _k2_bwd(
    grad_output: torch.Tensor,
    grad_final: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None,
    decay_granularity: str,
    is_delta: bool,
    read_before: bool,
    has_log_decay: bool,
    has_beta: bool,
) -> list[torch.Tensor]:
    """Fused backward as an opaque op. Absent optional operands arrive as empty tensors.

    Re-runs the recurrence under enable_grad and differentiates it, so the fused Triton
    backward kernel produces the operand gradients. Returns the six gradients in the
    forward's operand order (a zero tensor where an operand was absent).
    """
    ld_in = log_decay if has_log_decay else None
    bt_in = beta if has_beta else None
    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        k = key.detach().requires_grad_(True)
        v = value.detach().requires_grad_(True)
        ld = ld_in.detach().requires_grad_(True) if has_log_decay else None
        bt = bt_in.detach().requires_grad_(True) if has_beta else None
        init = initial_state.detach().requires_grad_(True)
        out, final = execute_matrix_state_recurrence(
            query=q, key=k, value=v, log_decay=ld, beta=bt, initial_state=init,
            scale=scale, decay_granularity=decay_granularity, is_delta=is_delta,
            read_before=read_before, normalizer=False,
        )
        targets = [q, k, v] + ([ld] if ld is not None else []) + ([bt] if bt is not None else []) + [init]
        grads = torch.autograd.grad([out, final], targets, [grad_output, grad_final],
                                    allow_unused=True)
    gq, gk, gv = grads[0], grads[1], grads[2]
    i = 3
    gg = grads[i] if has_log_decay else torch.zeros_like(log_decay)
    i += 1 if has_log_decay else 0
    gb = grads[i] if has_beta else torch.zeros_like(beta)
    i += 1 if has_beta else 0
    gi = grads[i]
    return [gq, gk, gv, gg.contiguous(), gb.contiguous(), gi.contiguous()]


@_k2_bwd.register_fake
def _(grad_output, grad_final, query, key, value, log_decay, beta, initial_state,
      scale, decay_granularity, is_delta, read_before, has_log_decay, has_beta):
    return [
        torch.empty_like(query), torch.empty_like(key), torch.empty_like(value),
        torch.empty_like(log_decay), torch.empty_like(beta), torch.empty_like(initial_state),
    ]


def _setup(ctx, inputs, output):
    (query, key, value, log_decay, beta, initial_state, scale, decay_granularity,
     is_delta, read_before) = inputs
    has_log_decay = log_decay is not None
    has_beta = beta is not None
    # Save tensors (empty stand-ins for absent optionals so the bwd op has fixed schema).
    ctx.save_for_backward(
        query, key, value,
        log_decay if has_log_decay else torch.empty(0, device=query.device),
        beta if has_beta else torch.empty(0, device=query.device),
        initial_state,
    )
    ctx.knobs = dict(scale=scale, decay_granularity=decay_granularity,
                     is_delta=is_delta, read_before=read_before,
                     has_log_decay=has_log_decay, has_beta=has_beta)


def _autograd_backward(ctx, grad_output, grad_final):
    query, key, value, log_decay, beta, initial_state = ctx.saved_tensors
    k = ctx.knobs
    gq, gk, gv, gg, gb, gi = torch.ops.urm.k2_recurrence_bwd(
        grad_output, grad_final, query, key, value, log_decay, beta, initial_state,
        k["scale"], k["decay_granularity"], k["is_delta"], k["read_before"],
        k["has_log_decay"], k["has_beta"],
    )
    # Return None for the absent optional operands and the four scalar knobs.
    return (gq, gk, gv,
            gg if k["has_log_decay"] else None,
            gb if k["has_beta"] else None,
            gi, None, None, None, None)


_k2_fwd.register_autograd(_autograd_backward, setup_context=_setup)


def k2_recurrence(query, key, value, log_decay, beta, initial_state, *,
                  scale, decay_granularity, is_delta, read_before):
    """The compile-opaque native K2 recurrence: a single fused node in a compiled graph."""
    return torch.ops.urm.k2_recurrence_fwd(
        query, key, value, log_decay, beta, initial_state,
        scale, decay_granularity, is_delta, read_before,
    )


__all__ = ["k2_recurrence"]
