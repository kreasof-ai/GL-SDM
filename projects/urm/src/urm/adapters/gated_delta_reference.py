"""Reference recurrences for the frozen FLA gated delta-rule contract.

These are the level-1 semantic oracle and the level-2 transparent eager
PyTorch baseline for docs/fla-gated-delta-rule.md. Both implement exactly:

    u_t  = S^T k_t
    dv_t = beta_t * (v_t - u_t)
    S_t  = exp(g_t) * S + k_t dv_t^T     (decay before update)
    o_t  = scale * S_t^T q_t             (post-update output)

The oracle detaches its inputs and exists for clarity and correctness only;
the eager baseline is differentiable and serves as the fp32 framework
reference for gradient checks. Both are O(T) Python loops and are never used
as performance baselines.
"""

from __future__ import annotations

import torch


def _reference_step(state, k_t, v_t, g_t, beta_t):
    """One gated delta-rule step shared verbatim by both reference levels."""
    state = state * torch.exp(g_t.float()).unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhk,bhkv->bhv", k_t.float(), state)
    delta = (v_t.float() - retrieved) * beta_t.float().unsqueeze(-1)
    state = state + torch.einsum("bhk,bhv->bhkv", k_t.float(), delta)
    return state


def _reference_loop(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False
):
    b, t, h, k_dim = q.shape
    hv = v.shape[2]
    resolved_scale = scale if scale is not None else k_dim**-0.5
    # GVA: value heads are grouped; q/k heads expand to HV via
    # repeat_interleave exactly as the upstream kernels do.
    if hv != h:
        groups = hv // h
        q = q.repeat_interleave(groups, dim=2)
        k = k.repeat_interleave(groups, dim=2)
    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = torch.zeros(
            b, hv, k_dim, v.shape[-1], dtype=torch.float32, device=q.device
        )
    outputs = []
    for index in range(t):
        state = _reference_step(
            state, k[:, index], v[:, index], g[:, index], beta[:, index]
        )
        outputs.append(
            resolved_scale * torch.einsum("bhk,bhkv->bhv", q[:, index].float(), state)
        )
    output = torch.stack(outputs, dim=1)
    return output.to(v.dtype), (state if output_final_state else None)


def oracle_gated_delta_rule(
    q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=False
):
    """Level 1: explicit fp32 recurrence loop (detached, clarity first)."""
    detached_inputs = [
        tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in (q, k, v, g, beta)
    ]
    return _reference_loop(
        *detached_inputs,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def eager_gated_delta_rule(
    q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=False
):
    """Level 2: transparent eager PyTorch recurrence (differentiable)."""
    return _reference_loop(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
