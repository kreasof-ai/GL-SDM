"""Small, differentiable equations used only by explicit correctness checks."""
import torch
import torch.nn.functional as F


def sdm(memory, ki, kw, v, beta, g, qi, qw):
    """Selected-slot decay, delta write, then read; addresses are global."""
    state = memory.float()
    outputs = []
    for t in range(v.shape[1]):
        keys = F.one_hot(ki[:, t].long(), state.shape[0]).float()
        k = (keys * kw[:, t].float().unsqueeze(-1)).sum(1)
        touched = keys.sum(1).clamp(max=1)
        decay = (touched * g[:, t].float()).sum(0).exp()
        state = state * decay.unsqueeze(-1)
        retrieved = k @ state
        delta = beta[:, t].float() * (v[:, t].float() - retrieved)
        state = state + k.T @ delta
        queries = F.one_hot(qi[:, t].long(), state.shape[0]).float()
        q = (queries * qw[:, t].float().unsqueeze(-1)).sum(1)
        outputs.append(q @ state)
    return torch.stack(outputs, 1).to(v.dtype), state


def gdn2(q, k, v, g, b, w, initial_state=None, output_final_state=False,
         use_qk_l2norm_in_kernel=False, cu_seqlens=None, scale=None, **kwargs):
    if cu_seqlens is not None:
        raise ValueError("reference check uses fixed-length batches")
    dtype = v.dtype
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    if use_qk_l2norm_in_kernel:
        q, k = (F.normalize(x, dim=-1, eps=1e-6) for x in (q, k))
    q = q * (scale if scale is not None else q.shape[-1] ** -0.5)
    state = (q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
             if initial_state is None else initial_state.float())
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp().unsqueeze(-1)
        erase = ((b[:, t] * k[:, t]).unsqueeze(-1) * state).sum(-2)
        state = state + k[:, t].unsqueeze(-1) * (w[:, t] * v[:, t] - erase).unsqueeze(-2)
        out.append((q[:, t].unsqueeze(-1) * state).sum(-2))
    return torch.stack(out, 1).to(dtype), state if output_final_state else None
