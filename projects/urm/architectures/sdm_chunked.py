"""External PyTorch schedule for the decayed sparse-delta memory law.

Routes come from the public URM product-key operator. This module changes the
execution schedule, not routing or the recurrence. Unlike the historical
sdm-reparam implementation, decay participates in both forward and backward.
The memory argument is a frozen snapshot; the returned state is a new tensor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_chunk_size(write_indices, log_decay, slots, requested):
    """Bound exponential factors using actual per-slot decay, not token count.

    This one scalar synchronization is outside the compiled numerical graph.
    It chooses an equivalent schedule and never detaches a numerical operand.
    """
    p, t, w = write_indices.shape
    size = min(requested, t)
    detached = log_decay.detach().float()
    while size > 1:
        n = (t + size - 1) // size
        pad = n * size - t
        addresses = F.pad(write_indices.long(), (0, 0, 0, pad)).reshape(p, n, size * w)
        decay = F.pad(detached.expand(p, t, w), (0, 0, 0, pad)).reshape(p, n, size * w)
        sums = torch.zeros(p, n, slots, device=decay.device).scatter_add(-1, addresses, decay)
        if float(sums.abs().max()) <= 60:
            break
        size = max(1, size // 2)
    return size


def _chunked(memory, ri, rw, wi, ww, values, beta, log_decay, chunk_size,
             zero_initial_state=False):
    p, t, d = values.shape
    slots = memory.shape[1]
    c = min(chunk_size, t)
    n = (t + c - 1) // c
    pad = n * c - t
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32

    def chunks(x):
        return F.pad(x, (0, 0, 0, pad)).reshape(p, n, c, x.shape[-1])

    ri, wi = chunks(ri.long()), chunks(wi.long())
    rw, ww, v, b, g = [chunks(x.to(dtype)) for x in (rw, ww, values, beta, log_decay)]
    slot_decay = torch.zeros(p, n, c, slots, device=values.device, dtype=dtype).scatter(
        -1, wi, g.expand_as(ww))
    cumul = slot_decay.cumsum(-2)
    end = cumul[..., -1:, :]
    write_cumul = cumul.gather(-1, wi)
    read_cumul = cumul.gather(-1, ri)
    write_end = end.expand(p, n, c, slots).gather(-1, wi)
    read_end = end.expand(p, n, c, slots).gather(-1, ri)
    dot_dtype = torch.bfloat16 if values.dtype == torch.bfloat16 else dtype

    def dense(indices, weights):
        return torch.zeros(p, n, c, slots, device=values.device, dtype=dot_dtype).scatter_add(
            -1, indices, weights.to(dot_dtype))

    # Exponentials and their saved backward tensors stay sparse. Only the
    # matmul operands expand to slot vectors, already in the compute dtype.
    plus = dense(wi, ww * (write_cumul - write_end * .5).exp())
    minus = dense(wi, ww * (write_end * .5 - write_cumul).exp())
    qplus = dense(ri, rw * (read_cumul - read_end * .5).exp())

    def mm(a, b):
        return (a.to(dot_dtype) @ b.to(dot_dtype)).to(dtype)

    strict_upper = torch.ones(c, c, device=values.device, dtype=torch.bool).triu(0)
    coupling = mm(plus, minus.transpose(-1, -2)).masked_fill(strict_upper, 0.)
    attention = mm(qplus, minus.transpose(-1, -2)).tril()
    eye = torch.eye(c, device=values.device, dtype=dtype).expand(p, n, c, c)
    system = eye + b * coupling
    initial_w = dense(wi, ww * write_cumul.exp())
    initial_q = dense(ri, rw * read_cumul.exp())
    final_w = dense(wi, ww * (write_end - write_cumul).exp())
    state = memory.to(dtype)
    outputs = []
    for i in range(n):
        if i == 0 and zero_initial_state:
            rhs = b[:, i] * v[:, i]
            base = torch.zeros_like(v[:, i])
        else:
            retrieved = mm(initial_w[:, i], state)
            base = mm(initial_q[:, i], state)
            rhs = b[:, i] * (v[:, i] - retrieved)
        delta = torch.linalg.solve_triangular(system[:, i], rhs,
                                             upper=False, unitriangular=True)
        outputs.append(base + mm(attention[:, i], delta))
        update = mm(final_w[:, i].transpose(-1, -2), delta)
        state = update if i == 0 and zero_initial_state else (
            cumul[:, i, -1, :, None].exp() * state + update
        )
    return torch.cat(outputs, dim=1)[:, :t].to(values.dtype), state.to(memory.dtype)


_compiled_chunked = torch.compile(_chunked, fullgraph=True, dynamic=False)


def _tokenwise(memory, ri, rw, wi, ww, values, beta, log_decay):
    """Stable limiting schedule for decay too strong for exponential factoring."""
    p, t, d = values.shape
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    state = memory.to(dtype)
    outputs = []
    for token in range(t):
        addresses = wi[:, token].long().unsqueeze(-1).expand(p, -1, d)
        decayed = state.gather(1, addresses) * log_decay[:, token].to(dtype).exp().unsqueeze(-1)
        weight = ww[:, token].to(dtype).unsqueeze(-1)
        retrieved = (weight * decayed).sum(1)
        delta = beta[:, token].to(dtype) * (values[:, token].to(dtype) - retrieved)
        state = state.scatter(1, addresses, decayed + weight * delta.unsqueeze(1))
        reads = ri[:, token].long().unsqueeze(-1).expand(p, -1, d)
        outputs.append((state.gather(1, reads) * rw[:, token].to(dtype).unsqueeze(-1)).sum(1))
    return torch.stack(outputs, 1).to(values.dtype), state.to(memory.dtype)


def chunked_sparse_delta_memory(memory, read_indices, read_weights, *,
                                write_indices, write_weights, values, beta,
                                log_decay, chunk_size=256, compiled=False,
                                zero_initial_state=False):
    """After-update reads with ordered cross-token collisions and unique writes.

    Indices must be valid partition-local routes; writes within each token must
    be unique. Gates have shape [P,T,1] and log_decay is nonpositive. FP64 inputs
    provide an exact algebraic test path; BF16 uses Tensor Core matmuls with
    FP32 solves/decay and rounds the returned readings and state to their input
    dtypes. It does not reproduce per-token BF16 storage rounding.

    zero_initial_state is an explicit lifecycle fact, valid only for a zero
    snapshot that does not require gradients. It is never inferred from values.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if zero_initial_state and memory.requires_grad:
        raise ValueError("zero_initial_state cannot discard initial-memory gradients")
    size = _safe_chunk_size(write_indices, log_decay, memory.shape[1], chunk_size)
    if size == 1:
        return _tokenwise(memory, read_indices, read_weights, write_indices, write_weights,
                          values, beta, log_decay)
    call = _compiled_chunked if compiled else _chunked
    with torch.autocast(values.device.type, enabled=False):
        return call(memory, read_indices, read_weights, write_indices, write_weights,
                    values, beta, log_decay, size, zero_initial_state)


__all__ = ["chunked_sparse_delta_memory"]
