"""Numerical experiment only: selected-slot-decay triangular chunk algebra.

This is NOT a backend or compiler anchor. It deliberately exposes the proposed
real-arithmetic formulation's cast policy: readings and chunk boundary memory
are cast to the input dtype, but intermediate token states are not rounded.
It cannot be integrated unless it passes the unchanged per-token-cast contract.
The production recurrence and its oracle are untouched.
"""

from __future__ import annotations

from urm.compiler.semantic import SparseReadTiming


def torch_selected_slot_triangular(
    memory,
    read_indices,
    read_weights,
    *,
    write_indices=None,
    write_weights=None,
    values=None,
    beta=None,
    log_decay=None,
    chunk_size=16,
    read_timing=SparseReadTiming.AFTER_UPDATE,
    accumulation_dtype=None,
):
    """Transparent dense-support algebra, autograd through every floating input.

    Let k_t be the sparse write vector and E_t the diagonal decay (exp(g_t)
    on selected slots, 1 elsewhere). Then M_t = E_t M_(t-1) + k_t d_t,
    d_t = beta_t (v_t - k_t^T E_t M_(t-1)). With G_t = E_t ... E_1:

      (I + tril(beta_i k_i^T E_i...E_(j+1) k_j, -1)) d
          = beta * (v - k_i^T G_i M_boundary).

    Prefix *log* decays produce causal interval products without inverse
    decays or overflow from exp(-prefix). Chunk boundary propagation remains
    ordered. Dense slot intermediates are for inspection, not a GPU design.
    """
    import torch

    if chunk_size not in (16, 32, 64):
        raise ValueError("numerical prototype chunks are frozen to 16/32/64")
    accumulation_dtype = accumulation_dtype or torch.float32
    if write_indices is None:
        if read_timing is not SparseReadTiming.CURRENT_STATE:
            raise ValueError("read-only prototype requires CURRENT_STATE")
        p = torch.arange(memory.shape[0], device=memory.device)[:, None, None]
        readings = (
            memory[p, read_indices].to(accumulation_dtype)
            * read_weights.to(accumulation_dtype)[..., None]
        ).sum(-2)
        return readings.to(memory.dtype), memory.clone()
    if read_timing not in (
        SparseReadTiming.BEFORE_UPDATE,
        SparseReadTiming.AFTER_UPDATE,
    ):
        raise ValueError("updates require BEFORE_UPDATE or AFTER_UPDATE")
    if any(x is None for x in (write_weights, values, beta, log_decay)):
        raise ValueError("all differentiable update inputs are required")
    if bool((write_indices.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("within-token duplicate write slots are unsupported")
    if bool((log_decay > 0).any()):
        raise ValueError("log decay must be nonpositive")

    state = memory.clone()
    parallel, sequence, _ = read_indices.shape
    slots = memory.shape[1]
    outputs = []
    for start in range(0, sequence, chunk_size):
        stop = min(start + chunk_size, sequence)
        count = stop - start
        wi = write_indices[:, start:stop].long()
        ri = read_indices[:, start:stop].long()
        shape = (parallel, count, slots)
        k = torch.zeros(shape, device=memory.device, dtype=accumulation_dtype).scatter(
            -1, wi, write_weights[:, start:stop].to(accumulation_dtype)
        )
        q = torch.zeros_like(k).scatter_add(
            -1, ri, read_weights[:, start:stop].to(accumulation_dtype)
        )
        selected = torch.zeros_like(k).scatter(
            -1, wi, torch.ones_like(wi, dtype=accumulation_dtype)
        )
        prefix = (selected * log_decay[:, start:stop].to(accumulation_dtype)).cumsum(1)
        boundary = state.to(accumulation_dtype)
        baseline = (k * prefix.exp()) @ boundary
        # Each row is independent chunk-local preprocessing. The list avoids a
        # P*C*C*S*D intermediate and makes the selected-slot interval explicit.
        overlap_rows = []
        for i in range(count):
            interval = (prefix[:, i : i + 1] - prefix[:, :i]).exp()
            overlap = (k[:, i : i + 1] * interval * k[:, :i]).sum(-1)
            overlap_rows.append(torch.nn.functional.pad(overlap, (0, count - i)))
        overlap = torch.stack(overlap_rows, dim=1)
        b = beta[:, start:stop].to(accumulation_dtype)
        system = (
            torch.eye(count, device=memory.device, dtype=accumulation_dtype)
            + b * overlap
        )
        rhs = b * (values[:, start:stop].to(accumulation_dtype) - baseline)
        delta = torch.linalg.solve_triangular(
            system, rhs, upper=False, unitriangular=True
        )

        before = read_timing is SparseReadTiming.BEFORE_UPDATE
        read_prefix = (
            torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), 1)
            if before
            else prefix
        )
        read_base = (q * read_prefix.exp()) @ boundary
        read_rows = []
        for i in range(count):
            end = i if before else i + 1
            interval = (read_prefix[:, i : i + 1] - prefix[:, :end]).exp()
            overlap = (q[:, i : i + 1] * interval * k[:, :end]).sum(-1)
            read_rows.append(torch.nn.functional.pad(overlap, (0, count - end)))
        reading = read_base + torch.stack(read_rows, dim=1) @ delta
        outputs.append(reading.to(memory.dtype))
        final_coefficients = k * (prefix[:, -1:] - prefix).exp()
        state = (
            prefix[:, -1].exp()[..., None] * boundary
            + final_coefficients.transpose(1, 2) @ delta
        ).to(memory.dtype)
    return torch.cat(outputs, dim=1), state


__all__ = ["torch_selected_slot_triangular"]
