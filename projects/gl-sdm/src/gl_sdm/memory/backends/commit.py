"""Project-owned deterministic sparse commit; URM does not lower this operation.

Each address has one writer. Colliding deltas are summed in canonical order;
backward gathers the output gradient for each proposal. No forward atomics.
"""
import torch
import triton
import triton.language as tl
from .primitives import require_cuda, _gather


@triton.jit
def _commit_forward(memory, indices, order, deltas, output, E: tl.constexpr,
                    D: tl.constexpr, BD: tl.constexpr):
    position = tl.program_id(0)
    entry = tl.load(order + position)
    addr = tl.load(indices + entry)
    prev_entry = tl.load(order + position - 1, position > 0, other=0)
    prev = tl.load(indices + prev_entry)
    if (position == 0) | (addr != prev):
        d = tl.arange(0, BD)
        accumulator = tl.full((BD,), 0.0, tl.float32)
        cursor = position
        same_address = cursor < E
        while same_address:
            e = tl.load(order + cursor)
            accumulator += tl.load(deltas + e * D + d, d < D, other=0.0)
            cursor += 1
            next_entry = tl.load(order + cursor, cursor < E, other=0)
            next_addr = tl.load(indices + next_entry)
            same_address = (cursor < E) & (next_addr == addr)
        old = tl.load(memory + addr * D + d, d < D, other=0.0)
        tl.store(output + addr * D + d, old + accumulator, d < D)


class _Commit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, indices, deltas, keys):
        require_cuda(memory)
        order = keys.argsort(stable=True)
        output = memory.clone()
        _commit_forward[(indices.numel(),)](memory, indices, order, deltas, output, indices.numel(),
                                            memory.shape[-1], triton.next_power_of_2(memory.shape[-1]), enable_fp_fusion=False)
        ctx.save_for_backward(indices)
        ctx.dim = memory.shape[-1]
        return output

    @staticmethod
    def backward(ctx, incoming):
        (indices,) = ctx.saved_tensors
        incoming = incoming.contiguous()
        gd = torch.empty((indices.numel(), ctx.dim), device=incoming.device, dtype=torch.float32)
        _gather[(indices.numel(),)](incoming, indices, gd, ctx.dim, triton.next_power_of_2(ctx.dim))
        return incoming, None, gd, None


def commit(memory, indices, deltas, keys=None):
    return _Commit.apply(memory, indices, deltas, indices if keys is None else keys)
