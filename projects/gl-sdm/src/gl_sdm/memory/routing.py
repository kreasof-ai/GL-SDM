"""Chunk product-key routing, row selection and native URM read dispatch.

Scores use [batch, token, head, factor] or [batch, reasoning, token, head, factor].
Equal scores prefer larger addresses, matching the frozen URM route contract.
"""
import torch
import torch.nn.functional as F


def torch_routed_read(memory, scores, width, reference=False):
    """Independent highest-address-tie product-key equation."""
    B, T, H, L = scores.shape
    half = L // 2
    factors = scores.reshape(B, T, H, 2, half)
    # Reverse address order before stable descending score ranking.
    order = factors.flip(-1).argsort(dim=-1, descending=True, stable=True)[..., :width]
    order = half - 1 - order
    top = factors.gather(-1, order)
    score = (top[..., 0, :, None] + top[..., 1, None, :]).flatten(-2)
    address = (order[..., 0, :, None] * half + order[..., 1, None, :]).flatten(-2)
    by_address = address.argsort(dim=-1, descending=True, stable=True)
    address, score = address.gather(-1, by_address), score.gather(-1, by_address)
    chosen = score.argsort(dim=-1, descending=True, stable=True)[..., :width]
    address, score = address.gather(-1, chosen), score.gather(-1, chosen)
    canonical = address.argsort(-1)
    address, weights = address.gather(-1, canonical), score.gather(-1, canonical).softmax(-1)
    rows = gather_rows(memory, address, reference)
    return (rows * weights[..., None]).sum(-2), address, weights


def absolute_addresses(memory, index):
    B, H, S, _ = memory.shape
    partitions = torch.arange(B * H, device=memory.device).view(B, 1, H, 1)
    return partitions * S + index.long()


def gather_rows(memory, index, reference=False):
    flat = memory.flatten(0, 2)
    address = absolute_addresses(memory, index)
    if reference:
        return F.one_hot(address, flat.shape[0]).float() @ flat
    return flat[address]


def routed_read(block, memory, scores, width, physical=None):
    if block.attn.backend == "urm" and not block.attn.reference:
        from gl_sdm.memory.backends.urm import routed_snapshot_read
        if scores.ndim == 5:
            group = max(1, 2048 // scores.shape[2])
            parts = []
            for r in range(0, scores.shape[1], group):
                piece = scores[:, r:r + group]
                result = routed_read(block, memory, piece.flatten(1, 2), width, physical)
                parts.append(tuple(t.reshape(*piece.shape[:-1], t.shape[-1]) for t in result))
            return tuple(torch.cat([p[j] for p in parts], 1) for j in range(3))
        output, index, weights = routed_snapshot_read(memory if physical is None else physical, scores, width,
            allow_large_route=getattr(block.attn, "experimental_large_route", False))
        output = output[..., :memory.shape[-1]]
    else:
        shape = scores.shape
        scores = scores.reshape(shape[0], -1, *shape[-2:])
        # Vectorized PyTorch oracle; dense one-hot fixtures are tested separately.
        output, index, weights = torch_routed_read(memory, scores, width)
        if len(shape) == 5:
            output, index, weights = (t.reshape(*shape[:-1], t.shape[-1]) for t in (output, index, weights))
    return output, index, weights
