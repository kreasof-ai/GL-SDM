"""Project-owned versioned sparse memory; all updates are functional.

A transaction is one token or chunk. Reads/proposals share its frozen view; weighted
row deltas are merged in proposal order and committed once. Duplicate addresses
use stable sorting and segmented sums, followed by unique-address index_copy.
There are no atomic scatter additions in the forward commit.
"""
from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryView:
    values: torch.Tensor  # [requests, heads, slots, value_dim], FP32
    version: int = 0
    lineage: object = field(default_factory=object)

    def __post_init__(self):
        if self.values.ndim != 4 or self.values.dtype != torch.float32 or min(self.values.shape) < 1 or self.version < 0:
            raise ValueError("memory views require nonempty [batch, heads, slots, dim] FP32 values and a nonnegative version")


@dataclass(frozen=True)
class WriteProposal:
    base_version: int
    addresses: torch.Tensor  # flattened request/head/slot addresses
    deltas: torch.Tensor     # [entries, value_dim], already weighted
    lineage: object
    sort_keys: torch.Tensor | None = None  # optional address/token/depth order


@dataclass(frozen=True)
class WriteBuffer:
    base_version: int
    proposals: tuple[WriteProposal, ...]
    lineage: object


@dataclass
class MemoryCache:
    view: MemoryView
    owner: object
    tokens: int = 0
    pending: tuple[WriteProposal, ...] = ()
    local: dict = field(default_factory=dict)


def addresses(view, requests, indices):
    heads, slots = view.values.shape[1:3]
    partitions = requests[:, None, None] * heads + torch.arange(heads, device=indices.device)[None, :, None]
    return partitions * slots + indices


def selected(view, requests, indices, reference=False):
    flat = view.values.flatten(0, 2)
    index = addresses(view, requests, indices)
    if reference:
        # Explicit dense equation for small oracle fixtures only.
        return F.one_hot(index.long(), flat.shape[0]).float() @ flat
    return flat[index]


def read(view, requests, indices, weights, reference=False, backend="torch"):
    if backend not in {"torch", "urm"}:
        raise ValueError("memory backend must be torch or urm")
    if backend == "urm" and not reference:
        from .kernels import read as native_read
        return native_read(view.values, requests, indices, weights)
    rows = selected(view, requests, indices, reference)
    return (rows * weights.float().unsqueeze(-1)).sum(-2)


def propose_write(view, requests, indices, weights, targets, beta, log_decay,
                  mass, reference=False, backend="torch"):
    """One delta update evaluated against the transaction's immutable snapshot.

    Router addresses within a proposal are unique. Multiple proposals may target
    the same slot. Their weighted deltas are summed, not applied sequentially.
    Fixed-depth mass is 1/R; ACT mass sums to one per request/token.
    """
    if backend not in {"torch", "urm"}:
        raise ValueError("memory backend must be torch or urm")
    if backend == "urm" and not reference:
        from .kernels import propose
        idx, delta = propose(view.values, requests, indices, weights, targets, beta, log_decay, mass)
        return WriteProposal(view.version, idx, delta, view.lineage)
    rows = selected(view, requests, indices, reference)
    decay = log_decay.float().exp().unsqueeze(-1)
    decayed = rows * decay
    retrieved = (decayed * weights.float().unsqueeze(-1)).sum(-2)
    error = beta.float() * (targets.float() - retrieved)
    delta = decayed - rows + weights.float().unsqueeze(-1) * error.unsqueeze(-2)
    delta = delta * mass.float()[:, None, None, None]
    return WriteProposal(view.version, addresses(view, requests, indices).flatten(), delta.flatten(0, 2), view.lineage)


def merge(view, proposals):
    proposals = tuple(proposals)
    if not proposals or any(p.base_version != view.version or p.lineage is not view.lineage for p in proposals):
        raise ValueError("write buffers must be nonempty and use the same snapshot version")
    return WriteBuffer(view.version, proposals, view.lineage)


def commit(view, buffer, reference=False, backend="torch"):
    if backend not in {"torch", "urm"}:
        raise ValueError("memory backend must be torch or urm")
    if buffer.base_version != view.version or buffer.lineage is not view.lineage:
        raise ValueError("stale memory transaction")
    flat = view.values.flatten(0, 2)
    idx = torch.cat([p.addresses for p in buffer.proposals])
    delta = torch.cat([p.deltas for p in buffer.proposals]).float()
    keyed = [p.sort_keys is not None for p in buffer.proposals]
    if any(keyed) and not all(keyed):
        raise ValueError("cannot mix keyed and unkeyed proposals")
    keys = torch.cat([p.sort_keys for p in buffer.proposals]) if all(keyed) else idx
    if backend == "urm" and not reference:
        from .kernels import commit as native_commit
        updated = native_commit(view.values, idx, delta, keys)
        return MemoryView(updated, view.version + 1, view.lineage)
    if reference:
        # Independent dense reduction, including cross-step address collisions.
        update = F.one_hot(idx.long(), flat.shape[0]).float().T @ delta
        updated = flat + update
    else:
        order = keys.argsort(stable=True)
        idx, delta = idx[order], delta[order]
        unique, lengths = idx.unique_consecutive(return_counts=True)
        # Sum within each address. Subtracting a global cumulative sum would
        # introduce cancellation from unrelated requests/heads/slots.
        updates = torch.segment_reduce(delta, "sum", lengths=lengths)
        updated = flat.index_copy(0, unique, flat[unique] + updates)
    return MemoryView(updated.view_as(view.values), view.version + 1, view.lineage)


class MemoryBank(nn.Module):
    def __init__(self, heads, slots, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(heads, slots, dim) * (heads * dim) ** -0.5)
        self.memory._sdm_memory_bank = True
        self.owner = object()

    def _apply(self, fn, recurse=True):
        # Reasoner weights may be BF16; transactional state and its learned
        # initializer remain FP32, preserving the original initializer values.
        values = self.memory.detach()
        gradient = None if self.memory.grad is None else self.memory.grad.detach()
        super()._apply(fn, recurse)
        self.memory.data = values.to(device=self.memory.device, dtype=torch.float32)
        if gradient is not None:
            self.memory.grad.data = gradient.to(device=self.memory.device, dtype=torch.float32)
        return self

    def snapshot(self, batch_size):
        return MemoryView(self.memory.unsqueeze(0).expand(batch_size, -1, -1, -1).clone())

    def new_cache(self, batch_size):
        return MemoryCache(self.snapshot(batch_size), self.owner)

    def validate_cache(self, cache, batch_size):
        expected = (batch_size, *self.memory.shape)
        if cache.owner is not self.owner or cache.view.values.shape != expected or cache.view.values.device != self.memory.device or cache.view.values.dtype != torch.float32:
            raise ValueError("memory cache belongs to another model, batch, device or dtype")
