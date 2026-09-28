"""Project-owned versioned sparse memory; all updates are functional.

A transaction is one token or chunk. Reads/proposals share its frozen view; weighted
row deltas are merged in proposal order and committed once. Duplicate addresses
use stable sorting and segmented sums, followed by unique-address index_copy.
There are no atomic scatter additions in the forward commit.
"""
import torch
import torch.nn.functional as F
from .state import MemoryView, WriteProposal, WriteBuffer


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
        from gl_sdm.memory.backends.token import read as native_read
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
        from gl_sdm.memory.backends.token import propose
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
        from gl_sdm.memory.backends.commit import commit as native_commit
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
