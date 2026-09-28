"""Build weighted deltas against a frozen chunk snapshot.

Fixed-depth writes batch all reasoning passes. Adaptive writes are collected
from active tokens each pass. Both order collisions by address, token and depth.
"""
import torch
from .state import WriteProposal
from . import routing


def adaptive_proposal(block, snapshot, normalized_write, chosen, mass, stopped,
                      offset, length, r, physical, ops):
    router = block.attn
    B = snapshot.values.shape[0]
    C, R = block.chunk_size, block.max_steps
    k, target, beta, decay = ops.write(normalized_write)
    k = k.new_zeros(B * length, router.heads, 2 * router.half).index_copy(0, chosen, k)
    prediction, index, weights = routing.routed_read(block, snapshot.values,
        k.view(B, length, router.heads, -1), router.num_writes, physical)
    prediction = prediction.reshape(B * length, router.heads, router.head_dim)[chosen]
    weights = weights.reshape(B * length, router.heads, router.num_writes)[chosen]
    rows = routing.gather_rows(snapshot.values, index).reshape(B * length, router.heads, router.num_writes, router.head_dim)[chosen]
    addresses = routing.absolute_addresses(snapshot.values, index).reshape(B * length, router.heads, router.num_writes)[chosen]
    token = chosen % length + offset
    write_mass = (stopped.float() if block.write_policy == "final" else mass) / C
    delta = ops.delta(rows, weights, prediction, target, beta, decay, write_mass)
    keys = addresses * (C * R) + token[:, None, None] * R + r
    return WriteProposal(snapshot.version, addresses.flatten(),
        delta.flatten(0, 2), snapshot.lineage, keys.flatten())


def fixed_proposal(block, snapshot, history, offset, length, physical, ops):
    router = block.attn
    B = snapshot.values.shape[0]
    C, R = block.chunk_size, block.max_steps
    z = torch.stack(history, 1)
    k, target, beta, decay = ops.write(z)
    prediction, index, weights = routing.routed_read(block, snapshot.values, k, router.num_writes, physical)
    rows = routing.gather_rows(snapshot.values, index.reshape(B, -1, router.heads, router.num_writes))
    rows = rows.reshape(B, R, length, router.heads, router.num_writes, router.head_dim)
    addresses = routing.absolute_addresses(snapshot.values, index.reshape(B, -1, router.heads, router.num_writes)).reshape_as(index)
    mass = torch.full((B, R, length), 1 / (R * C), device=snapshot.values.device)
    if block.write_policy == "final":
        mass = mass * 0
        mass[:, -1] = 1 / C
    delta = ops.delta(rows, weights, prediction, target, beta, decay, mass)
    token = torch.arange(offset, offset + length, device=snapshot.values.device).view(1, 1, length, 1, 1)
    step = torch.arange(R, device=snapshot.values.device).view(1, R, 1, 1, 1)
    keys = addresses * (C * R) + token * R + step
    return WriteProposal(snapshot.version, addresses.flatten(), delta.reshape(-1, router.head_dim),
        snapshot.lineage, keys.flatten())
