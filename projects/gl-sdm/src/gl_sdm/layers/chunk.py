"""Chunk schedule: local attention -> tied global reasoning -> boundary commit.

All tokens in a chunk read one immutable bank. Local causal attention supplies
within-chunk context once before reasoning; it is not interleaved with global
passes. Pending proposals and local KV cache survive partial serving calls.
"""
import torch
import torch.nn.functional as F
from gl_sdm.memory import merge, commit
from gl_sdm.memory import routing
from gl_sdm.memory.chunk_writes import adaptive_proposal, fixed_proposal
from gl_sdm.runtime.dense import ChunkOps
from .regularization import sigreg


def forward(block, inputs, cache=None):
    B, T, D = inputs.shape
    if B < 1 or T < 1:
        raise ValueError("GL-SDM requires a nonempty token batch")
    retain_state = cache is not None
    if cache is None:
        cache = block.bank.new_cache(B)
    else:
        block.bank.validate_cache(cache, B)
    outputs, depths, ponders = [], [], []
    C, R = block.chunk_size, block.max_steps
    scale = R ** -0.5
    ops = ChunkOps(block, inputs)
    router = block.attn
    position = 0
    while position < T:
        offset = cache.tokens % C
        length = min(C - offset, T - position)
        snapshot = cache.view
        # Select frozen URM's existing vector schedule through zero channels;
        # the logical memory, retrieved values and gradients retain their width.
        physical = F.pad(snapshot.values, (0, 64)) if router.head_dim == 64 and router.backend == "urm" and not router.reference else None
        # A terminal training chunk has no consumer of its outgoing state.
        # Serving always retains proposals, including an unfinished chunk.
        writes_needed = retain_state or position + length < T
        # Local attention runs once, before all global reasoning passes.
        fragment = inputs[:, position:position + length]
        x = fragment + block.local_context(block.local_norm(fragment), cache.local)
        h = x.reshape(-1, D)
        condition = block.input_proj(h) * scale
        accumulated = torch.zeros(B * length, device=inputs.device)
        weighted = torch.zeros_like(h)
        active = torch.ones(B * length, device=inputs.device, dtype=torch.bool)
        depth = torch.full((B * length,), R if block.halt is None else 0, device=inputs.device, dtype=torch.int64)
        remainder = torch.zeros_like(accumulated)
        proposals = list(cache.pending)
        history = []
        # Global passes share a frozen bank; only adaptive dense work compacts.
        for r in range(R):
            chosen = None if block.halt is None else active.nonzero().flatten()
            if chosen is not None and chosen.numel() == 0:
                break
            current = h if chosen is None else h[chosen]
            normalized = block.norm1(current)
            q = router.q(normalized).float().view(-1, router.heads, 2 * router.half)
            if chosen is not None:
                q = q.new_zeros(B * length, router.heads, 2 * router.half).index_copy(0, chosen, q)
            readings, _, _ = routing.routed_read(block, snapshot.values, q.view(B, length, router.heads, -1), router.num_reads, physical)
            readings = readings.reshape(B * length, router.heads, router.head_dim)
            if chosen is not None:
                readings = readings[chosen]
            updated, normalized_write, p = ops.step(current,
                condition if chosen is None else condition[chosen], readings, scale)
            if chosen is None:
                h = updated
                mass = accumulated.new_full((B * length,), 1 / R)
                stopped = torch.full_like(mass, r == R - 1, dtype=torch.bool)
            else:
                h = h.index_copy(0, chosen, updated)
                depth = depth.index_add(0, chosen, torch.ones_like(chosen))
                stopped = (accumulated[chosen] + p >= 1 - block.halt_epsilon) | (r == R - 1)
                mass = torch.where(stopped, 1 - accumulated[chosen], p)
                accumulated = accumulated.index_add(0, chosen, mass)
                weighted = weighted.index_add(0, chosen, (updated.float() * mass[:, None]).to(h.dtype))
                remainder = remainder.index_copy(0, chosen, torch.where(stopped, mass, remainder[chosen]))
                active = active.index_copy(0, chosen, ~stopped)
            if not writes_needed:
                continue
            if chosen is None:
                history.append(normalized_write.view(B, length, D))
                continue
            proposals.append(adaptive_proposal(block, snapshot, normalized_write,
                chosen, mass, stopped, offset, length, r, physical, ops))
        if history:
            proposals.append(fixed_proposal(block, snapshot, history,
                offset, length, physical, ops))
        # Proposals become visible only at the absolute chunk boundary.
        cache.tokens += length
        if cache.tokens % C == 0 and writes_needed:
            cache.view = commit(snapshot, merge(snapshot, proposals), backend="torch" if router.reference else router.backend)
            cache.pending, cache.local = (), {}
        else:
            cache.pending = tuple(proposals)
        outputs.append((h if block.halt is None else weighted).view(B, length, D))
        depths.append(depth.view(B, length))
        ponders.append((depth.float() + remainder).sum())
        position += length
    hidden = torch.cat(outputs, 1)
    block.last_depth = torch.cat(depths, 1).detach()
    auxiliary = sum(ponders) if block.halt is not None else hidden.new_zeros(())
    return hidden, sigreg(hidden, block.reg_mode, block.sketch_dim), auxiliary
