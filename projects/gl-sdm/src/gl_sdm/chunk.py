"""Causal chunk transactions using URM route/read forward and backward.

All reasoning within a chunk reads one immutable bank. A local causal SDPA
surround supplies within-chunk context; writes become visible at the next
absolute chunk boundary. This is a different write clock from token GL-SDM.
"""
import torch
import torch.nn.functional as F
from .memory import WriteProposal, merge, commit
from .regularization import sigreg
from .compilation import compiled


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
        from .urm_adapter import routed_snapshot_read
        if scores.ndim == 5:
            group = max(1, 2048 // scores.shape[2])
            parts = []
            for r in range(0, scores.shape[1], group):
                piece = scores[:, r:r + group]
                result = routed_read(block, memory, piece.flatten(1, 2), width, physical)
                parts.append(tuple(t.reshape(*piece.shape[:-1], t.shape[-1]) for t in result))
            return tuple(torch.cat([p[j] for p in parts], 1) for j in range(3))
        output, index, weights = routed_snapshot_read(memory if physical is None else physical, scores, width)
        output = output[..., :memory.shape[-1]]
    else:
        shape = scores.shape
        scores = scores.reshape(shape[0], -1, *shape[-2:])
        # Vectorized PyTorch oracle; dense one-hot fixtures are tested separately.
        output, index, weights = torch_routed_read(memory, scores, width)
        if len(shape) == 5:
            output, index, weights = (t.reshape(*shape[:-1], t.shape[-1]) for t in (output, index, weights))
    return output, index, weights


def proposal_delta(rows, weights, prediction, target, beta, log_decay, mass):
    decay = log_decay.exp()
    error = beta * (target.float() - decay * prediction)
    return mass[..., None, None, None] * ((decay[..., None] - 1) * rows + weights[..., None] * error[..., None, :])


def dense_step(current, condition, readings, scale, norm2, fcw, fcb, pw, pb,
               read_norm, ow, ob, norm1, hw, hb, head_dim):
    # Pure tensor arithmetic keeps model parameter names and checkpoint ABI.
    r = F.rms_norm(readings.to(current.dtype), (head_dim,), read_norm, eps=1e-6)
    updated = current + condition + F.linear(r.flatten(-2), ow, ob) * scale
    z = F.rms_norm(updated, (updated.shape[-1],), norm2, eps=1e-6)
    x, gate = F.linear(z, fcw, fcb).chunk(2, -1)
    updated = updated + F.linear(gate * x.relu().square(), pw, pb) * scale
    z = F.rms_norm(updated, (updated.shape[-1],), norm1, eps=1e-6)
    halt = F.linear(z, hw, hb).float().sigmoid().flatten() if hw is not None else None
    return updated, z, halt


def write_coefficients(z, weight, bias, heads, dim, score_width):
    pieces = F.linear(z, weight, bias).split([heads * score_width, heads * dim, heads, heads], -1)
    shape = z.shape[:-1]
    return (pieces[0].float().reshape(*shape, heads, score_width),
        pieces[1].reshape(*shape, heads, dim), pieces[2].float().sigmoid().unsqueeze(-1),
        -F.softplus(pieces[3].float() - 4).unsqueeze(-1))


_compiled_step = compiled(dense_step)
_compiled_delta = compiled(proposal_delta)
_compiled_write = compiled(write_coefficients)


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
    compiled = block.compile_dense and inputs.is_cuda and not block.attn.reference
    step_fn = _compiled_step if compiled else dense_step
    delta_fn = _compiled_delta if compiled else proposal_delta
    write_fn = _compiled_write if compiled else write_coefficients
    router = block.attn
    projections = [router.k, router.v, router.beta, router.decay]
    write_weight = torch.cat([p.weight for p in projections])
    write_bias = torch.cat([p.bias for p in projections])
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
        for r in range(R):
            chosen = None if block.halt is None else active.nonzero().flatten()
            if chosen is not None and chosen.numel() == 0:
                break
            current = h if chosen is None else h[chosen]
            normalized = block.norm1(current)
            q = router.q(normalized).float().view(-1, router.heads, 2 * router.half)
            if chosen is not None:
                q = q.new_zeros(B * length, router.heads, 2 * router.half).index_copy(0, chosen, q)
            readings, _, _ = routed_read(block, snapshot.values, q.view(B, length, router.heads, -1), router.num_reads, physical)
            readings = readings.reshape(B * length, router.heads, router.head_dim)
            if chosen is not None:
                readings = readings[chosen]
            updated, normalized_write, p = step_fn(current,
                condition if chosen is None else condition[chosen], readings, scale,
                block.norm2.weight, block.mlp.fc.weight, block.mlp.fc.bias,
                block.mlp.proj.weight, block.mlp.proj.bias, router.read_norm.weight,
                router.proj.weight, router.proj.bias, block.norm1.weight,
                None if block.halt is None else block.halt.weight,
                None if block.halt is None else block.halt.bias, router.head_dim)
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
            k, target, beta, decay = write_fn(normalized_write, write_weight, write_bias,
                router.heads, router.head_dim, 2 * router.half)
            k = k.new_zeros(B * length, router.heads, 2 * router.half).index_copy(0, chosen, k)
            prediction, index, weights = routed_read(block, snapshot.values,
                k.view(B, length, router.heads, -1), router.num_writes, physical)
            prediction = prediction.reshape(B * length, router.heads, router.head_dim)[chosen]
            weights = weights.reshape(B * length, router.heads, router.num_writes)[chosen]
            rows = gather_rows(snapshot.values, index).reshape(B * length, router.heads, router.num_writes, router.head_dim)[chosen]
            addresses = absolute_addresses(snapshot.values, index).reshape(B * length, router.heads, router.num_writes)[chosen]
            token = chosen % length + offset
            write_mass = (stopped.float() if block.write_policy == "final" else mass) / C
            delta = delta_fn(rows, weights, prediction, target, beta, decay, write_mass)
            keys = addresses * (C * R) + token[:, None, None] * R + r
            proposals.append(WriteProposal(snapshot.version, addresses.flatten(),
                delta.flatten(0, 2), snapshot.lineage, keys.flatten()))
        if history:
            z = torch.stack(history, 1)
            k, target, beta, decay = write_fn(z, write_weight, write_bias,
                router.heads, router.head_dim, 2 * router.half)
            prediction, index, weights = routed_read(block, snapshot.values, k, router.num_writes, physical)
            rows = gather_rows(snapshot.values, index.reshape(B, -1, router.heads, router.num_writes))
            rows = rows.reshape(B, R, length, router.heads, router.num_writes, router.head_dim)
            addresses = absolute_addresses(snapshot.values, index.reshape(B, -1, router.heads, router.num_writes)).reshape_as(index)
            mass = accumulated.new_full((B, R, length), 1 / (R * C))
            if block.write_policy == "final":
                mass = mass * 0
                mass[:, -1] = 1 / C
            delta = delta_fn(rows, weights, prediction, target, beta, decay, mass)
            token = torch.arange(offset, offset + length, device=inputs.device).view(1, 1, length, 1, 1)
            step = torch.arange(R, device=inputs.device).view(1, R, 1, 1, 1)
            keys = addresses * (C * R) + token * R + step
            proposals.append(WriteProposal(snapshot.version, addresses.flatten(), delta.reshape(-1, router.head_dim),
                snapshot.lineage, keys.flatten()))
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
