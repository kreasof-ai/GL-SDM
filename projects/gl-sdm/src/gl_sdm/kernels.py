"""GL-SDM CUDA operators; URM supplies the compiled snapshot read.

Training saves selected rows, not whole bank versions. Forward commits have
one owner per address and sum collisions in proposal order without atomics.
The backward of each sparse operation writes unique request/head/slot rows.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .urm_adapter import snapshot_read


def require_cuda(values):
    if not values.is_cuda or not values.is_contiguous():
        raise ValueError("GL-SDM native kernels require contiguous CUDA tensors")
    if torch.cuda.get_device_capability(values.device) < (8, 0):
        raise ValueError("GL-SDM native kernels require SM80 or newer")


def validate_access(values, requests, indices, weights):
    """Structural checks; the router guarantees valid, unique local slots.

    Native reads additionally require ascending addresses. Avoid scanning route
    values or synchronizing the GPU in the trusted production path.
    """
    require_cuda(values)
    if indices.ndim != 3 or indices.shape[1] != values.shape[1] or requests.shape != (indices.shape[0],) or weights.shape != indices.shape:
        raise ValueError("native routes require [active requests, heads, width] indices/weights")
    if requests.dtype != torch.int64 or indices.dtype != torch.int64:
        raise ValueError("native request and slot indices require int64")
    if any(t.device != values.device or not t.is_contiguous() for t in (requests, indices, weights)):
        raise ValueError("native operands must be contiguous and share the memory device")


@triton.jit
def _score_key(value, address):
    # Torch treats +0 and -0 as equal; smaller addresses win score ties.
    value = tl.where(value == 0, 0.0, value)
    bits = value.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits >> 31 != 0, ~bits, bits ^ 0x80000000).to(tl.uint64)
    return (ordered << 32) | (0xFFFFFFFF - address.to(tl.uint64))


@triton.jit
def _route_forward(scores, indices, weights, HALF: tl.constexpr, WIDTH: tl.constexpr,
                   BH: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, BH)
    left = tl.load(scores + row * 2 * HALF + h, h < HALF, other=-float("inf"))
    right = tl.load(scores + row * 2 * HALF + HALF + h, h < HALF, other=-float("inf"))
    left_sorted = tl.sort(_score_key(left, h), descending=True)
    right_sorted = tl.sort(_score_key(right, h), descending=True)
    k = tl.arange(0, BK)
    li = (0xFFFFFFFF - (tl.gather(left_sorted, k, 0) & 0xFFFFFFFF)).to(tl.int32)
    ri = (0xFFFFFFFF - (tl.gather(right_sorted, k, 0) & 0xFFFFFFFF)).to(tl.int32)
    ls = tl.load(scores + row * 2 * HALF + li, k < tl.minimum(WIDTH, HALF), other=0.0)
    rs = tl.load(scores + row * 2 * HALF + HALF + ri, k < tl.minimum(WIDTH, HALF), other=0.0)
    addr = (li[:, None] * HALF + ri[None, :]).reshape(BC)
    sums = (ls[:, None] + rs[None, :]).reshape(BC)
    c = tl.arange(0, BC)
    valid = (c // BK < tl.minimum(WIDTH, HALF)) & (c % BK < tl.minimum(WIDTH, HALF))
    ranked = tl.sort(_score_key(tl.where(valid, sums, -float("inf")), addr), descending=True)
    w = tl.arange(0, BW)
    chosen = (0xFFFFFFFF - (tl.gather(ranked, w, 0) & 0xFFFFFFFF)).to(tl.int32)
    # Canonical ascending addresses satisfy URM's supplied-route contract.
    chosen = tl.sort(tl.where(w < WIDTH, chosen, 0x7FFFFFFF), descending=False)
    logits = tl.load(scores + row * 2 * HALF + chosen // HALF, w < WIDTH, other=-float("inf"))
    logits += tl.load(scores + row * 2 * HALF + HALF + chosen % HALF, w < WIDTH, other=0.0)
    exp = libdevice.exp(logits - tl.max(logits, 0))
    prob = exp / tl.sum(exp, 0)
    tl.store(indices + row * WIDTH + w, chosen, w < WIDTH)
    tl.store(weights + row * WIDTH + w, prob, w < WIDTH)


@triton.jit
def _route_backward(indices, weights, incoming, output, HALF: tl.constexpr,
                    WIDTH: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    row = tl.program_id(0)
    w = tl.arange(0, BW)
    idx = tl.load(indices + row * WIDTH + w, w < WIDTH, other=-1)
    prob = tl.load(weights + row * WIDTH + w, w < WIDTH, other=0.0)
    grad = tl.load(incoming + row * WIDTH + w, w < WIDTH, other=0.0)
    g = prob * (grad - tl.sum(prob * grad, 0))
    h = tl.arange(0, BH)
    left = tl.sum(tl.where((h[:, None] == idx[None, :] // HALF) & (w[None, :] < WIDTH), g[None, :], 0.0), 1)
    right = tl.sum(tl.where((h[:, None] == idx[None, :] % HALF) & (w[None, :] < WIDTH), g[None, :], 0.0), 1)
    tl.store(output + row * 2 * HALF + h, left, h < HALF)
    tl.store(output + row * 2 * HALF + HALF + h, right, h < HALF)


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, count):
        require_cuda(scores)
        half = scores.shape[-1] // 2
        if half > 256 or count > 64 or not 1 <= count <= half * half:
            raise ValueError("GL-SDM native routing supports factor extent <=256 and route width <=64")
        indices = torch.empty((*scores.shape[:-1], count), dtype=torch.int64, device=scores.device)
        weights = torch.empty_like(indices, dtype=torch.float32)
        bk, bw = triton.next_power_of_2(min(count, half)), triton.next_power_of_2(count)
        _route_forward[(scores.numel() // (2 * half),)](
            scores, indices, weights, half, count, triton.next_power_of_2(half), bk, bk * bk, bw,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(indices, weights)
        ctx.half = half
        return indices, weights

    @staticmethod
    def backward(ctx, _, incoming):
        indices, weights = ctx.saved_tensors
        output = torch.empty((*weights.shape[:-1], 2 * ctx.half), dtype=torch.float32, device=weights.device)
        _route_backward[(weights.numel() // weights.shape[-1],)](
            indices, weights, incoming.contiguous(), output, ctx.half, weights.shape[-1],
            triton.next_power_of_2(ctx.half), triton.next_power_of_2(weights.shape[-1]),
            enable_fp_fusion=False)
        return output, None


def route(scores, count):
    return _Route.apply(scores.float().contiguous(), count)


@triton.jit
def _addresses(requests, indices, output, HEADS: tl.constexpr, SLOTS: tl.constexpr,
               WIDTH: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + i // (HEADS * WIDTH), i < N, other=0)
    slot = tl.load(indices + i, i < N, other=0)
    address = (request * HEADS + (i // WIDTH) % HEADS) * SLOTS + slot
    tl.store(output + i, address, i < N)


def absolute_addresses(values, requests, indices):
    result = torch.empty_like(indices)
    _addresses[(triton.cdiv(indices.numel(), 128),)](
        requests, indices, result, values.shape[1], values.shape[2], indices.shape[-1], indices.numel(), 128)
    return result


@triton.jit
def _gather(values, indices, output, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    idx = tl.load(indices + row)
    value = tl.load(values + idx * D + d, d < D, other=0.0)
    tl.store(output + row * D + d, value, d < D)


@triton.jit
def _read_backward(rows, addresses, weights, incoming, gm, gw,
                   K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    idx = tl.load(addresses + row)
    w = tl.load(weights + row)
    g = tl.load(incoming + (row // K) * D + d, d < D, other=0.0)
    x = tl.load(rows + row * D + d, d < D, other=0.0)
    tl.store(gm + idx * D + d, w * g, d < D)
    tl.store(gw + row, tl.sum(g * x, 0))


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, addresses, weights):
        require_cuda(values)
        output = snapshot_read(values, addresses, weights)
        # URM's ordinary autograd read saves the entire bank. This wrapper
        # retains only selected rows so causal tokens do not retain T banks.
        rows = torch.empty((*weights.shape, values.shape[-1]), device=values.device, dtype=torch.float32)
        _gather[(addresses.numel(),)](values, addresses, rows, values.shape[-1], triton.next_power_of_2(values.shape[-1]))
        ctx.save_for_backward(rows, addresses, weights)
        ctx.memory_shape = values.shape
        return output

    @staticmethod
    def backward(ctx, incoming):
        rows, addresses, weights = ctx.saved_tensors
        gm = torch.zeros(ctx.memory_shape, device=rows.device, dtype=torch.float32)
        gw = torch.empty_like(weights)
        _read_backward[(addresses.numel(),)](rows, addresses, weights, incoming.contiguous(), gm, gw,
                                            weights.shape[-1], rows.shape[-1], triton.next_power_of_2(rows.shape[-1]),
                                            enable_fp_fusion=False)
        return gm, None, gw


def read(values, requests, indices, weights):
    weights = weights.float().contiguous()
    validate_access(values, requests, indices, weights)
    addresses = absolute_addresses(values, requests, indices)
    if not torch.is_grad_enabled() or not (values.requires_grad or weights.requires_grad):
        return snapshot_read(values, addresses, weights)
    return _Read.apply(values, addresses, weights)


@triton.jit
def _propose_forward(memory, requests, indices, weights, targets, beta, log_decay, mass,
                     addresses, deltas, saved_rows, H: tl.constexpr, S: tl.constexpr,
                     K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    req = tl.load(requests + row // H)
    k, d = tl.arange(0, BK), tl.arange(0, BD)
    mask = (k[:, None] < K) & (d[None, :] < D)
    idx = tl.load(indices + row * K + k, k < K, other=0)
    addr = (req * H + row % H) * S + idx
    x = tl.load(memory + addr[:, None] * D + d[None, :], mask, other=0.0)
    w = tl.load(weights + row * K + k, k < K, other=0.0)
    a = libdevice.exp(tl.load(log_decay + row))
    b = tl.load(beta + row)
    m = tl.load(mass + row // H)
    t = tl.load(targets + row * D + d, d < D, other=0.0)
    decayed = a * x
    retrieved = tl.sum(decayed * w[:, None], 0)
    error = b * (t - retrieved)
    delta = (decayed - x + w[:, None] * error[None, :]) * m
    offset = (row * K + k[:, None]) * D + d[None, :]
    tl.store(saved_rows + offset, x, mask)
    tl.store(deltas + offset, delta, mask)
    tl.store(addresses + row * K + k, addr, k < K)


@triton.jit
def _propose_backward(rows, addresses, weights, targets, beta, log_decay, mass, incoming,
                      gm, gw, gt, gb, gg, gmass, H: tl.constexpr, K: tl.constexpr,
                      D: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    k, d = tl.arange(0, BK), tl.arange(0, BD)
    mask = (k[:, None] < K) & (d[None, :] < D)
    offset = (row * K + k[:, None]) * D + d[None, :]
    x = tl.load(rows + offset, mask, other=0.0)
    g = tl.load(incoming + offset, mask, other=0.0)
    w = tl.load(weights + row * K + k, k < K, other=0.0)
    addr = tl.load(addresses + row * K + k, k < K, other=0)
    a, b = libdevice.exp(tl.load(log_decay + row)), tl.load(beta + row)
    m = tl.load(mass + row // H)
    t = tl.load(targets + row * D + d, d < D, other=0.0)
    retrieved = tl.sum(x * w[:, None], 0)
    error = t - a * retrieved
    weighted_grad = tl.sum(g * w[:, None], 0)
    dx = m * ((a - 1.0) * g - a * b * w[:, None] * weighted_grad[None, :])
    dw = m * b * tl.sum(g * error[None, :] - a * x * weighted_grad[None, :], 1)
    dt = m * b * weighted_grad
    db = m * tl.sum(weighted_grad * error, 0)
    dg = m * a * tl.sum(tl.sum(g * x, 0) - b * weighted_grad * retrieved, 0)
    unweighted = a * x - x + w[:, None] * (b * error)[None, :]
    dm = tl.sum(tl.sum(g * unweighted, 1), 0)
    tl.store(gm + addr[:, None] * D + d[None, :], dx, mask)
    tl.store(gw + row * K + k, dw, k < K)
    tl.store(gt + row * D + d, dt, d < D)
    tl.store(gb + row, db)
    tl.store(gg + row, dg)
    tl.store(gmass + row, dm)


class _Propose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, requests, indices, weights, targets, beta, log_decay, mass):
        validate_access(memory, requests, indices, weights)
        A, H, K = indices.shape
        D = memory.shape[-1]
        if targets.shape != (A, H, D) or beta.shape != (A, H, 1) or log_decay.shape != beta.shape or mass.shape != (A,):
            raise ValueError("native proposal target, gate, decay or mass shape mismatch")
        if any(t.device != memory.device for t in (targets, beta, log_decay, mass)):
            raise ValueError("native proposal operands must share the memory device")
        rows = torch.empty((A, H, K, D), dtype=torch.float32, device=memory.device)
        deltas = torch.empty_like(rows)
        addresses = torch.empty_like(indices)
        _propose_forward[(A * H,)](memory, requests, indices, weights, targets, beta, log_decay, mass,
                                  addresses, deltas, rows, H, memory.shape[2], K, D,
                                  triton.next_power_of_2(K), triton.next_power_of_2(D), enable_fp_fusion=False)
        ctx.save_for_backward(rows, addresses, weights, targets, beta, log_decay, mass)
        ctx.memory_shape = memory.shape
        ctx.mark_non_differentiable(addresses)
        return addresses, deltas

    @staticmethod
    def backward(ctx, _, incoming):
        rows, addresses, weights, targets, beta, log_decay, mass = ctx.saved_tensors
        A, H, K, D = rows.shape
        gm = torch.zeros(ctx.memory_shape, device=rows.device, dtype=torch.float32)
        gw, gt, gb, gg = [torch.empty_like(t) for t in (weights, targets, beta, log_decay)]
        gmass = torch.empty((A, H), device=rows.device, dtype=torch.float32)
        _propose_backward[(A * H,)](rows, addresses, weights, targets, beta, log_decay, mass,
                                   incoming.contiguous(), gm, gw, gt, gb, gg, gmass, H, K, D,
                                   triton.next_power_of_2(K), triton.next_power_of_2(D), enable_fp_fusion=False)
        return gm, None, None, gw, gt, gb, gg, gmass.sum(-1)


def propose(memory, requests, indices, weights, targets, beta, log_decay, mass):
    floats = [t.float().contiguous() for t in (weights, targets, beta, log_decay, mass)]
    addresses, deltas = _Propose.apply(memory, requests, indices, *floats)
    return addresses.flatten(), deltas.flatten(0, 2)


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
