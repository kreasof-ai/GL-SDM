"""Float64 selected-slot delta references for one partition, after-update reads.

Routes are dense vectors with zeros outside the selected slots. The decay mask
is explicit: a selected slot with zero write weight still decays. No intermediate
state casts are modeled. This module does not certify a BF16 lowering.
"""

from __future__ import annotations

import numpy as np


def _inputs(memory, writes, reads, values, beta, log_decay, selected):
    m, w, q, v, b, g = (
        np.asarray(x, dtype=np.float64)
        for x in (memory, writes, reads, values, beta, log_decay)
    )
    selected = np.asarray(selected, dtype=bool)
    if m.ndim != 2 or w.ndim != 2 or v.ndim != 2:
        raise ValueError("memory, writes and values must be matrices")
    t, s = w.shape
    if (
        q.shape != (t, s)
        or selected.shape != (t, s)
        or v.shape != (t, m.shape[1])
        or m.shape[0] != s
        or b.shape != (t,)
        or g.shape != (t,)
    ):
        raise ValueError("incompatible state, route, value or schedule shapes")
    if any(not np.isfinite(x).all() for x in (m, w, q, v, b, g)):
        raise ValueError("inputs must be finite")
    if (g > 0).any():
        raise ValueError("this contract requires nonpositive log decay")
    if (w[~selected] != 0).any():
        raise ValueError("write support must be a subset of selected slots")
    return m, w, q, v, b, g, selected


def recurrent(memory, writes, reads, values, beta, log_decay, selected):
    """Independent token recurrence returning readings and final memory."""
    m, w, q, v, b, g, selected = _inputs(
        memory, writes, reads, values, beta, log_decay, selected
    )
    m = m.copy()
    out = np.empty_like(v)
    for t in range(len(v)):
        z = np.exp(g[t] * selected[t])[:, None] * m
        delta = b[t] * (v[t] - w[t] @ z)
        m = z + w[t, :, None] * delta
        out[t] = q[t] @ m
    return out, m


def chunked(memory, writes, reads, values, beta, log_decay, selected, *, chunk_size):
    """Chunk-local triangular solve with exact boundary propagation in reals.

    Pairwise decay is exp(log-prefix differences), never a product of clipped
    inverse exponentials. Dense arrays and np.linalg.solve are oracle machinery,
    not a proposed GPU schedule. Partial final chunks are supported.
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    m, w, q, v, b, g, selected = _inputs(
        memory, writes, reads, values, beta, log_decay, selected
    )
    m = m.copy()
    out = np.empty_like(v)
    for start in range(0, len(v), chunk_size):
        stop = min(start + chunk_size, len(v))
        wc, qc, vc, bc = w[start:stop], q[start:stop], v[start:stop], b[start:stop]
        c = stop - start
        prefix = np.cumsum(g[start:stop, None] * selected[start:stop], axis=0)
        initial = np.exp(prefix)[:, :, None] * m
        v0 = np.einsum("ts,tsd->td", wc, initial)
        y0 = np.einsum("ts,tsd->td", qc, initial)
        a = np.zeros((c, c))
        omega = np.zeros((c, c))
        for t in range(c):
            for j in range(t + 1):
                transport = np.exp(prefix[t] - prefix[j])
                omega[t, j] = (qc[t] * transport) @ wc[j]
                if j < t:
                    a[t, j] = (wc[t] * transport) @ wc[j]
        delta = np.linalg.solve(np.eye(c) + bc[:, None] * a, bc[:, None] * (vc - v0))
        out[start:stop] = y0 + omega @ delta
        fold = wc * np.exp(prefix[-1] - prefix)
        m = np.exp(prefix[-1])[:, None] * m + fold.T @ delta
    return out, m


def recurrent_vjp(
    memory,
    writes,
    reads,
    values,
    beta,
    log_decay,
    selected,
    output_cotangent,
    final_cotangent,
):
    """Analytical reverse recurrence, including decay and final-state gradients.

    Route support is fixed; gradients are not defined for discrete selection.
    The write gradient is masked to that support.
    """
    m, w, q, v, b, g, selected = _inputs(
        memory, writes, reads, values, beta, log_decay, selected
    )
    dy = np.asarray(output_cotangent, dtype=np.float64)
    carry = np.asarray(final_cotangent, dtype=np.float64).copy()
    if dy.shape != v.shape or carry.shape != m.shape:
        raise ValueError("cotangent shapes must match outputs")
    tape = []
    for t in range(len(v)):
        decay = np.exp(g[t] * selected[t])
        z = decay[:, None] * m
        residual = v[t] - w[t] @ z
        delta = b[t] * residual
        m = z + w[t, :, None] * delta
        tape.append((decay, z, residual, delta, m))
    dw, dq, dv, db, dg = (np.zeros_like(x) for x in (w, q, v, b, g))
    for t in reversed(range(len(v))):
        decay, z, residual, delta, state = tape[t]
        dq[t] = state @ dy[t]
        total = carry + q[t, :, None] * dy[t]
        ddelta = w[t] @ total
        dv[t] = b[t] * ddelta
        db[t] = ddelta @ residual
        dw[t] = (total @ delta - z @ dv[t]) * selected[t]
        dz = total - w[t, :, None] * dv[t]
        dg[t] = np.sum(dz * z * selected[t, :, None])
        carry = decay[:, None] * dz
    return {
        "memory": carry,
        "writes": dw,
        "reads": dq,
        "values": dv,
        "beta": db,
        "log_decay": dg,
    }


def sparse_delta_state(
    memory,
    read_addresses,
    read_weights,
    *,
    write_addresses=None,
    write_weights=None,
    values=None,
    beta=None,
    log_decay=None,
    spec,
):
    """Canonical K3 sparse-delta state — the address-index signature every tier shares.

    Same operand form (address indices, not slot vectors) as the Torch reference
    and native launcher. Runs in float64. Returns ``(readings, updated_memory)``.
    """
    from urm.ir.program import SparseReadTiming as _RT

    mem = np.asarray(memory, dtype=np.float64)
    r_idx = np.asarray(read_addresses, dtype=np.int64)
    r_w = np.asarray(read_weights, dtype=np.float64)
    w_idx = np.asarray(write_addresses, dtype=np.int64)
    w_w = np.asarray(write_weights, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    g = np.asarray(log_decay, dtype=np.float64)
    if spec.read_timing is not _RT.AFTER_UPDATE:
        raise ValueError("the K3 NumPy oracle covers the after-update read")
    P, T, reads = r_idx.shape
    slots = mem.shape[1]
    outs = np.empty((P, T, mem.shape[2]), dtype=np.float64)
    finals = np.empty_like(mem)
    for p in range(P):
        writes = np.zeros((T, slots))
        reads_v = np.zeros((T, slots))
        sel = np.zeros((T, slots), dtype=bool)
        for t in range(T):
            writes[t, w_idx[p, t]] = w_w[p, t]
            reads_v[t, r_idx[p, t]] = r_w[p, t]
            sel[t, w_idx[p, t]] = True
        out_p, m = recurrent(mem[p], writes, reads_v, vals[p], b[p, :, 0], g[p, :, 0], sel)
        outs[p] = out_p
        finals[p] = m
    return outs, finals
