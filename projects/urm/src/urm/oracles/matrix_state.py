"""Float64 canonical K2 matrix-state delta recurrence for one partition.

This is the dense-address specialization of the K3 sparse-slot oracle
(:mod:`urm.oracles.sparse_slot`): the sparse write/read slot vectors become dense
key/query vectors over the key dimension, and the per-slot decay becomes a
per-head scalar or per-key-channel diagonal. For one independent partition
(state ``M`` of shape ``[K, V]``), with scalar write strength ``beta_t`` and a
nonpositive log-decay schedule, the after-update recurrence is::

    Z_t   = G_t M_(t-1)                    # G_t diagonal decay
    h_t   = k_t^T Z_t                      # retrieval under the decayed state
    delta = beta_t (v_t - h_t)             # delta-rule correction
    M_t   = Z_t + k_t delta^T              # rank-one write
    y_t   = scale * q_t^T M_t              # read (after the update)

``G_t`` is ``exp(g_t)``: a scalar for head-granularity decay or a length-``K``
vector for key-channel decay. Read-before-update is supported by reading the
pre-update state. No intermediate state casts are modeled; this module does not
certify a BF16 lowering. It is an oracle for representation and composition
checks, never a performance backend.
"""

from __future__ import annotations

import numpy as np


def _inputs(memory, keys, queries, values, beta, log_decay):
    m, k, q, v, b, g = (
        np.asarray(x, dtype=np.float64)
        for x in (memory, keys, queries, values, beta, log_decay)
    )
    if m.ndim != 2:
        raise ValueError("memory must be a [K, V] matrix")
    key_dim, value_dim = m.shape
    if k.ndim != 2 or q.ndim != 2 or v.ndim != 2:
        raise ValueError("keys, queries and values must be [T, ...] matrices")
    t = k.shape[0]
    if k.shape[1] != key_dim or q.shape != (t, key_dim) or v.shape != (t, value_dim):
        raise ValueError("incompatible state, key, query or value shapes")
    if b.shape != (t,):
        raise ValueError("beta must have shape [T]")
    # log_decay is [T] (head scalar) or [T, K] (key-channel).
    if g.shape not in ((t,), (t, key_dim)):
        raise ValueError("log_decay must have shape [T] or [T, K]")
    if any(not np.isfinite(x).all() for x in (m, k, q, v, b, g)):
        raise ValueError("inputs must be finite")
    if (g > 0).any():
        raise ValueError("this contract requires nonpositive log decay")
    return m, k, q, v, b, g


def _decay_factor(g_t):
    """Return the diagonal decay for one token as a broadcastable factor.

    A scalar schedule (head decay) broadcasts over the whole state; a length-``K``
    schedule (key-channel decay) broadcasts over the value dimension.
    """
    return np.exp(g_t) if g_t.ndim == 0 else np.exp(g_t)[:, None]


def recurrent(memory, keys, queries, values, beta, log_decay, *, scale=1.0,
              read_before_update=False):
    """Independent token recurrence returning readings and final memory."""
    m, k, q, v, b, g = _inputs(memory, keys, queries, values, beta, log_decay)
    m = m.copy()
    out = np.empty_like(v)
    for t in range(len(v)):
        z = _decay_factor(g[t]) * m
        if read_before_update:
            out[t] = scale * (q[t] @ z)
        h = k[t] @ z
        delta = b[t] * (v[t] - h)
        m = z + k[t][:, None] * delta[None, :]
        if not read_before_update:
            out[t] = scale * (q[t] @ m)
    return out, m


def chunked(memory, keys, queries, values, beta, log_decay, *, chunk_size,
            scale=1.0):
    """Chunk-local triangular solve with exact boundary propagation in reals.

    This is the shared K2/K3 chunked formulation specialized to dense keys with
    head-granularity decay (a scalar per token). Pairwise decay is
    exp(log-prefix differences), never a product of clipped inverse exponentials.
    Dense arrays and np.linalg.solve are oracle machinery, not a proposed GPU
    schedule. Partial final chunks are supported. Read timing is after-update
    (the mandatory matrix case).
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    m, k, q, v, b, g = _inputs(memory, keys, queries, values, beta, log_decay)
    if g.ndim != 1:
        raise ValueError("the chunked oracle currently specializes head decay")
    m = m.copy()
    out = np.empty_like(v)
    for start in range(0, len(v), chunk_size):
        stop = min(start + chunk_size, len(v))
        kc, qc, vc, bc = k[start:stop], q[start:stop], v[start:stop], b[start:stop]
        c = stop - start
        # prefix[t] = sum of log decays up to and including token t (within chunk).
        prefix = np.cumsum(g[start:stop])
        # Decay from the chunk boundary to each token, applied to the boundary state.
        initial = np.exp(prefix)[:, None, None] * m[None, :, :]
        # v0[t] = k_t^T (decayed boundary state); y0[t] = scale * q_t^T (same).
        v0 = np.einsum("tk,tkv->tv", kc, initial)
        y0 = np.einsum("tk,tkv->tv", qc, initial)
        # transport[t, j] = decay from token j's write to token t.
        a = np.zeros((c, c))
        omega = np.zeros((c, c))
        for ti in range(c):
            for j in range(ti + 1):
                transport = np.exp(prefix[ti] - prefix[j])
                omega[ti, j] = transport * (qc[ti] @ kc[j])
                if j < ti:
                    a[ti, j] = transport * (kc[ti] @ kc[j])
        # Unit lower-triangular solve for the write corrections.
        delta = np.linalg.solve(np.eye(c) + bc[:, None] * a, bc[:, None] * (vc - v0))
        out[start:stop] = scale * (y0 + omega @ delta)
        # Propagate the chunk-boundary state: decay the old boundary to the chunk
        # end and fold in each write's contribution transported to the chunk end.
        fold = kc * np.exp(prefix[-1] - prefix)[:, None]
        m = np.exp(prefix[-1]) * m + fold.T @ delta
    return out, m


def recurrent_vjp(memory, keys, queries, values, beta, log_decay,
                  output_cotangent, final_cotangent, *, scale=1.0):
    """Analytical reverse recurrence, including decay and final-state gradients.

    After-update reads. Gradients are returned for memory, keys, queries, values,
    beta, and log_decay. Head-granularity decay (scalar per token). The adjoint
    mirrors :func:`urm.oracles.sparse_slot.recurrent_vjp` with dense keys.
    """
    m, k, q, v, b, g = _inputs(memory, keys, queries, values, beta, log_decay)
    if g.ndim != 1:
        raise ValueError("the vjp oracle currently specializes head decay")
    dy = np.asarray(output_cotangent, dtype=np.float64)
    carry = np.asarray(final_cotangent, dtype=np.float64).copy()
    if dy.shape != v.shape or carry.shape != m.shape:
        raise ValueError("cotangent shapes must match outputs")
    tape = []
    for t in range(len(v)):
        decay = np.exp(g[t])
        z = decay * m
        residual = v[t] - k[t] @ z
        delta = b[t] * residual
        m = z + k[t][:, None] * delta[None, :]
        tape.append((decay, z, residual, delta, m))
    dk, dq, dv, db, dg = (np.zeros_like(x) for x in (k, q, v, b, g))
    for t in reversed(range(len(v))):
        decay, z, residual, delta, state = tape[t]
        # Read: y_t = scale * q_t^T M_t.
        dq[t] = scale * (state @ dy[t])
        total = carry + scale * q[t][:, None] * dy[t][None, :]
        # Write: M_t = Z_t + k_t delta^T.
        ddelta = k[t] @ total
        dv[t] = b[t] * ddelta
        db[t] = ddelta @ residual
        dk[t] = total @ delta - b[t] * (z @ ddelta)
        dz = total - b[t] * k[t][:, None] * ddelta[None, :]
        dg[t] = np.sum(dz * z)
        carry = decay * dz
    return {
        "memory": carry,
        "keys": dk,
        "queries": dq,
        "values": dv,
        "beta": db,
        "log_decay": dg,
    }
