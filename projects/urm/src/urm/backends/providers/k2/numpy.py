"""Float64 canonical K2 matrix-state delta recurrence for one partition.

This is the dense-address specialization of the K3 sparse-slot oracle
(:mod:`urm.backends.providers.k3.numpy`): the sparse write/read slot vectors become dense
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

from ....ir.program import K2GateScope, K2ReadTiming, LinearDeltaSpec


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
              read_before_update=False, is_delta=True, normalizer=False,
              epsilon=1e-6, erase_gate=None, write_gate=None, retrieval_keys=None,
              left_transitions=None, update_keys=None, update_values=None,
              rank_beta=None):
    """Independent token recurrence returning readings and final memory.

    ``is_delta=True`` applies the delta-rule correction ``delta = beta (v - k^T
    Z)``; ``is_delta=False`` is the additive linear update ``M = Z + k v^T``
    (beta is ignored). ``normalizer=True`` tracks a denominator state ``z_t``
    (the same decay, accumulating the keys) and reads ``y = scale * (q^T M) /
    max(q^T z, epsilon)`` - the query/key-normalized (linear-attention) form.

    ``erase_gate``/``write_gate`` (each ``[T]``) generalize the delta rule to the
    dual-gate form ``update = write_gate * v - erase_gate * (k^T Z)``; the plain
    delta rule is the special case ``erase_gate = write_gate = beta``. Returns
    ``(out, m)`` or ``(out, (m, z))`` when ``normalizer`` is set.
    """
    m, k, q, v, b, g = _inputs(memory, keys, queries, values, beta, log_decay)
    retr = k if retrieval_keys is None else np.asarray(retrieval_keys, dtype=np.float64)
    if retr.shape != k.shape:
        raise ValueError("retrieval_keys must match keys shape [T, K]")
    dual_gate = erase_gate is not None or write_gate is not None
    if dual_gate:
        erase = np.asarray(erase_gate, dtype=np.float64)
        write = np.asarray(write_gate, dtype=np.float64)
        if erase.shape != k.shape or write.shape != v.shape:
            raise ValueError("erase_gate must be [T,K] and write_gate [T,V]")
    m = m.copy()
    norm = np.zeros(m.shape[0]) if normalizer else None
    out = np.empty_like(v)
    for t in range(len(v)):
        if left_transitions is not None:
            z = np.asarray(left_transitions[t], dtype=np.float64) @ m
        else:
            decay = _decay_factor(g[t])
            z = decay * m
        if normalizer:
            norm_decay = decay[:, 0] if decay.ndim == 2 else decay
            norm = norm_decay * norm
        if read_before_update:
            denom = (q[t] @ norm) if normalizer else None
            out[t] = scale * (q[t] @ z)
            if normalizer:
                out[t] = out[t] / max(denom, epsilon)
        if update_keys is not None:
            uk = np.asarray(update_keys[t], dtype=np.float64)
            uv = np.asarray(update_values[t], dtype=np.float64)
            rb = np.asarray(rank_beta[t], dtype=np.float64)
            for r in range(uk.shape[0]):
                delta_r = rb[r] * (uv[r] - uk[r] @ z)
                z = z + uk[r][:, None] * delta_r[None, :]
            m = z
        elif dual_gate:
            delta = write[t] * v[t] - (erase[t] * k[t]) @ z
            m = z + k[t][:, None] * delta[None, :]
        elif is_delta:
            delta = b[t] * (v[t] - retr[t] @ z)
            m = z + k[t][:, None] * delta[None, :]
        else:
            delta = v[t]
            m = z + k[t][:, None] * delta[None, :]
        if normalizer:
            norm = norm + k[t]
        if not read_before_update:
            out[t] = scale * (q[t] @ m)
            if normalizer:
                out[t] = out[t] / max(q[t] @ norm, epsilon)
    if normalizer:
        return out, (m, norm)
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
        prefix = np.cumsum(g[start:stop])
        initial = np.exp(prefix)[:, None, None] * m[None, :, :]
        v0 = np.einsum("tk,tkv->tv", kc, initial)
        y0 = np.einsum("tk,tkv->tv", qc, initial)
        a = np.zeros((c, c))
        omega = np.zeros((c, c))
        for ti in range(c):
            for j in range(ti + 1):
                transport = np.exp(prefix[ti] - prefix[j])
                omega[ti, j] = transport * (qc[ti] @ kc[j])
                if j < ti:
                    a[ti, j] = transport * (kc[ti] @ kc[j])
        delta = np.linalg.solve(np.eye(c) + bc[:, None] * a, bc[:, None] * (vc - v0))
        out[start:stop] = scale * (y0 + omega @ delta)
        fold = kc * np.exp(prefix[-1] - prefix)[:, None]
        m = np.exp(prefix[-1]) * m + fold.T @ delta
    return out, m


def recurrent_vjp(memory, keys, queries, values, beta, log_decay,
                  output_cotangent, final_cotangent, *, scale=1.0):
    """Analytical reverse recurrence, including decay and final-state gradients.

    After-update reads. Gradients are returned for memory, keys, queries, values,
    beta, and log_decay. Head-granularity decay (scalar per token). The adjoint
    mirrors :func:`urm.backends.providers.k3.numpy.recurrent_vjp` with dense keys.
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
        dq[t] = scale * (state @ dy[t])
        total = carry + scale * q[t][:, None] * dy[t][None, :]
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


def linear_delta_state(
    initial_state,
    keys,
    queries,
    values,
    beta,
    log_decay,
    *,
    spec: LinearDeltaSpec,
    scale: float | None = None,
):
    """Canonical K2 linear-delta state law — the batched signature every tier shares.

    This is the uniform interface: same role order, same batched shapes, same
    descriptor and same return as the Torch reference
    (:func:`urm.backends.providers.k2.torch.linear_delta_state`) and the
    native Triton schedule. Shapes: ``initial_state`` ``[B, H, K, V]``;
    ``keys``/``queries`` ``[B, H, T, K]``; ``values`` ``[B, H, T, V]``;
    ``beta``/``log_decay`` per gate scope. Runs in float64 (the oracle tier).
    Returns ``(out [B,H,T,V], final_state [B,H,K,V])``.
    """
    from ....ir.program import K2GateScope as _GS

    m0 = np.asarray(initial_state, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    g = np.asarray(log_decay, dtype=np.float64)
    if b.ndim == 4 and b.shape[-1] == 1:
        b = b[..., 0]
    if g.ndim == 4 and g.shape[-1] == 1:
        g = g[..., 0]
    if spec.scale_rule.value == "one":
        resolved = 1.0
    elif spec.scale_rule.value == "key_dim_rsqrt":
        resolved = float(k.shape[-1]) ** -0.5
    else:
        if scale is None:
            raise ValueError("scale_rule=explicit_operand requires a scale value")
        resolved = float(scale)
    B, H = k.shape[0], k.shape[1]
    out = np.empty((B, H, v.shape[2], v.shape[3]), dtype=np.float64)
    final = np.empty_like(m0)
    for bi in range(B):
        for hi in range(H):
            decay = np.zeros(k.shape[2]) if spec.gate_scope is _GS.NONE else g[bi, hi]
            o, m = recurrent(
                m0[bi, hi], k[bi, hi], q[bi, hi], v[bi, hi], b[bi, hi], decay,
                scale=resolved,
                is_delta=spec.delta,
                read_before_update=spec.read_timing.value == "before_update",
                normalizer=spec.normalized,
                epsilon=spec.epsilon,
            )
            out[bi, hi] = o[0] if spec.normalized else o
            final[bi, hi] = m[0] if spec.normalized else m
    return out, final
