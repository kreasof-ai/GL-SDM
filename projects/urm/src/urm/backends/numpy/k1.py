"""Float64 canonical K1 normalized routed reduction (softmax attention).

The K1 core computes, for each query row over its visible keys::

    scores = scale * Q K^T + bias
    P      = softmax(scores)        # over visible keys, after masking
    Y      = P V

with explicit routing (an additive/boolean attention mask and causal masking),
normalization (the softmax denominator), and no state. This is the canonical,
backend-independent execution path every K1 architecture lowers into; it is an
oracle for representation and composition checks, never a performance backend.
Grouped-query head sharing is expressed by broadcasting each key/value head over
its query-head group, not by materializing expanded K/V.
"""

from __future__ import annotations

import numpy as np


def _inputs(query, key, value):
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (query, key, value))
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("query/key/value must be [H, T, d] matrices")
    q_heads, q_len, key_dim = q.shape
    kv_heads, k_len, key_dim_k = key.shape
    if k.shape[:1] != v.shape[:1] or k.shape[1] != v.shape[1]:
        raise ValueError("key and value must share head and source dimensions")
    if key_dim != key_dim_k:
        raise ValueError("query and key must share the key dimension")
    if q_heads % kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    if min(q_heads, q_len, key_dim, k_len, v.shape[-1]) <= 0:
        raise ValueError("K1 dimensions must be positive")
    if any(not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("inputs must be finite")
    return q, k, v


def attention(query, key, value, *, scale=None, causal=True, score_bias=None,
              attention_mask=None):
    """Canonical K1 normalized routed reduction for one (batched) head stack.

    ``query`` is ``[Hq, Tq, K]``, ``key``/``value`` are ``[Hkv, Tk, K]``/``[Hkv,
    Tk, V]``. ``attention_mask`` is broadcastable to ``[Hq, Tq, Tk]``: boolean
    (True = visible) or additive float. ``score_bias`` is an additive float
    broadcastable to ``[Hq, Tq, Tk]``. Fully masked rows produce zero output.
    Returns the ``[Hq, Tq, V]`` output.
    """
    q, k, v = _inputs(query, key, value)
    q_heads, q_len, key_dim = q.shape
    kv_heads, k_len, _ = k.shape
    if scale is None:
        scale = key_dim ** -0.5
    group = q_heads // kv_heads
    k_b = np.repeat(k, group, axis=0) if group != 1 else k
    v_b = np.repeat(v, group, axis=0) if group != 1 else v

    scores = np.einsum("htk,hsk->hts", q, k_b) * scale
    if score_bias is not None:
        scores = scores + np.asarray(score_bias, dtype=np.float64)
    if causal:
        q_pos = np.arange(q_len)[:, None] + (k_len - q_len)
        k_pos = np.arange(k_len)[None, :]
        causal_mask = k_pos <= q_pos
        scores = np.where(causal_mask[None], scores, -np.inf)
    if attention_mask is not None:
        mask = np.asarray(attention_mask)
        if mask.dtype == bool:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask.astype(np.float64)

    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp = np.where(np.isfinite(scores), np.exp(scores - row_max), 0.0)
    denom = exp.sum(axis=-1, keepdims=True)
    probs = np.divide(exp, denom, out=np.zeros_like(exp), where=denom != 0)
    return np.einsum("hts,hsv->htv", probs, v_b)


def attention_probs(query, key, value, *, scale=None, causal=True,
                    score_bias=None, attention_mask=None):
    """The canonical K1 normalized weight matrix ``P`` (before the V reduce).

    Exposed so composed K1 operations (differential, parallax, ...) can build on
    the single canonical reduction without re-deriving its masking/softmax.
    Returns the ``[Hq, Tq, Tk]`` probabilities.
    """
    q, k, v = _inputs(query, key, value)
    q_heads, q_len, key_dim = q.shape
    kv_heads, k_len, _ = k.shape
    if scale is None:
        scale = key_dim ** -0.5
    group = q_heads // kv_heads
    k_b = np.repeat(k, group, axis=0) if group != 1 else k
    scores = np.einsum("htk,hsk->hts", q, k_b) * scale
    if score_bias is not None:
        scores = scores + np.asarray(score_bias, dtype=np.float64)
    if causal:
        q_pos = np.arange(q_len)[:, None] + (k_len - q_len)
        k_pos = np.arange(k_len)[None, :]
        scores = np.where((k_pos <= q_pos)[None], scores, -np.inf)
    if attention_mask is not None:
        mask = np.asarray(attention_mask)
        if mask.dtype == bool:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask.astype(np.float64)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp = np.where(np.isfinite(scores), np.exp(scores - row_max), 0.0)
    denom = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, denom, out=np.zeros_like(exp), where=denom != 0)


def attention_vjp(query, key, value, output_cotangent, *, scale=None, causal=True,
                  score_bias=None, attention_mask=None):
    """Analytical adjoint returning query/key/value cotangents (dense K/V).

    Key/value cotangents are accumulated onto the shared (unexpanded) kv heads,
    matching the grouped-query ownership of the forward.
    """
    q, k, v = _inputs(query, key, value)
    dy = np.asarray(output_cotangent, dtype=np.float64)
    q_heads, q_len, key_dim = q.shape
    kv_heads = k.shape[0]
    group = q_heads // kv_heads
    if scale is None:
        scale = key_dim ** -0.5
    k_b = np.repeat(k, group, axis=0) if group != 1 else k
    v_b = np.repeat(v, group, axis=0) if group != 1 else v

    scores = np.einsum("htk,hsk->hts", q, k_b) * scale
    if score_bias is not None:
        scores = scores + np.asarray(score_bias, dtype=np.float64)
    if causal:
        q_pos = np.arange(q_len)[:, None] + (k.shape[1] - q_len)
        k_pos = np.arange(k.shape[1])[None, :]
        scores = np.where((k_pos <= q_pos)[None], scores, -np.inf)
    if attention_mask is not None:
        mask = np.asarray(attention_mask)
        if mask.dtype == bool:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask.astype(np.float64)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp = np.where(np.isfinite(scores), np.exp(scores - row_max), 0.0)
    denom = exp.sum(axis=-1, keepdims=True)
    probs = np.divide(exp, denom, out=np.zeros_like(exp), where=denom != 0)

    dprobs = np.einsum("htv,hsv->hts", dy, v_b)
    delta = np.sum(dprobs * probs, axis=-1, keepdims=True)
    dscores = probs * (dprobs - delta) * scale
    dq = np.einsum("hts,hsk->htk", dscores, k_b)
    dk_b = np.einsum("hts,htk->hsk", dscores, q)
    dv_b = np.einsum("hts,htv->hsv", probs, dy)
    if group != 1:
        dk = dk_b.reshape(kv_heads, group, k.shape[1], key_dim).sum(axis=1)
        dv = dv_b.reshape(kv_heads, group, v.shape[1], v.shape[2]).sum(axis=1)
    else:
        dk, dv = dk_b, dv_b
    return {"query": dq, "key": dk, "value": dv}


def k1_softmax_attention(
    query,
    key,
    value,
    *,
    descriptor,
    score_bias=None,
    attention_mask=None,
    scale=None,
):
    """Canonical K1 softmax attention — the batched signature every tier shares.

    Same role order, batched shapes (``[B, T, H, D]``), closed descriptor and
    return as the Torch reference and native Triton schedule. Runs in float64.
    Returns output ``[B, T, H, Dv]``.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    q = np.transpose(q, (0, 2, 1, 3))  # [B,T,H,D] -> [B,H,T,D] for the oracle
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))
    out = np.stack(
        [
            attention(
                q[b], k[b], v[b],
                scale=scale,
                causal=descriptor.causal,
                score_bias=score_bias,
                attention_mask=attention_mask,
            )
            for b in range(q.shape[0])
        ]
    )
    return np.transpose(out, (0, 2, 1, 3))


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class K1NumpyProvider:
    name = "urm.reference.numpy.k1.softmax_attention.v1"
    family = "k1"
    tier = "reference"

    def decline(self, request) -> str | None:
        from ...ir.program import K1Descriptor

        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 NumPy provider requires a closed K1Descriptor"
        return None

    def execute(self, request, operands):
        out = k1_softmax_attention(
            operands["query"], operands["key"], operands["value"],
            descriptor=request.descriptor,
            score_bias=operands.get("score_bias"),
            attention_mask=operands.get("attention_mask"),
            scale=None if operands.get("scale") is None else float(operands["scale"]),
        )
        return {"output": out}


PROVIDERS = (K1NumpyProvider(),)


def attention_online(query, key, value, *, scale=None, causal=True, score_bias=None,
                     attention_mask=None, block_size=64):
    """K1 online (tiled) softmax: the performance-axis form of the K1 equation.

    This is the running ``(m, l, a)`` online-softmax reduction — the same
    equation as :func:`attention`, evaluated by streaming key blocks with a
    running row maximum and exponent sum, never materializing the score matrix.
    It is the same-equation oracle for the native Triton tiled schedule: a fast
    K1 kernel is verified against this form, proving the online reassociation
    is exact in reals (and bounding its float envelope against the fp64
    materialized form). Not a performance backend — an equation-preserving
    reference for the tiled lowering.
    """
    q, k, v = _inputs(query, key, value)
    q_heads, q_len, key_dim = q.shape
    kv_heads, k_len, _ = k.shape
    if scale is None:
        scale = key_dim ** -0.5
    group = q_heads // kv_heads
    k_b = np.repeat(k, group, axis=0) if group != 1 else k
    v_b = np.repeat(v, group, axis=0) if group != 1 else v

    # Precompute additive bias and visibility per (query, source) once.
    bias = np.zeros((q_heads, q_len, k_len), dtype=np.float64)
    if score_bias is not None:
        bias = bias + np.asarray(score_bias, dtype=np.float64)
    visible = np.ones((q_heads, q_len, k_len), dtype=bool)
    if causal:
        q_pos = np.arange(q_len)[:, None] + (k_len - q_len)
        k_pos = np.arange(k_len)[None, :]
        visible &= (k_pos <= q_pos)[None]
    if attention_mask is not None:
        mask = np.asarray(attention_mask)
        visible &= mask if mask.dtype == bool else np.isfinite(mask)

    out = np.zeros((q_heads, q_len, v_b.shape[-1]), dtype=np.float64)
    for h in range(q_heads):
        for t in range(q_len):
            m = -np.inf  # running row maximum
            l = 0.0      # running exponent sum
            a = np.zeros(v_b.shape[-1], dtype=np.float64)  # running weighted value
            for start in range(0, k_len, block_size):
                stop = min(start + block_size, k_len)
                s = scale * (k_b[h, start:stop] @ q[h, t]) + bias[h, t, start:stop]
                vis = visible[h, t, start:stop]
                s = np.where(vis, s, -np.inf)
                m_b = np.max(s) if np.any(vis) else -np.inf
                m_new = max(m, m_b)
                # Rescale the running accumulator to the new maximum.
                if np.isfinite(m_new):
                    l = l * np.exp(m - m_new) if np.isfinite(m) else l
                    a = a * np.exp(m - m_new) if np.isfinite(m) else a
                    exp_b = np.where(vis, np.exp(s - m_new), 0.0)
                    l = l + exp_b.sum()
                    a = a + exp_b @ v_b[h, start:stop]
                m = m_new
            out[h, t] = a / l if l != 0 else 0.0  # all-masked row -> zero
    return out


__all__.append("attention_online") if "__all__" in dir() else None
