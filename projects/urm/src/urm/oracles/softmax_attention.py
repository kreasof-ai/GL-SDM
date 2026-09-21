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
    # Broadcast each kv head over its query-head group without materializing K/V.
    k_b = np.repeat(k, group, axis=0) if group != 1 else k
    v_b = np.repeat(v, group, axis=0) if group != 1 else v

    scores = np.einsum("htk,hsk->hts", q, k_b) * scale
    if score_bias is not None:
        scores = scores + np.asarray(score_bias, dtype=np.float64)
    if causal:
        # Query position i (0-based) aligns to key position i + (Tk - Tq).
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

    # Softmax with a fully-masked-row guard: empty rows yield zero output.
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

    # dP = dy V^T ; softmax adjoint: dS = P * (dP - sum(dP * P)).
    dprobs = np.einsum("htv,hsv->hts", dy, v_b)
    delta = np.sum(dprobs * probs, axis=-1, keepdims=True)
    dscores = probs * (dprobs - delta) * scale
    dq = np.einsum("hts,hsk->htk", dscores, k_b)
    dk_b = np.einsum("hts,htk->hsk", dscores, q)
    dv_b = np.einsum("hts,htv->hsv", probs, dy)
    # Fold the broadcast group cotangents back onto the shared kv heads.
    if group != 1:
        dk = dk_b.reshape(kv_heads, group, k.shape[1], key_dim).sum(axis=1)
        dv = dv_b.reshape(kv_heads, group, v.shape[1], v.shape[2]).sum(axis=1)
    else:
        dk, dv = dk_b, dv_b
    return {"query": dq, "key": dk, "value": dv}
