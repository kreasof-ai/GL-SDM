"""Parity gates for arch-011 Wall Attention (A13 CHANNEL_DECAY score law).

Verified against the pinned fla source (fla/ops/wall_attn/naive.py @ 864a87f6,
the sweep's verification origin): the typed K1 channel-decay call matches the
pinned eager reference on identical operands. The pinned's base-2 form
(``exp2(P·RCP_LN2)``) equals the natural-exp form here; the gate is passed in
natural log.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from architectures.wall_attention import WallAttentionLayer

H, D, T = 4, 8, 6
RCP_LN2 = 1.0 / math.log(2.0)


def _operands(seed: int):
    torch.manual_seed(seed)
    q = torch.randn(2, T, H, D)
    k = torch.randn(2, T, H, D)
    v = torch.randn(2, T, H, D)
    # per-channel log2 gate (the pinned works in log2); small to keep exp2 stable
    g_log2 = -torch.rand(2, T, H, D) * 0.5
    return q, k, v, g_log2


def test_wall_matches_pinned_naive():
    if not torch.cuda.is_available():
        pytest.skip("pinned wall cumsum (chunk_global_cumsum) requires CUDA")
    q, k, v, g_log2 = [t.cuda() for t in _operands(seed=5)]
    layer = WallAttentionLayer(H, D)
    with torch.no_grad():
        # exp2(cumsum(g_log2)·RCP_LN2 diff) == exp(cumsum(g_log2) diff): pass g_log2
        # directly (the channel-decay law folds the base-2 RCP_LN2 into the score).
        actual = layer(q, k, v, g_log2)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.wall_attn.naive.naive_wall_attn")
    expected = naive(q, k, v, g_log2, scale=D ** -0.5)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"wall parity: max abs err {err}"


def test_zero_gate_reduces_to_causal_softmax():
    """A zero per-channel gate makes exp(P_in − P_jn) = 1: plain causal softmax.

    Uses match_pinned_base2=False so the scale is the plain key-dim rule (no
    RCP_LN2 fold), making the reduction to plain causal softmax exact.
    """
    q, k, v, _ = _operands(seed=9)
    layer = WallAttentionLayer(H, D)
    with torch.no_grad():
        actual = layer(q, k, v, torch.zeros(2, T, H, D))
        s = torch.einsum("bqhd,bkhd->bhqk", q * D ** -0.5, k)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf"))
        expected = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v)
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"zero-gate reduces to causal softmax: max abs err {err}"


def test_gradients_flow_to_channel_gate():
    q, k, v, g_log2 = _operands(seed=13)
    g = g_log2.requires_grad_(True)
    WallAttentionLayer(H, D)(q, k, v, g).square().sum().backward()
    assert g.grad is not None and g.grad.abs().sum().item() > 0
