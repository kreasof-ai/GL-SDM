"""Parity gates for arch-008 FoX / Forgetting Attention (A13 score/reducer).

Verified against the pinned fla source (fla/ops/forgetting_attn/naive.py @
864a87f6, the sweep's verification origin): the typed K1 call with the external
cumulative forget-gate score bias (gc_i − gc_j) matches the pinned naive
reference (fp32).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.forgetting_attention import ForgettingAttentionLayer

H, D, T = 4, 8, 6


def _operands(seed: int):
    torch.manual_seed(seed)
    q = torch.randn(2, T, H, D)
    k = torch.randn(2, T, H, D)
    v = torch.randn(2, T, H, D)
    g = torch.nn.functional.logsigmoid(torch.randn(2, T, H))  # per-head log forget gate
    return q, k, v, g


def test_fox_matches_pinned_naive():
    q, k, v, g = _operands(seed=5)
    layer = ForgettingAttentionLayer(H, D)
    with torch.no_grad():
        actual = layer(q, k, v, g)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.forgetting_attn.naive.naive_forgetting_attn")
    expected = naive(q, k, v, g, scale=None)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"fox parity: max abs err {err}"


def test_cumulative_gate_bias_is_applied():
    """The score bias is gc_i − gc_j: a zero gate reduces to plain causal softmax."""
    q, k, v, _ = _operands(seed=9)
    layer = ForgettingAttentionLayer(H, D)
    with torch.no_grad():
        zero_g = torch.zeros(2, T, H)
        actual = layer(q, k, v, zero_g)
        # manual plain causal softmax
        s = torch.einsum("bqhd,bkhd->bhqk", q * D ** -0.5, k)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf"))
        expected = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"zero-gate reduces to causal softmax: max abs err {err}"


def test_gradients_flow_to_gate():
    q, k, v, g = _operands(seed=13)
    g.requires_grad_(True)
    ForgettingAttentionLayer(H, D)(q, k, v, g).square().sum().backward()
    assert g.grad is not None and g.grad.abs().sum().item() > 0
