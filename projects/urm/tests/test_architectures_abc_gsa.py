"""Parity gates for arch-048 ABC and arch-049 GSA (two-stage slot-summary combinator).

Verified against the pinned fla source (fla/ops/abc/naive.py + gsa/naive.py @
864a87f6, the sweep's verification origin): the two typed K2 additive
channel-gate calls + external interstage slot softmax match the pinned naive
recurrence (fp32, CPU) on identical operands.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.abc_gsa import ABCLayer, GSALayer

H, DK, DV, M = 2, 4, 4, 3


def _operands(seed: int, T: int = 5):
    torch.manual_seed(seed)
    q = torch.randn(1, H, T, DK)
    k = torch.randn(1, H, T, DK)
    v = torch.randn(1, H, T, DV)
    s = torch.randn(1, H, T, M)
    return q, k, v, s


def test_abc_matches_pinned_naive_recurrent():
    q, k, v, s = _operands(seed=5)
    layer = ABCLayer(H, DK, DV, M)
    with torch.no_grad():
        actual = layer(q, k, v, s)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.abc.naive.naive_recurrent_abc")
    # naive_recurrent_abc takes [B, H, T, *] and derives g from s when g=None.
    expected, _ = naive(q, k, v, s, g=None, scale=None)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"abc parity: max abs err {err}"


def test_gsa_matches_pinned_naive_recurrent():
    q, k, v, s = _operands(seed=9)
    # GSA: g explicitly supplied per slot (negative log-decay).
    g = -torch.rand(1, H, q.shape[2], M)
    layer = GSALayer(H, DK, DV, M)
    with torch.no_grad():
        actual = layer(q, k, v, s, g)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.abc.naive.naive_recurrent_abc")
    expected, _ = naive(q, k, v, s, g=g, scale=None)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"gsa parity: max abs err {err}"


def test_gradients_flow_through_both_stages():
    q, k, v, s = _operands(seed=13)
    for t in (q, k, v, s):
        t.requires_grad_(True)
    ABCLayer(H, DK, DV, M)(q, k, v, s).square().sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v), ("s", s)):
        assert t.grad is not None and t.grad.abs().sum().item() > 0, f"{name} has no gradient"
