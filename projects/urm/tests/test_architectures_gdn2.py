"""Parity gates for arch-027 GDN2 (A8 generalized rank-1 transition).

Verified against the pinned fla source (fla/ops/gdn2/naive.py @ 864a87f6, the
sweep's verification origin): the typed K2 generalized rank-1 call with the
erase_gate (b) and write_gate (w) roles matches the pinned naive recurrence on
identical operands (fp32).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.gdn2 import GDN2Layer

H, K, V, T = 2, 4, 4, 5


def _operands(seed: int):
    torch.manual_seed(seed)
    q = torch.randn(2, T, H, K)
    k = torch.randn(2, T, H, K)
    v = torch.randn(2, T, H, V)
    g = -torch.rand(2, T, H, K) * 0.3   # log decay
    b = torch.rand(2, T, H, K)          # erase gate
    w = torch.rand(2, T, H, V)          # write gate
    return q, k, v, g, b, w


def test_gdn2_matches_pinned_naive():
    q, k, v, g, b, w = _operands(seed=5)
    layer = GDN2Layer(H, K, V)
    with torch.no_grad():
        actual = layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                       g.transpose(1, 2), b.transpose(1, 2), w.transpose(1, 2))
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.gdn2.naive.naive_recurrent_gdn2")
    expected, _ = naive(q, k, v, g, b, w, scale=None)
    err = (actual.transpose(1, 2) - expected).abs().max().item()
    assert err < 2e-3, f"gdn2 parity: max abs err {err}"


def test_gdn2_erase_and_write_gates_are_distinct():
    """The erase gate and write gate act on different axes — not collapsible to one beta."""
    q, k, v, g, b, w = _operands(seed=9)
    layer = GDN2Layer(H, K, V)
    with torch.no_grad():
        # Changing only the write gate changes the output.
        out1 = layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                     g.transpose(1, 2), b.transpose(1, 2), w.transpose(1, 2))
        w2 = w * 2
        out2 = layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                     g.transpose(1, 2), b.transpose(1, 2), w2.transpose(1, 2))
    assert (out1 - out2).abs().max().item() > 1e-3, "write gate has no effect"


def test_gdn2_gradients_flow_to_gates():
    q, k, v, g, b, w = _operands(seed=13)
    layer = GDN2Layer(H, K, V)
    bt = b.transpose(1, 2).requires_grad_(True)
    wt = w.transpose(1, 2).requires_grad_(True)
    layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
          g.transpose(1, 2), bt, wt).square().sum().backward()
    assert bt.grad is not None and bt.grad.abs().sum().item() > 0
    assert wt.grad is not None and wt.grad.abs().sum().item() > 0
