"""Parity gates for arch-042 RWKV-7 (A8 DPLR low-rank transition).

Verified against the pinned fla source (fla/ops/rwkv7/fused_recurrent.py @
864a87f6, the sweep's verification origin), which maps to
fused_recurrent_dplr_delta_rule with gk = w: S_t = Diag(exp w)·S_{t-1} +
(a_tᵀS_{t-1})⊗b_t + v_t⊗k_t, read after update. The typed K2 low-rank call
matches the pinned DPLR naive recurrence on identical operands (fp32).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.rwkv7 import RWKV7Layer

H, K, V, T = 2, 4, 4, 5


def _operands(seed: int):
    torch.manual_seed(seed)
    r = torch.randn(2, T, H, K)
    w = -torch.rand(2, T, H, K) * 0.3   # log decay
    k = torch.randn(2, T, H, K)
    v = torch.randn(2, T, H, V)
    a = torch.randn(2, T, H, K)         # rank-1 read factor
    b = torch.randn(2, T, H, K)         # rank-1 write factor
    return r, w, k, v, a, b


def test_rwkv7_matches_pinned_dplr_naive():
    r, w, k, v, a, b = _operands(seed=5)
    layer = RWKV7Layer(H, K, V)
    with torch.no_grad():
        actual = layer(r, w, k, v, a, b)
    from benchmarks.comparators.fla_k2 import fla_op
    dplr = fla_op("fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence")
    # dplr takes [b,h,l,d] (head-first)
    expected, _ = dplr(r.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                       a.transpose(1, 2), b.transpose(1, 2), w.transpose(1, 2))
    err = (actual - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"rwkv7 (dplr) parity: max abs err {err}"


def test_rwkv7_low_rank_term_is_active():
    """The rank-1 transition (a⊗b) must affect the output (not just decay + write)."""
    r, w, k, v, a, b = _operands(seed=9)
    layer = RWKV7Layer(H, K, V)
    with torch.no_grad():
        out1 = layer(r, w, k, v, a, b)
        out2 = layer(r, w, k, v, torch.zeros_like(a), b)  # alpha=0 kills the low-rank term
    assert (out1 - out2).abs().max().item() > 1e-3, "low-rank transition has no effect"


def test_rwkv7_gradients_flow_to_rank1_factors():
    r, w, k, v, a, b = _operands(seed=13)
    layer = RWKV7Layer(H, K, V)
    a.requires_grad_(True)
    b.requires_grad_(True)
    layer(r, w, k, v, a, b).square().sum().backward()
    assert a.grad is not None and a.grad.abs().sum().item() > 0
    assert b.grad is not None and b.grad.abs().sum().item() > 0
