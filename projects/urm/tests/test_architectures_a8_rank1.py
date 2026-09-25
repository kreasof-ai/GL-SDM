"""Parity gates for arch-031 IPLR and arch-037 Comba (A8 generalized rank-1 sub-family).

Verified against the pinned fla sources (@ 864a87f6, the sweep's verification
origin): IPLR (identity-plus-rank-1, no decay) and Comba (dual-key delta with an
independent predict key) — both on the admitted generalized rank-1 LinearDeltaSpec.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.comba import CombaLayer
from architectures.iplr import IPLRLayer

H, K, V, T = 2, 4, 4, 5


# --- arch-031 IPLR ---

def test_iplr_matches_pinned_naive():
    torch.manual_seed(5)
    q = torch.randn(2, H, T, K) * 0.5
    k = torch.randn(2, H, T, K) * 0.5
    v = torch.randn(2, H, T, V)
    alpha = torch.randn(2, H, T, K) * 0.3
    beta = torch.randn(2, H, T, K) * 0.3
    layer = IPLRLayer(H, K, V)
    with torch.no_grad():
        actual = layer(q, k, v, alpha, beta)
    from benchmarks.comparators.fla_k2 import fla_op
    iplr = fla_op("fla.ops.generalized_delta_rule.iplr.naive.iplr_recurrence")
    expected, _ = iplr(q, k, v, alpha, beta)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"iplr parity: max abs err {err}"


def test_iplr_rank1_transition_active():
    torch.manual_seed(7)
    q = torch.randn(1, H, T, K); k = torch.randn(1, H, T, K); v = torch.randn(1, H, T, V)
    alpha = torch.randn(1, H, T, K); beta = torch.randn(1, H, T, K)
    layer = IPLRLayer(H, K, V)
    with torch.no_grad():
        out1 = layer(q, k, v, alpha, beta)
        out2 = layer(q, k, v, torch.zeros_like(alpha), beta)
    assert (out1 - out2).abs().max().item() > 1e-3


# --- arch-037 Comba ---

def test_comba_matches_pinned_naive():
    torch.manual_seed(9)
    q = torch.randn(2, H, T, K)
    k = torch.randn(2, H, T, K)
    v = torch.randn(2, H, T, V)
    p = torch.randn(2, H, T, K)               # independent predict key
    beta = torch.rand(2, H, T)                # per-head scalar
    g = torch.nn.functional.logsigmoid(torch.randn(2, H, T))  # per-head log decay
    layer = CombaLayer(H, K, V)
    with torch.no_grad():
        actual = layer(q, k, v, p, beta, g)
    from benchmarks.comparators.fla_k2 import fla_op
    comba = fla_op("fla.ops.comba.naive.naive_recurrent_comba")
    # comba naive takes [B,T,H,*]
    expected, _ = comba(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        p.transpose(1, 2), beta.transpose(1, 2), g.transpose(1, 2), scale=None)
    err = (actual.transpose(1, 2) - expected).abs().max().item()
    assert err < 2e-3, f"comba parity: max abs err {err}"


def test_comba_predict_key_is_independent():
    """The predict key p (retrieval) and write key k (commit) are distinct operands."""
    torch.manual_seed(11)
    q = torch.randn(1, H, T, K); k = torch.randn(1, H, T, K); v = torch.randn(1, H, T, V)
    p = torch.randn(1, H, T, K); beta = torch.rand(1, H, T)
    g = torch.nn.functional.logsigmoid(torch.randn(1, H, T))
    layer = CombaLayer(H, K, V)
    with torch.no_grad():
        out1 = layer(q, k, v, p, beta, g)
        out2 = layer(q, k, v, k, beta, g)  # p := k collapses toward the canonical delta
    assert (out1 - out2).abs().max().item() > 1e-3


def test_comba_gradients_flow():
    torch.manual_seed(13)
    q = torch.randn(1, H, T, K); k = torch.randn(1, H, T, K); v = torch.randn(1, H, T, V)
    p = torch.randn(1, H, T, K, requires_grad=True)
    beta = torch.rand(1, H, T); g = torch.nn.functional.logsigmoid(torch.randn(1, H, T))
    CombaLayer(H, K, V)(q, k, v, p, beta, g).square().sum().backward()
    assert p.grad is not None and p.grad.abs().sum().item() > 0
