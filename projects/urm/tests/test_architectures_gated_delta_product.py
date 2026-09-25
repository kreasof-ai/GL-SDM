"""Parity gates for arch-029 Gated DeltaProduct (A8 ordered multi-delta transition).

Verified against the pinned fla source (fla/ops/gated_delta_product/naive.py @
864a87f6, the sweep's verification origin): per token, head-scalar decay then R
ordered Householder delta factors (each retrieving from the state the previous
factor produced), read after update. The typed K2 num_deltas=R call matches the
pinned naive recurrence (fp32).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.gated_delta_product import GatedDeltaProductLayer

H, K, V, T, R = 2, 4, 4, 4, 3
SCALE = K ** -0.5


def _operands(seed: int):
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T * R, H, K)
    v = torch.randn(1, T * R, H, V)
    beta = torch.rand(1, T * R, H)
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, H))
    return q, k, v, beta, g


def test_delta_product_matches_pinned_naive():
    q, k, v, beta, g = _operands(seed=5)
    layer = GatedDeltaProductLayer(H, K, V, R)
    with torch.no_grad():
        actual = layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                       g.transpose(1, 2), beta.transpose(1, 2), SCALE)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product")
    expected, _ = naive(q, k, v, g, beta, SCALE, num_householder=R)
    err = (actual.transpose(1, 2) - expected).abs().max().item()
    assert err < 2e-3, f"delta_product parity: max abs err {err}"


def test_ordered_factors_matter():
    """The R factors are applied in order: permuting them changes the output."""
    q, k, v, beta, g = _operands(seed=7)
    layer = GatedDeltaProductLayer(H, K, V, R)
    with torch.no_grad():
        out1 = layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                     g.transpose(1, 2), beta.transpose(1, 2), SCALE)
        # Reverse the within-token factor order (R factors per token in the time axis).
        kR = k.view(1, T, R, H, K).flip(2).reshape(1, T * R, H, K)
        vR = v.view(1, T, R, H, V).flip(2).reshape(1, T * R, H, V)
        betaR = beta.view(1, T, R, H).flip(2).reshape(1, T * R, H)
        out2 = layer(q.transpose(1, 2), kR.transpose(1, 2), vR.transpose(1, 2),
                     g.transpose(1, 2), betaR.transpose(1, 2), SCALE)
    assert (out1 - out2).abs().max().item() > 1e-3, "factor order has no effect"


def test_delta_product_gradients_flow():
    q, k, v, beta, g = _operands(seed=13)
    layer = GatedDeltaProductLayer(H, K, V, R)
    k = k.requires_grad_(True)
    layer(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
          g.transpose(1, 2), beta.transpose(1, 2), SCALE).square().sum().backward()
    assert k.grad is not None and k.grad.abs().sum().item() > 0
