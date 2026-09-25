"""Parity gates for arch-068 TDA (A13 THRESHOLD_RELU_POWER reducer + Diff combinator).

Verified against the pinned tda source (triton_threshold_attention.py, the
sweep's verification origin): out = (ReLU(Q@K^T − τ))^p @ V with position-
dependent threshold τ_i = β·sqrt(2·log(i+1)/d), causal mask zeroing j>i before
the threshold, NO normalization denominator. The pinned is a Triton kernel with
no torch reference, so the oracle is the equation transcription; the
differential variant reuses the admitted Diff combinator (two K1 threshold
calls + typed Merge).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.tda import TDALayer

H, D, T = 2, 8, 6
BETA, POWER = 1.0, 2.0


def _operands(seed: int):
    torch.manual_seed(seed)
    return (torch.randn(1, T, H, D), torch.randn(1, T, H, D), torch.randn(1, T, H, D))


def _manual_threshold(q, k, v, beta, power):
    s = torch.einsum("bqhd,bkhd->bhqk", q * D ** -0.5, k)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    s = s.masked_fill(~mask, 0.0)
    i = torch.arange(1, T + 1).float()
    tau = beta * torch.sqrt(2.0 * torch.log(i) / D)
    relu = torch.clamp(s - tau.view(1, 1, -1, 1), min=0.0)
    return torch.einsum("bhqk,bkhd->bqhd", relu.pow(power), v)


def test_tda_single_path_matches_equation():
    q, k, v = _operands(seed=5)
    layer = TDALayer(H, D, beta=BETA, relu_power=POWER, differential=False)
    with torch.no_grad():
        actual = layer(q, k, v)
    expected = _manual_threshold(q, k, v, BETA, POWER)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"tda single-path parity: max abs err {err}"


def test_tda_threshold_zeroes_below_threshold():
    """Scores below τ_i contribute exactly zero (no normalization denominator).

    The first query (i=1) has τ = β·sqrt(2·log(1)/d) = 0, so only rows i≥2 are
    thresholded to zero under a huge β.
    """
    q, k, v = _operands(seed=7)
    big = TDALayer(H, D, beta=1e6, relu_power=POWER, differential=False)
    with torch.no_grad():
        out = big(q, k, v)
    # Rows i≥2 (index ≥1) are fully below threshold → zero; row 0 has τ=0.
    assert out[:, 1:].abs().max().item() == 0.0
    # Row 0 reduces to ReLU(score)^p @ v (τ=0).
    s0 = torch.einsum("bqhd,bkhd->bhqk", q[:, :1] * D ** -0.5, k[:, :1])
    expected0 = torch.einsum("bhqk,bkhd->bqhd", torch.clamp(s0, min=0.0).pow(POWER), v[:, :1])
    assert (out[:, :1] - expected0).abs().max().item() < 1e-5


def test_tda_differential_uses_diff_combinator():
    """out = out1 − clamp(λ,0,1)·out2 over two threshold-ReLU paths."""
    q, k, v = _operands(seed=11)
    torch.manual_seed(12)
    q2, k2 = torch.randn(1, T, H, D), torch.randn(1, T, H, D)
    layer = TDALayer(H, D, beta=BETA, relu_power=POWER, differential=True)
    lam = 0.5
    with torch.no_grad():
        actual = layer(q, k, v, query2=q2, key2=k2, lam=lam)
    o1 = _manual_threshold(q, k, v, BETA, POWER)
    o2 = _manual_threshold(q2, k2, v, BETA, POWER)
    expected = o1 - lam * o2
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"tda differential parity: max abs err {err}"


def test_tda_gradients_flow():
    q, k, v = _operands(seed=13)
    for t in (q, k, v):
        t.requires_grad_(True)
    TDALayer(H, D, beta=BETA, relu_power=POWER, differential=False)(q, k, v).square().sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and t.grad.abs().sum().item() > 0, f"{name} has no gradient"
