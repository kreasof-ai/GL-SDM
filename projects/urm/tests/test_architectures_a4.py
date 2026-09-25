"""Parity gates for A4 hierarchical chunks: arch-009 Log-linear + arch-046 LogLinearMamba2.

Verified against the pinned fla source (fla/ops/log_linear_attn/naive.py @
864a87f6, the sweep's verification origin): the hierarchical dyadic level-mask
law o = (H ∘ qkᵀ)·v with H = Σ_level exp(segsum(g)) ∘ mask_level matches the
pinned naive_log_linear_attn on identical operands. LogLinearMamba2 shares the
same hierarchical mixer, driven by the Mamba-2 frontend (external).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from architectures.log_linear_attention import (
    LogLinearAttentionLayer,
    log_linear_disjoint_reference,
)
from architectures.log_linear_mamba2 import LogLinearMamba2Layer

H, D, T = 2, 8, 8


def _operands(seed: int):
    torch.manual_seed(seed)
    L = int(math.ceil(math.log2(T))) + 1
    q = torch.randn(2, T, H, D)
    k = torch.randn(2, T, H, D)
    v = torch.randn(2, T, H, D)
    g = torch.nn.functional.logsigmoid(torch.randn(2, T, H))
    ls = torch.randn(2, T, H, L)
    return q, k, v, g, ls


def test_log_linear_matches_pinned_naive():
    q, k, v, g, ls = _operands(seed=5)
    layer = LogLinearAttentionLayer(H, D)
    with torch.no_grad():
        actual = layer(q, k, v, g, ls)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.log_linear_attn.naive.naive_log_linear_attn")
    expected = naive(q, k, v, g, ls)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"log_linear parity: max abs err {err}"


def test_log_linear_hierarchy_is_active():
    """The level scales must affect the output (the hierarchy is not a plain decay)."""
    q, k, v, g, ls = _operands(seed=7)
    layer = LogLinearAttentionLayer(H, D)
    with torch.no_grad():
        out1 = layer(q, k, v, g, ls)
        out2 = layer(q, k, v, g, torch.zeros_like(ls))
    assert (out1 - out2).abs().max().item() > 1e-3, "level scales have no effect"


def test_log_linear_mamba2_frontend_composition():
    """LogLinearMamba2 drives the shared hierarchical law via the Mamba-2 frontend."""
    torch.manual_seed(9)
    hidden = 32
    layer = LogLinearMamba2Layer(hidden, H, D, num_levels=int(math.ceil(math.log2(T))) + 1)
    x = torch.randn(2, T, hidden)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (2, T, H, D)
    assert torch.isfinite(out).all()


def test_log_linear_gradients_flow():
    q, k, v, g, ls = _operands(seed=13)
    g.requires_grad_(True)
    LogLinearAttentionLayer(H, D)(q, k, v, g, ls).square().sum().backward()
    assert g.grad is not None and g.grad.abs().sum().item() > 0


def test_disjoint_block_form_matches_pinned_naive():
    """The disjoint dyadic-block decomposition is the exact A4 reference oracle.

    Each causal pair (t, j≤t) belongs to exactly one dyadic level; the disjoint form
    sums the per-level contributions with decay exp(gcum[t]−gcum[j]). It reproduces
    the pinned naive (and the full-matrix layer) at 0.0 — this is the honest A4
    reference. The banked recurrent form (the pinned chunked kernel's carry-cascade
    state with within-chunk decay) is the residual schedule, recorded not claimed.
    """
    q, k, v, g, ls = _operands(seed=17)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.log_linear_attn.naive.naive_log_linear_attn")
    with torch.no_grad():
        expected = naive(q, k, v, g, ls)
        disjoint = log_linear_disjoint_reference(q, k, v, g, ls)
    err = (disjoint - expected).abs().max().item()
    assert err < 1e-4, f"disjoint-block A4 oracle vs pinned naive: max abs err {err}"
