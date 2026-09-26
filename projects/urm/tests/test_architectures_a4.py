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
    BankedLogLinearMixer,
    construct_H_matrix,
    dyadic_level_of,
)
from architectures.log_linear_mamba2 import LogLinearMamba2Layer

H, D, T = 2, 8, 8


def log_linear_disjoint_reference(q, k, v, g, level_scales):
    """The disjoint dyadic-block reference form of the hierarchical law (exact oracle).

    Equivalent to ``construct_H_matrix`` + the contraction, but computed as a sum over
    the disjoint dyadic blocks — each key j contributes to query t at exactly one level
    with decay ``exp(gcum[t]−gcum[j])`` (gcum the per-head prefix cumsum of g) and the
    per-level scale ``level_scales[t, level]``. Verified against the pinned
    ``naive_log_linear_attn`` and the full-matrix form at 0.0. This is the honest
    reference oracle; the banked recurrent form (the pinned chunked kernel's
    ``LogLinearAttentionState`` with the carry-cascade promote/reset and within-chunk
    decay) is the residual schedule, not yet derived to parity.
    """
    B, T, H, D = q.shape
    num_levels = level_scales.shape[-1]
    gcum = g.permute(0, 2, 1).cumsum(-1)                      # [B,H,T]
    out = torch.zeros(B, T, H, D, dtype=torch.float32)
    for t in range(T):
        for j in range(t + 1):
            l = dyadic_level_of(t, j, num_levels)
            aij = torch.exp(gcum[..., t] - gcum[..., j])       # [B,H]
            score = (q[:, t].float() * k[:, j].float()).sum(-1)  # [B,H]
            out[:, t] += (level_scales[:, t, :, l].float() * aij * score).unsqueeze(-1) * v[:, j].float()
    return out.to(q.dtype)


class LogLinearAttentionLayer(torch.nn.Module):
    """Log-linear attention mixer (full-matrix reference form of the hierarchical law).

    Retained as the independent full-matrix comparator; the public banked path is
    :class:`BankedLogLinearMixer`.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, g, level_scales):
        """q/k/v [B,T,H,D]; g per-head log decay [B,T,H]; level_scales [B,T,H,L_levels]."""
        H = construct_H_matrix(g.permute(0, 2, 1), level_scales.permute(0, 2, 3, 1))  # [B,H,T,T]
        M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
        return torch.einsum("bhlc,bchp->blhp", M, v)


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


def test_banked_public_op_matches_pinned_naive():
    """The public dyadic_banked_state op reproduces the pinned naive (the banked
    carry-cascade lifecycle derived to parity)."""
    q, k, v, g, ls = _operands(seed=3)
    L = ls.shape[-1]
    mixer = BankedLogLinearMixer(H, D, L)
    with torch.no_grad():
        actual = mixer(q, k, v, g, ls)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.log_linear_attn.naive.naive_log_linear_attn")
    expected = naive(q, k, v, g, ls)
    err = (actual - expected).abs().max().item()
    assert err < 1e-3, f"banked public op vs pinned naive: max abs err {err}"


def test_banked_public_op_matches_disjoint_and_fullmatrix():
    """The banked op matches both independent comparators across shapes (incl. partial
    dyadic blocks at non-power-of-two T)."""
    for (Tl, seed) in ((8, 21), (7, 22), (13, 23), (24, 24)):
        torch.manual_seed(seed)
        L = int(math.ceil(math.log2(Tl))) + 1
        q = torch.randn(2, Tl, H, D); k = torch.randn(2, Tl, H, D); v = torch.randn(2, Tl, H, D)
        g = torch.nn.functional.logsigmoid(torch.randn(2, Tl, H)); ls = torch.randn(2, Tl, H, L)
        mixer = BankedLogLinearMixer(H, D, L)
        with torch.no_grad():
            banked = mixer(q, k, v, g, ls)
            disjoint = log_linear_disjoint_reference(q, k, v, g, ls)
            fullmatrix = LogLinearAttentionLayer(H, D)(q, k, v, g, ls)
        assert (banked - disjoint).abs().max().item() < 1e-3, f"T={Tl}: banked vs disjoint"
        assert (banked - fullmatrix).abs().max().item() < 1e-3, f"T={Tl}: banked vs full-matrix"


def test_banked_gradients_flow():
    q, k, v, g, ls = _operands(seed=25)
    L = ls.shape[-1]
    mixer = BankedLogLinearMixer(H, D, L)
    q.requires_grad_(True); ls.requires_grad_(True)
    mixer(q, k, v, g, ls).square().sum().backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0
    assert ls.grad is not None and ls.grad.abs().sum().item() > 0


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
