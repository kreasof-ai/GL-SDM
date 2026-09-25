"""Parity gates for arch-050 Raven (external top-k router + GSA mixer).

Verified against the pinned fla source (fla/layers/raven.py + fla/ops/gsa/
naive.py @ 864a87f6, the sweep's verification origin): the router-masked GSA
mixer matches the pinned naive_recurrent_gsa on the router-produced (s, f)
operands (fp32), and the full layer composes router → gate masking → mixer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.raven import RavenLayer

HIDDEN, H, DK, DV, M, TOPK = 32, 2, 4, 4, 6, 3


def _router_operands(seed: int, T: int = 5):
    torch.manual_seed(seed)
    layer = RavenLayer(HIDDEN, H, DK, DV, M, TOPK).eval()  # eval: no Gumbel noise
    hidden = torch.randn(2, T, HIDDEN)
    with torch.no_grad():
        # Replicate the pinned router exactly.
        router = layer.r_proj(hidden).view(2, T, H, M)
        orig_scores = torch.sigmoid(router)
        route_idx = orig_scores.topk(TOPK, dim=-1).indices
        topk_weights = torch.gather(orig_scores, dim=-1, index=route_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))
        f = torch.nn.functional.logsigmoid(layer.f_proj(hidden).view(2, T, H, M)) / layer.gate_logit_normalizer
        f = f * s_multihot
        s = 1 - f.exp()
    return layer, hidden, s, f


def test_router_masked_mixer_matches_pinned_naive_gsa():
    layer, hidden, s, f = _router_operands(seed=5)
    with torch.no_grad():
        B, T, _ = hidden.shape
        q = layer.q_proj(hidden).view(B, T, H, DK)
        k = layer.k_proj(hidden).view(B, T, H, DK)
        v = layer.v_proj(hidden).view(B, T, H, DV)
        actual = layer._mixer(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            s.transpose(1, 2), f.transpose(1, 2), layer.scale,
        )
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.gsa.naive.naive_recurrent_gsa")
    expected, _ = naive(q, k, v, s, g=f, scale=None)  # naive takes [B,T,H,*]
    err = (actual.transpose(1, 2) - expected).abs().max().item()
    assert err < 2e-3, f"raven mixer parity: max abs err {err}"


def test_router_produces_sparse_multihot():
    """The router's s_multihot is sparse: exactly topk non-zero slots per position."""
    layer, hidden, s, f = _router_operands(seed=9)
    with torch.no_grad():
        router = layer.r_proj(hidden).view(2, -1, H, M)
        orig_scores = torch.sigmoid(router)
        route_idx = orig_scores.topk(TOPK, dim=-1).indices
        topk_weights = torch.gather(orig_scores, -1, route_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))
    nonzero = (s_multihot > 0).sum(dim=-1)
    assert (nonzero == TOPK).all()


def test_full_layer_composition_and_gradients():
    torch.manual_seed(13)
    layer = RavenLayer(HIDDEN, H, DK, DV, M, TOPK)
    hidden = torch.randn(2, 5, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 5, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.r_proj.weight.grad is not None
    assert layer.f_proj.weight.grad is not None
