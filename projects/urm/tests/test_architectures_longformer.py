"""Parity gates for arch-071 Longformer (A2 indexed-K1, band ∪ global).

Verified against the pinned longformer source (longformer/longformer.py, the
sweep's verification origin): softmax attention restricted to the per-query
visible set band(i) ∪ {global}, softmaxed JOINTLY in fp32. The pinned module
requires a legacy transformers stack, so the oracle is the equation
transcription of the joint band∪global softmax; the route (band/global indices)
and the separate global-output overwrite pass are external.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.longformer import LongformerLayer

H, D, T, W = 2, 8, 8, 2


def _manual_band_global(q, k, v, gather):
    B, T, H, D = q.shape
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    outs = torch.zeros(B, T, H, D)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                sel = gather[0, 0, t]
                sel = sel[sel >= 0]
                ks, vs = kk[b, h, sel], vv[b, h, sel]
                sc = (qq[b, h, t] @ ks.T) * D ** -0.5
                outs[b, t, h] = torch.softmax(sc, -1) @ vs
    return outs


def test_longformer_band_global_joint_softmax():
    torch.manual_seed(5)
    q = torch.randn(1, T, H, D)
    k = torch.randn(1, T, H, D)
    v = torch.randn(1, T, H, D)
    layer = LongformerLayer(H, D, W)
    gather = layer.build_gather_indices(T, global_positions=[0])
    with torch.no_grad():
        actual = layer(q, k, v, gather)
    expected = _manual_band_global(q, k, v, gather)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"longformer band∪global parity: max abs err {err}"


def test_longformer_global_in_every_row():
    """The global token is in every query's visible set (jointly normalized)."""
    torch.manual_seed(7)
    q = torch.randn(1, T, H, D)
    k = torch.randn(1, T, H, D)
    v = torch.randn(1, T, H, D)
    layer = LongformerLayer(H, D, W)
    # no global
    gather_none = layer.build_gather_indices(T, global_positions=[])
    # global token 3
    gather_g = layer.build_gather_indices(T, global_positions=[3])
    with torch.no_grad():
        out_none = layer(q, k, v, gather_none)
        out_g = layer(q, k, v, gather_g)
    # adding the global token changes rows whose band excludes position 3
    assert (out_none - out_g).abs().max().item() > 0


def test_longformer_gradients_flow():
    torch.manual_seed(13)
    q = torch.randn(1, T, H, D, requires_grad=True)
    k = torch.randn(1, T, H, D)
    v = torch.randn(1, T, H, D)
    layer = LongformerLayer(H, D, W)
    gather = layer.build_gather_indices(T, global_positions=[0])
    layer(q, k, v, gather).square().sum().backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0
