"""Parity gates for the UT (typed causal triangular transform) axis:
arch-013 DeltaFormer (strict-causal value correction) + arch-010 PaTH
(Householder operand correction).

Verified against the pinned fla sources (@ 864a87f6, the sweep's verification
origin): DeltaFormer's two-stage strict-past triangular value correction +
causal attention, and PaTH's single-chunk Householder UT operand correction +
cumulative-gate softmax — both matching the pinned naive references (fp32).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.deltaformer import DeltaFormerLayer
from architectures.path_attention import PaTHAttentionLayer

H, D, T = 2, 8, 8


# --- arch-013 DeltaFormer ---

def test_deltaformer_matches_pinned_naive():
    torch.manual_seed(5)
    q = torch.randn(2, H, T, D); k = torch.randn(2, H, T, D); v = torch.randn(2, H, T, D)
    beta = torch.rand(2, H, T)
    layer = DeltaFormerLayer(H, D)
    with torch.no_grad():
        actual = layer(q, k, v, beta)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.deltaformer.naive.naive_deltaformer_attn_head_first")
    expected = naive(q, k, v, beta)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"deltaformer parity: max abs err {err}"


def test_deltaformer_correction_is_strict_past():
    """The value correction is strict-past: u_t depends only on u_{<t}. A future
    value perturbation leaves earlier corrections unchanged."""
    torch.manual_seed(7)
    q = torch.randn(1, H, T, D); k = torch.randn(1, H, T, D); v = torch.randn(1, H, T, D)
    layer = DeltaFormerLayer(H, D)
    with torch.no_grad():
        base = layer(q, k, v)
        v2 = v.clone(); v2[:, :, -1] = torch.randn(1, H, D)
        perturbed = layer(q, k, v2)
    # outputs at earlier tokens unchanged (stage-2 is causal, stage-1 strict)
    assert (base[:, :, :-1] - perturbed[:, :, :-1]).abs().max().item() < 1e-5


def test_deltaformer_gradients_flow():
    torch.manual_seed(13)
    q = torch.randn(1, H, T, D, requires_grad=True); k = torch.randn(1, H, T, D)
    v = torch.randn(1, H, T, D)
    DeltaFormerLayer(H, D)(q, k, v).square().sum().backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0


# --- arch-010 PaTH ---

def test_path_single_chunk_matches_pinned():
    torch.manual_seed(5)
    HQ = 4
    scale = D ** -0.5
    q = torch.randn(1, T, HQ, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    w = torch.randn(1, T, H, D) * 0.3
    beta = torch.rand(1, T, H) * 0.5
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, HQ))
    layer = PaTHAttentionLayer(HQ, D)
    with torch.no_grad():
        actual = layer(q, k, v, w, beta, g, scale)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.path_attn.naive.naive_path_attn")
    expected = naive(q, k, v, w, beta, g, scale, chunk_size=T)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"path single-chunk parity: max abs err {err}"


def test_path_householder_correction_active():
    """The Householder w/beta correction must affect the output."""
    torch.manual_seed(9)
    HQ = 4
    scale = D ** -0.5
    q = torch.randn(1, T, HQ, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    w = torch.randn(1, T, H, D) * 0.3
    beta = torch.rand(1, T, H) * 0.5
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, HQ))
    layer = PaTHAttentionLayer(HQ, D)
    with torch.no_grad():
        out1 = layer(q, k, v, w, beta, g, scale)
        out2 = layer(q, k, v, torch.zeros_like(w), beta, g, scale)  # w=0 kills the correction
    assert (out1 - out2).abs().max().item() > 1e-3
