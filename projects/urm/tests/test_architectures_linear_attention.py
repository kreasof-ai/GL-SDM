"""Parity gates for arch-015 Linear Attention (U2.A, no decay).

Verified against the pinned fla source (fla/ops/linear_attn + layers @
864a87f6, the sweep's verification origin): the typed U2.A mixer matches the
pinned fused_recurrent_linear_attn (GPU) in both the plain and normalized
variants (the K2 descriptor's normalized=True denominator-state path).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.linear_attention import LinearAttentionLayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def _mixer_out(layer, q, k, v):
    B, T = q.shape[0], q.shape[1]
    return layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device=q.device),
        "log_decay": torch.zeros(B, H, T, device=q.device),
        "initial_state": torch.zeros(B, H, DK, DV, device=q.device),
    })["output"]


@pytest.mark.parametrize("normalize", (False, True))
def test_mixer_matches_pinned_fused_recurrent_linear_attn_cuda(normalize):
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_linear_attn requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    fm = LinearAttentionLayer._feature_map
    q = fm(torch.randn(B, T, H, DK)).cuda()
    k = fm(torch.randn(B, T, H, DK)).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    layer = LinearAttentionLayer(HIDDEN, H, DK, DV, normalize=normalize).cuda()
    out = _mixer_out(layer, q, k, v)
    fused = fla_op("fla.ops.linear_attn.fused_recurrent_linear_attn")
    expected, _ = fused(q=q, k=k, v=v, scale=None, output_final_state=True, normalize=normalize)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"linear_attn normalize={normalize} parity: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = LinearAttentionLayer(HIDDEN, H, DK, DV, normalize=True)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.q_proj.weight.grad is not None
