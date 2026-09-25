"""Parity gates for arch-018 Simple GLA and arch-016 Lightning Attention.

Both verified against the pinned fla simple_gla op (fla/ops/simple_gla +
layers @ 864a87f6, the sweep's verification origin): 018 with a data-dependent
head-scalar gate ``g``, 016 with a static per-head schedule ``g_gamma``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.simple_gla import LightningAttentionLayer, SimpleGLALayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def _mixer_out(layer, q, k, v, g):
    B, T = q.shape[0], q.shape[1]
    return layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device=q.device),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device=q.device),
    })["output"]


def test_simple_gla_mixer_matches_pinned_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_simple_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    k = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 8  # [B,T,H]
    layer = SimpleGLALayer(HIDDEN, H, DK, DV).cuda()
    out = _mixer_out(layer, q, k, v, g)
    fused = fla_op("fla.ops.simple_gla.fused_recurrent_simple_gla")
    expected, _ = fused(q=q, k=k, v=v, g=g, scale=None, output_final_state=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"simple_gla parity: max abs err {err}"


def test_lightning_mixer_matches_pinned_static_g_gamma_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_simple_gla requires CUDA")
    torch.manual_seed(9)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    k = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    layer = LightningAttentionLayer(HIDDEN, H, DK, DV, layer_idx=1, num_layers=4).cuda()
    g_gamma = layer.g_gamma.cuda()  # [H]
    g = g_gamma.view(1, 1, H).expand(B, T, H)
    out = _mixer_out(layer, q, k, v, g)
    fused = fla_op("fla.ops.simple_gla.fused_recurrent_simple_gla")
    expected, _ = fused(q=q, k=k, v=v, g_gamma=g_gamma, scale=None, output_final_state=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"lightning static g_gamma parity: max abs err {err}"


def test_simple_gla_full_layer_composition_and_gradients():
    torch.manual_seed(13)
    layer = SimpleGLALayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    out.square().sum().backward()
    assert layer.gk_proj.weight.grad is not None
    assert layer.q_proj.weight.grad is not None
