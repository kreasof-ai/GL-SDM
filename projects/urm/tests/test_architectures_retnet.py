"""Parity gates for arch-017 RetNet / multiscale retention (U2.A static decay).

Verified against the pinned fla source (fla/ops/retention + layers @ 864a87f6,
the sweep's verification origin): the typed U2.A static-head-decay mixer
matches the pinned fused_recurrent_retention (GPU), with the static
γ_h = 1 − 2^(−5−h) schedule supplied as an external log-decay operand.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.retnet import RetNetLayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_retention_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_retention requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    k = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    layer = RetNetLayer(HIDDEN, H, DK, DV).cuda()
    log_gamma = layer.log_gamma.cuda()
    g = log_gamma.view(1, 1, H).expand(B, T, H)
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device="cuda"),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]
    fused = fla_op("fla.ops.retention.fused_recurrent_retention")
    expected, _ = fused(q=q, k=k, v=v, scale=None, output_final_state=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"retention parity: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = RetNetLayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.q_proj.weight.grad is not None
    assert layer.g_proj.weight.grad is not None
