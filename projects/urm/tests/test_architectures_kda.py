"""Parity gates for arch-028 KDA (U2.D + key-channel decay, explicit 1/√K read-scale).

Verified against the pinned fla source (fla/ops/kda + layers @ 864a87f6, the
sweep's verification origin): the typed U2.D channel-gate mixer matches the
pinned fused_recurrent_kda (GPU) on identical operands.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.kda import KDALayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_kda_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_kda requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    g = -torch.rand(B, T, H, DK).cuda()  # per-channel log decay
    beta = torch.rand(B, T, H).cuda()

    layer = KDALayer(HIDDEN, H, DK, DV).cuda()
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": beta.transpose(1, 2),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.kda.fused_recurrent_kda")
    expected, _ = fused(q=q, k=k, v=v, g=g, beta=beta, scale=None, output_final_state=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"kda parity: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = KDALayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.f_proj.weight.grad is not None
    assert layer.A_log.grad is not None
