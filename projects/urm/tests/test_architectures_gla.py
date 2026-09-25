"""Parity gates for arch-019 GLA (U2.A channel-diagonal gate).

Verified against the pinned fla source (fla/ops/gla + fla/layers/gla.py @
864a87f6, the sweep's verification origin): the typed U2.A channel-gate mixer
matches the pinned fused_recurrent_gla on identical operands (GPU), and the
full layer composes projections → mixer → output gate → o_proj.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.gla import GLALayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_gla_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    k = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    gk = torch.nn.functional.logsigmoid(torch.randn(B, T, H, DK)).cuda() / 16

    layer = GLALayer(HIDDEN, H, DK, DV)
    out = layer._run_mixer({
        "query": q.transpose(1, 2),
        "key": k.transpose(1, 2),
        "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device="cuda"),
        "log_decay": gk.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.gla.fused_recurrent_gla")
    # Pinned fused op takes [B, T, H, *].
    expected, _state = fused(
        q=q, k=k, v=v, gk=gk, scale=None, output_final_state=True
    )
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"mixer parity vs pinned fused op: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = GLALayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.gk_proj[0].weight.grad is not None
    assert layer.q_proj.weight.grad is not None
