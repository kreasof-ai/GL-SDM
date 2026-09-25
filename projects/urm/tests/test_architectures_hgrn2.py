"""Parity gates for arch-024 HGRN2 (U2.A channel-decay, value-first layout).

Verified against the pinned fla source (fla/layers/hgrn2.py @ 864a87f6, the
sweep's verification origin): the typed U2.A channel-gate mixer matches the
pinned fused_recurrent_gla (gk=g, state_v_first=True, GPU) on the pinned
operands (q = swish, k = 1 − exp(g), v = i_proj).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.hgrn2 import HGRN2Layer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_gla_hgrn2_form_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.nn.functional.silu(torch.randn(B, T, H, DK)).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, DK)).cuda()
    k = (1 - g.exp())

    layer = HGRN2Layer(HIDDEN, H, DK, DV).cuda()
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device="cuda"),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.gla.fused_recurrent_gla")
    expected, _ = fused(q=q, k=k, v=v, gk=g, scale=None, output_final_state=True,
                        state_v_first=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"hgrn2 mixer parity: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = HGRN2Layer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.f_proj.weight.grad is not None
    assert layer.i_proj.weight.grad is not None
