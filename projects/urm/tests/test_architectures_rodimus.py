"""Parity gates for arch-036 Rodimus (U2.A channel-decay + external input gate).

Verified against the pinned fla source (fla/layers/rodimus.py @ 864a87f6, the
sweep's verification origin): the typed U2.A channel-gate mixer matches the
pinned fused_recurrent_gla (gk=rt_gate_log, state_v_first=True, GPU) on the
pinned frontend operands (k = l2norm(k)·it_gate, rt_gate_log =
−softplus(g_gate)·sigmoid(τ_gate)).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.rodimus import RodimusLayer
from benchmarks.comparators.fla_k2 import fla_op

D_INNER, MEM = 32, 16


def test_mixer_matches_pinned_fused_recurrent_gla_rodimus_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    g_gate = torch.nn.functional.softplus(torch.randn(B, T, MEM)).cuda()
    tau_gate = torch.sigmoid(torch.randn(B, T, MEM)).cuda()
    it_gate = g_gate ** tau_gate
    rt_gate_log = (-g_gate) * tau_gate
    q = torch.randn(B, T, MEM).cuda()
    k = torch.nn.functional.normalize(torch.randn(B, T, MEM, device="cuda").float(), dim=-1) * it_gate
    v = torch.randn(B, T, D_INNER).cuda()

    layer = RodimusLayer(D_INNER, MEM).cuda()
    out = layer._run_mixer({
        "query": q.unsqueeze(1), "key": k.unsqueeze(1), "value": v.unsqueeze(1),
        "beta": torch.ones(B, 1, T, device="cuda"),
        "log_decay": rt_gate_log.unsqueeze(1),
        "initial_state": torch.zeros(B, 1, MEM, D_INNER, device="cuda"),
    })["output"].squeeze(1)

    fused = fla_op("fla.ops.gla.fused_recurrent_gla")
    # Pinned fused op: [B, T, H, *] with H=1, scale=1 (scale_rule="one").
    expected, _ = fused(
        q=q.unsqueeze(2), k=k.unsqueeze(2), v=v.unsqueeze(2),
        gk=rt_gate_log.unsqueeze(2), scale=1, output_final_state=True,
        state_v_first=True,
    )
    err = (out - expected.squeeze(2)).abs().max().item()
    assert err < 2e-3, f"rodimus mixer parity: max abs err {err}"


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = RodimusLayer(D_INNER, MEM)
    hidden = torch.randn(2, 6, D_INNER)
    out = layer(hidden)
    assert out.shape == (2, 6, D_INNER)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.g_gate_proj.weight.grad is not None
    assert layer.k_proj.weight.grad is not None
