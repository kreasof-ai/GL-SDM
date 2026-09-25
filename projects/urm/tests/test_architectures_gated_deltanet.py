"""Parity gates for arch-026 Gated DeltaNet (U2.D + head-scalar decay).

Verified against the pinned fla source (fla/ops/gated_delta_rule + layers @
864a87f6, the sweep's verification origin): the typed U2.D head-gate mixer
matches the pinned fused_recurrent_gated_delta_rule (GPU), with the gate value
g = −exp(A_log)·softplus(g_in + dt_bias) computed externally.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.gated_deltanet import GatedDeltaNetLayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_gated_delta_rule_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_gated_delta_rule requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    beta = torch.rand(B, T, H).cuda()
    g_in = torch.randn(B, T, H).cuda()
    A_log = torch.randn(H).cuda()
    dt_bias = torch.randn(H).cuda()
    g = -torch.exp(A_log) * torch.nn.functional.softplus(g_in + dt_bias)

    layer = GatedDeltaNetLayer(HIDDEN, H, DK, DV).cuda()
    with torch.no_grad():
        layer.A_log.copy_(A_log)
        layer.dt_bias.copy_(dt_bias)
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": beta.transpose(1, 2),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule")
    # Pinned op takes g as the per-head scalar log-decay (computed externally above).
    expected, _ = fused(
        q=q, k=k, v=v, g=g, beta=beta, scale=None, output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"gated_deltanet parity: max abs err {err}"


def test_gate_value_matches_external_schedule():
    """g = −exp(A_log)·softplus(g_in + dt_bias), computed externally."""
    torch.manual_seed(11)
    layer = GatedDeltaNetLayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    g = layer._gate(hidden)
    g_in = layer.g_proj(hidden)
    expected = -torch.exp(layer.A_log) * torch.nn.functional.softplus(g_in + layer.dt_bias)
    assert (g - expected).abs().max().item() < 1e-6
    assert (g <= 0).all()


def test_full_layer_composition_and_gradients():
    torch.manual_seed(7)
    layer = GatedDeltaNetLayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    out.square().sum().backward()
    assert layer.A_log.grad is not None
    assert layer.q_proj.weight.grad is not None
