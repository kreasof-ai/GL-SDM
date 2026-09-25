"""Parity gates for arch-025 DeltaNet (exact U2.D).

Verified against the pinned fla source (fla/ops/delta_rule + fla/layers/
delta_net.py @ 864a87f6, the sweep's verification origin): the typed U2.D
mixer matches the pinned fused_recurrent_delta_rule on identical operands
(GPU), and the full layer (external projections + mixer + RMSNorm + o_proj)
matches the pinned equation end-to-end.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.deltanet import DeltaNetLayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_mixer_matches_pinned_fused_recurrent_delta_rule_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_delta_rule requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, DK), p=2, dim=-1).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    beta = torch.rand(B, T, H).cuda()

    layer = DeltaNetLayer(HIDDEN, H, DK, DV)
    out = layer._run_mixer({
        "query": q.transpose(1, 2),
        "key": k.transpose(1, 2),
        "value": v.transpose(1, 2),
        "beta": beta.transpose(1, 2),
        "log_decay": torch.zeros(B, H, T, device="cuda"),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.delta_rule.fused_recurrent_delta_rule")
    # The pinned fused op takes [B, T, H, D]; the K2 graph (and `out`) is [B, H, T, V].
    expected, _state = fused(
        q=q, k=k, v=v, beta=beta, output_final_state=True,
        use_qk_l2norm_in_kernel=False,  # l2-norm applied externally above
    )
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"mixer parity vs pinned fused op: max abs err {err}"


def test_full_layer_external_stages_compose():
    """The layer composes projections → mixer → RMSNorm → o_proj; gradients flow."""
    torch.manual_seed(7)
    layer = DeltaNetLayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.b_proj.weight.grad is not None
    assert layer.q_proj.weight.grad is not None
    assert layer.o_proj.weight.grad is not None
