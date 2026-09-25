"""Parity gates for arch-020 Based and arch-021 ReBased (U2.A normalized linear
attention after an external feature map).

Verified against the pinned fla source (fla/layers/based.py + rebased.py +
modules/feature_map.py @ 864a87f6, the sweep's verification origin): the typed
normalized U2.A mixer matches the pinned fused_recurrent_linear_attn
(normalize=True, scale=1) on feature-mapped operands (GPU).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.based_attention import BasedLayer, ReBasedLayer, rebased_feature_map, taylor_feature_map
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def _mixer_out(layer, q, k, v):
    B, T = q.shape[0], q.shape[1]
    dk_fm = q.shape[-1]
    return layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device=q.device),
        "log_decay": torch.zeros(B, H, T, device=q.device),
        "initial_state": torch.zeros(B, H, dk_fm, DV, device=q.device),
        "scale": torch.ones((), dtype=torch.float32, device=q.device),
    })["output"]


@pytest.mark.parametrize("name,layer_cls,fm", (
    ("based", BasedLayer, taylor_feature_map),
    ("rebased", ReBasedLayer, rebased_feature_map),
))
def test_mixer_matches_pinned_normalized_linear_attn_cuda(name, layer_cls, fm):
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_linear_attn requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = fm(torch.randn(B, T, H, DK)).cuda()
    k = fm(torch.randn(B, T, H, DK)).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    layer = layer_cls(HIDDEN, H, DK, DV).cuda()
    out = _mixer_out(layer, q, k, v)
    fused = fla_op("fla.ops.linear_attn.fused_recurrent_linear_attn")
    expected, _ = fused(q=q, k=k, v=v, scale=1.0, output_final_state=True, normalize=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"{name} parity: max abs err {err}"


@pytest.mark.parametrize("layer_cls", (BasedLayer, ReBasedLayer))
def test_full_layer_composition_and_gradients(layer_cls):
    torch.manual_seed(7)
    layer = layer_cls(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.q_proj.weight.grad is not None
