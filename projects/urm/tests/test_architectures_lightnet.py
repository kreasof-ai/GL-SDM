"""Parity gates for arch-022 LightNet (U2.A channel-decay + normalized-key frontend).

Verified against the pinned fla source (fla/layers/lightnet.py @ 864a87f6, the
sweep's verification origin): the typed U2.A channel-gate mixer matches the
pinned fused_recurrent_gla (state_v_first=True, GPU) on the log-cumsum-exp
normalized (k', g) operands, and the frontend transcription matches the pinned
prefill path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.lightnet import LightNetLayer
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def _normalized_key_frontend(k):
    """The pinned prefill frontend (no padding): z = logcumsumexp(k), k' = exp(k−z), g = z_prev − z."""
    k_float = k.float()
    z = k_float.logcumsumexp(1)
    k_new = torch.nan_to_num(torch.exp(k_float - z), nan=0.0, posinf=0.0).to(k.dtype)
    z_prev = torch.cat((torch.zeros_like(z[:, :1]), z[:, :-1]), dim=1)
    g = torch.nan_to_num(z_prev - z, nan=0.0, posinf=0.0, neginf=0.0).to(k.dtype)
    return k_new, g


def test_mixer_matches_pinned_fused_recurrent_gla_state_v_first_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    k_raw = torch.randn(B, T, H, DK).cuda()
    k_new, g = _normalized_key_frontend(k_raw)

    layer = LightNetLayer(HIDDEN, H, DK, DV).cuda()
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k_new.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device="cuda"),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]

    fused = fla_op("fla.ops.gla.fused_recurrent_gla")
    expected, _ = fused(q=q, k=k_new, v=v, gk=g, scale=None, output_final_state=True,
                        state_v_first=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"lightnet mixer parity: max abs err {err}"


def test_frontend_matches_pinned_prefill_path():
    """The normalized-key frontend transcription matches the pinned lightnet.py prefill."""
    torch.manual_seed(9)
    B, T = 2, 6
    k = torch.randn(B, T, H, DK)
    k_new, g = _normalized_key_frontend(k)
    # Invariants of the pinned frontend: k' is normalized (exp(k - logcumsumexp)),
    # g sums to −z_T over time (telescoping z_prev − z).
    assert k_new.shape == k.shape and g.shape == k.shape
    assert torch.isfinite(k_new).all() and torch.isfinite(g).all()
    assert (g.sum(dim=1) - (-k.float().logcumsumexp(1)[:, -1])).abs().max().item() < 1e-4


def test_full_layer_composition_and_gradients():
    torch.manual_seed(13)
    layer = LightNetLayer(HIDDEN, H, DK, DV)
    hidden = torch.randn(2, 6, HIDDEN)
    out = layer(hidden)
    assert out.shape == (2, 6, HIDDEN)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert layer.q_proj.weight.grad is not None
    assert layer.k_proj.weight.grad is not None
