"""Parity gates for arch-004 MLA (Multi-head Latent Attention).

Verified against the pinned fla equation (fla/layers/mla.py @ 864a87f6, the
sweep's verification origin) via the fp32 transcription oracle: latent
projection + RMSNorm + per-head expansion + shared rope key, with the causal
U1.S mixer over the expanded heads, on identical parameters. Compressed-cache
equality (the pinned TODO) is the recorded blocker, not claimed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.mla_attention import MLALayer
from benchmarks.comparators.fla_mla import fla_mla_oracle

CFG = dict(
    hidden_size=32, num_heads=4, kv_lora_rank=8, qk_rope_head_dim=8,
    v_head_dim=8, qk_nope_head_dim=8,
)


def _build(seed: int, q_lora_rank=None):
    torch.manual_seed(seed)
    cfg = dict(CFG, q_lora_rank=q_lora_rank)
    urm = MLALayer(**cfg)
    hidden = torch.randn(2, 6, CFG["hidden_size"])
    return urm, hidden


def _parity(seed: int, q_lora_rank=None):
    urm, hidden = _build(seed, q_lora_rank)
    expected, _id = fla_mla_oracle(
        hidden,
        q_proj=urm.q_proj, k_rope=urm.k_rope, kv_proj=urm.kv_proj,
        o_proj=urm.o_proj, num_heads=CFG["num_heads"],
        qk_rope_head_dim=CFG["qk_rope_head_dim"],
        qk_nope_head_dim=CFG["qk_nope_head_dim"],
        v_head_dim=CFG["v_head_dim"],
    )
    with torch.no_grad():
        actual = urm(hidden)
    return (actual - expected).abs().max().item()


def test_layer_matches_pinned_mla_equation():
    err = _parity(seed=5)
    assert err < 1e-4, f"layer parity: max abs err {err}"


def test_layer_matches_pinned_mla_with_q_lora():
    err = _parity(seed=9, q_lora_rank=8)
    assert err < 1e-4, f"q-lora parity: max abs err {err}"


def test_gradients_flow_through_latent_projections():
    urm, hidden = _build(seed=13)
    urm(hidden).square().sum().backward()
    for name, param in urm.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
    assert urm.kv_proj[0].weight.grad.abs().sum().item() > 0
    assert urm.o_proj.weight.grad.abs().sum().item() > 0
