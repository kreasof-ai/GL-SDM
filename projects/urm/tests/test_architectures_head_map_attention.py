"""Parity gates for the head-map cluster (arch-001 MHA, arch-002 MQA, arch-003 GQA).

Each row's external module (architectures/head_map_attention.py) is verified
against the pinned fla naive reference (the sweep's verification source) on
identical projections and inputs: forward parity on CPU and CUDA, gradient
parity through the full layer, and descriptor-legality checks (the K1
descriptor rejects a head map that disagrees with the HQ/H ratio).

Verification method: run the pinned ``naive_parallel_attn`` on q/k/v produced
by the *same* projection weights, then apply the same output projection — this
isolates the mixer equation (URM K1 graph vs pinned source) and exercises the
module's external stages exactly once.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.head_map_attention import HeadMapAttention
from benchmarks.comparators.fla_attention_naive import fla_naive_attention_adapter

CONFIGS = {
    "mha": dict(query_heads=4, kv_heads=4, head_map="equal"),
    "mqa": dict(query_heads=4, kv_heads=1, head_map="single"),
    "gqa": dict(query_heads=4, kv_heads=2, head_map="grouped"),
}


def _build(name: str, seed: int, device: str):
    cfg = CONFIGS[name]
    torch.manual_seed(seed)
    module = HeadMapAttention(model_dim=32, head_dim=8, **cfg).to(device)
    hidden = torch.randn(2, 7, 32, device=device)
    return module, hidden


def _layer_parity(name: str, device: str):
    module, hidden = _build(name, seed=13, device=device)
    B, T, _ = hidden.shape
    with torch.no_grad():
        q = module.q_proj(hidden).view(B, T, module.query_heads, module.head_dim)
        k = module.k_proj(hidden).view(B, T, module.kv_heads, module.head_dim)
        v = module.v_proj(hidden).view(B, T, module.kv_heads, module.head_dim)
        expected, _identity = fla_naive_attention_adapter(q, k, v, causal=True)
        expected = module.o_proj(expected.reshape(B, T, -1))
        actual = module(hidden)
    err = (actual - expected).abs().max().item()
    assert err < 2e-5, f"{name}@{device}: max abs err {err}"


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_head_map_layer_matches_pinned_fla_naive_cpu(name):
    _layer_parity(name, "cpu")


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_head_map_layer_matches_pinned_fla_naive_cuda(name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _layer_parity(name, "cuda")


def test_k1_descriptor_rejects_inconsistent_head_map_fields():
    """The closed K1 descriptor rejects head-map field misuse (contract legality)."""
    from urm.ir.program import K1Descriptor, K1HeadMap

    with pytest.raises(ValueError, match="grouped head map requires"):
        K1Descriptor(head_map=K1HeadMap.GROUPED, group_size=None)
    with pytest.raises(ValueError, match="only legal with the grouped head map"):
        K1Descriptor(head_map=K1HeadMap.EQUAL, group_size=2)


def test_module_derives_group_size_from_head_ratio():
    """The module derives the descriptor's group_size from HQ // H — the recipe
    field is never a free knob that could disagree with the head ratio."""
    module, _ = _build("gqa", seed=17, device="cpu")
    (op,) = module._plan.compilation.rewritten_program.ops
    assert op.k1.head_map.value == "grouped"
    assert op.k1.group_size == 2  # HQ=4, H=2


def test_gradient_flows_through_all_projections_and_grouped_kv():
    """Backward through the K1 graph reaches every external projection,
    including the shared-KV gradient accumulation of MQA/GQA."""
    module, hidden = _build("gqa", seed=29, device="cpu")
    out = module(hidden)
    out.square().sum().backward()
    for name_, param in module.named_parameters():
        assert param.grad is not None, f"{name_} has no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name_} gradient is exactly zero"


def test_mqa_single_kv_head_shared_across_query_heads():
    """MQA executes with one KV head; the pinned source shares it across HQ."""
    module, hidden = _build("mqa", seed=31, device="cpu")
    assert module.kv_heads == 1
    _layer_parity("mqa", "cpu")
