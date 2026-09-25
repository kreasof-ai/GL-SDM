"""Parity gates for arch-066 CAT decoder attention.

Verified against the pinned fla CAT source (fla/models/cat/modeling_cat.py @
864a87f6, the sweep's verification origin): the decoder mixer (U1.S over the
CAT structural mask, GQA-capable) matches the pinned mask equation evaluated
eagerly, and the structural mask itself matches the pinned
get_cat_mask_mod predicate cell-for-cell.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.cat_attention import CATDecoderAttention, cat_structural_mask
from benchmarks.comparators.cat_attention import cat_attention_oracle, pinned_cat_mask

NUM_HEADS, HIDDEN, BLOCK = 4, 32, 4


def test_structural_mask_matches_pinned_predicate():
    """cat_structural_mask == get_cat_mask_mod, cell for cell."""
    T = 11
    mask = cat_structural_mask(T, BLOCK)
    for q in range(T):
        for kv in range(T):
            assert mask[q, kv].item() == pinned_cat_mask(0, 0, q, kv, block_size=BLOCK)


def _build_pair(seed: int, num_kv_heads: int, T: int = 9):
    torch.manual_seed(seed)
    module = CATDecoderAttention(HIDDEN, NUM_HEADS, BLOCK, num_kv_heads=num_kv_heads)
    hidden = torch.randn(2, T, HIDDEN)
    return module, hidden


def _layer_parity(num_kv_heads: int, seed: int):
    module, hidden = _build_pair(seed, num_kv_heads)
    with torch.no_grad():
        q = module.q_proj(hidden).view(2, hidden.shape[1], NUM_HEADS, module.head_dim)
        k = module.k_proj(hidden).view(2, hidden.shape[1], num_kv_heads, module.head_dim)
        v = module.v_proj(hidden).view(2, hidden.shape[1], num_kv_heads, module.head_dim)
        expected, _id = cat_attention_oracle(q, k, v, block_size=BLOCK)
        expected = module.o_proj(expected.reshape(2, hidden.shape[1], -1))
        actual = module(hidden)
    return (actual - expected).abs().max().item()


def test_layer_matches_pinned_equation_mha():
    err = _layer_parity(num_kv_heads=NUM_HEADS, seed=5)
    assert err < 1e-5, f"mha parity: max abs err {err}"


def test_layer_matches_pinned_equation_gqa():
    err = _layer_parity(num_kv_heads=2, seed=7)
    assert err < 1e-5, f"gqa parity: max abs err {err}"


def test_compressed_tokens_visible_across_blocks():
    """A query in a later block attends to all prior compressed tokens
    (kv % block == 0) plus its local block — the CAT structure, not plain causal."""
    T, block = 9, BLOCK
    mask = cat_structural_mask(T, block)
    # Query 8 (block 2): sees local block {8} and compressed tokens {0,4,8}.
    visible = mask[8].nonzero().flatten().tolist()
    assert visible == [0, 4, 8]
    # Plain causal would also see 1,2,3,5,6,7 — CAT does not.
    assert 1 not in visible and 5 not in visible


def test_gradients_flow_through_all_projections():
    module, hidden = _build_pair(seed=13, num_kv_heads=2)
    module(hidden).square().sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} gradient is exactly zero"
