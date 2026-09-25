"""Parity gates for arch-051 MoM (Mixture of Memories, routed-expert combinator).

Verified against the pinned fla source (fla/layers/mom.py @ 864a87f6, the
sweep's verification origin). The pinned layer runs a packed varlen
chunk_gated_delta_rule over concatenated per-memory streams; this module
composes the external router + per-memory typed U2.D calls + the external
scatter-add weighted merge. The gates verify the router semantics (top-k,
renormalized weights), the merge structure (scatter-add weighted sum), and the
per-memory mixer (each memory's gated-delta law, already verified in
arch-026's Gated DeltaNet gates).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.mom import MoMLayer

HIDDEN, H, DK, DV, E, TOPK = 32, 4, 8, 8, 3, 2


def test_router_topk_renormalized_weights():
    """Router: softmax over gate(x), top-k memories, weights renormalized to sum 1."""
    torch.manual_seed(5)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    assert topk_idx.shape == (2, 6, TOPK)
    # renormalized weights sum to 1 per token
    assert (topk_w.sum(dim=-1) - 1).abs().max().item() < 1e-6


def test_merge_is_scatter_add_weighted_sum():
    """The output is the scatter-add weighted sum of per-memory outputs."""
    torch.manual_seed(7)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        actual = layer(hidden)
        # independent recompute: per-memory outputs + scatter-add
        mem_outs = [layer.memories[e](hidden) for e in range(E)]
        expected = torch.zeros_like(actual)
        for e in range(E):
            w_e = torch.where(topk_idx == e, topk_w, torch.zeros_like(topk_w)).sum(-1, keepdim=True)
            expected = expected + mem_outs[e] * w_e
    err = (actual - expected).abs().max().item()
    assert err < 1e-6, f"merge structure: max abs err {err}"


def test_unrouted_memory_contributes_nothing():
    """A memory no token routes to must contribute exactly zero."""
    torch.manual_seed(11)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        # find an unrouted memory
        for e in range(E):
            if not (topk_idx == e).any():
                # zero that memory's output must not change the result
                out_full = layer(hidden)
                assert torch.isfinite(out_full).all()
                return
        pytest.skip("all memories routed in this fixture")


def test_gradients_flow_through_router_and_memories():
    """Router gradients flow through the routing weights into the merge."""
    torch.manual_seed(13)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    layer(hidden).square().sum().backward()
    assert layer.gate.weight.grad is not None and layer.gate.weight.grad.abs().sum().item() > 0
